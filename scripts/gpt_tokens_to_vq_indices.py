import argparse
import gzip
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import torch


def load_pickle(path: str) -> List[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(obj: Any, path: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "wb"
    with opener(path, mode) as f:
        pickle.dump(obj, f)


GLOBAL_RE = re.compile(r"^<GLOBAL_CODE_(\d+)>$")
LOCAL_RE = re.compile(r"^<LOCAL_CODE_(\d+)>$")
PROPERTY_BIN_RE = re.compile(r"^<PROPERTY_BIN_(\d+)>$")
SITE_START_RE = re.compile(r"^<SITE_(\d+)>$")
SITE_END_RE = re.compile(r"^</SITE_(\d+)>$")


def parse_vq_indices_from_tokens(tokens: List[str]):
    """
    从 token 序列中解析全局 / 局部 VQ 索引：
      - 全局：<GLOBAL> ... <GLOBAL_CODE_x> ... </GLOBAL>
      - 局部：<SITES> <SITE_i> ... <LOCAL_CODE_x> ... </SITE_i> ... </SITES>

    返回：
      global_codes: List[int]
      local_codes_per_site: Dict[int, List[int]]  (site_index -> [codes])
      property_bin: Optional[int]
    """
    global_codes: List[int] = []
    local_codes_per_site: Dict[int, List[int]] = {}
    property_bin = None

    for tok in tokens:
        m_prop = PROPERTY_BIN_RE.match(tok)
        if m_prop:
            property_bin = int(m_prop.group(1))
            break

    in_global = False
    for tok in tokens:
        if tok == "<GLOBAL>":
            in_global = True
            continue
        if tok == "</GLOBAL>":
            in_global = False
        if in_global:
            m = GLOBAL_RE.match(tok)
            if m:
                global_codes.append(int(m.group(1)))

    current_site = None
    for tok in tokens:
        m_start = SITE_START_RE.match(tok)
        if m_start:
            current_site = int(m_start.group(1))
            if current_site not in local_codes_per_site:
                local_codes_per_site[current_site] = []
            continue
        m_end = SITE_END_RE.match(tok)
        if m_end:
            current_site = None
            continue
        if current_site is not None:
            m_local = LOCAL_RE.match(tok)
            if m_local:
                local_codes_per_site[current_site].append(int(m_local.group(1)))

    return global_codes, local_codes_per_site, property_bin


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse <GLOBAL_CODE_x>/<LOCAL_CODE_x> from GPT-generated token sequences "
            "and save discrete VQ indices for later training use."
        )
    )
    parser.add_argument(
        "--tokens_path",
        type=str,
        required=True,
        help="Path to generated token sequences (pkl or pkl.gz), "
             "produced by scripts/generate_tokens_from_prompts.py.",
    )
    parser.add_argument(
        "--vq_checkpoint",
        type=str,
        required=False,
        help="(Optional) Path to pretrained CrystalVQVAE checkpoint (.pth). "
             "If provided, script will also compute z_g/z_l vectors.",
    )
    parser.add_argument(
        "--vq_config",
        type=str,
        required=False,
        help="(Optional) VQ-VAE yaml config used at training time. Required if --vq_checkpoint is set.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/vq_indices/gpt_tokens_indices.pkl.gz",
        help="Output path to save parsed indices (and optional VQ vectors) as pkl(.gz).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only process first N samples (for debugging).",
    )
    parser.add_argument(
        "--expected_local_codes",
        type=int,
        default=0,
        help="If >0, enforce each <SITE_i> has exactly this many <LOCAL_CODE_x> tokens (pad/trunc disabled; mismatch raises).",
    )
    parser.add_argument(
        "--expected_global_codes",
        type=int,
        default=0,
        help="If >0, enforce <GLOBAL> contains exactly this many <GLOBAL_CODE_x> tokens (pad/trunc disabled; mismatch raises).",
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="If set, skip invalid/ragged samples instead of raising (recommended for large-scale filtering).",
    )
    parser.add_argument(
        "--save_skipped_ids",
        type=str,
        default="",
        help="If set, save skipped sample ids to this text file (one id per line).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens_path = Path(args.tokens_path)
    assert tokens_path.is_file(), f"tokens_path not found: {tokens_path}"

    project_root = Path(__file__).resolve().parent.parent
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    vq_model = None
    if args.vq_checkpoint:
        assert args.vq_config, "--vq_config is required when --vq_checkpoint is set"
        from scripts.cif2token import load_yaml_config, build_model_from_config

        config = load_yaml_config(args.vq_config)
        checkpoint = torch.load(args.vq_checkpoint, map_location=device, weights_only=False)
        vq_model = build_model_from_config(config, checkpoint, device)
        vq_model.eval()

    print(f"Loading GPT-generated tokens from {tokens_path}")
    samples = load_pickle(str(tokens_path))

    results: List[Dict[str, Any]] = []
    skipped_ids: List[str] = []
    for idx, sample in enumerate(samples):
        if args.max_samples and args.max_samples > 0 and idx >= args.max_samples:
            break

        raw_id = sample.get("id", None)
        prompt_id = sample.get("prompt_id", None)
        # 统一使用 base_id 作为 key（与 CSV/material_id 对齐）
        if prompt_id is not None:
            sample_id = str(prompt_id)
        else:
            # 兼容旧格式："mp-31117__gen_0" -> "mp-31117"
            if isinstance(raw_id, str) and "__gen_" in raw_id:
                sample_id = raw_id.split("__gen_")[0]
            else:
                sample_id = str(raw_id) if raw_id is not None else None
        tokens = sample.get("tokens", None)
        if not tokens or not isinstance(tokens, list):
            continue

        global_codes, local_codes_per_site, property_bin = parse_vq_indices_from_tokens(tokens)
        if len(global_codes) == 0 or len(local_codes_per_site) == 0:
            # 没有找到任何 code token，跳过
            continue

        # 这里只保存离散索引，形状遵循 (1, num_global_codes) / (num_sites, num_local_codes_per_site)
        if args.expected_global_codes and args.expected_global_codes > 0:
            if len(global_codes) != args.expected_global_codes:
                msg = f"id={sample_id}: expected {args.expected_global_codes} global codes, got {len(global_codes)}"
                if args.skip_invalid:
                    skipped_ids.append(str(sample_id))
                    continue
                raise ValueError(msg)
        indices_g = torch.tensor(global_codes, dtype=torch.long, device=device).unsqueeze(0)  # (1, G)

        site_indices = sorted(local_codes_per_site.keys())
        local_rows: List[List[int]] = []
        inferred_L = None
        for s_idx in site_indices:
            codes_s = local_codes_per_site[s_idx]
            if not codes_s:
                # 缺失该 site 的 local codes：这会导致 ragged list，直接报错更利于定位
                msg = f"id={sample_id}: site={s_idx} has no <LOCAL_CODE_x> tokens"
                if args.skip_invalid:
                    skipped_ids.append(str(sample_id))
                    local_rows = []
                    break
                raise ValueError(msg)

            if inferred_L is None:
                inferred_L = len(codes_s)
            if args.expected_local_codes and args.expected_local_codes > 0:
                if len(codes_s) != args.expected_local_codes:
                    msg = (
                        f"id={sample_id}: site={s_idx} expected {args.expected_local_codes} local codes, got {len(codes_s)}"
                    )
                    if args.skip_invalid:
                        skipped_ids.append(str(sample_id))
                        local_rows = []
                        break
                    raise ValueError(msg)
            else:
                # 未显式给 expected_local_codes：用第一个 site 推断，并要求所有 site 一致
                if len(codes_s) != inferred_L:
                    msg = (
                        f"id={sample_id}: ragged local codes detected at site={s_idx}: "
                        f"expected {inferred_L}, got {len(codes_s)}"
                    )
                    if args.skip_invalid:
                        skipped_ids.append(str(sample_id))
                        local_rows = []
                        break
                    raise ValueError(msg)
            local_rows.append(codes_s)
        if not local_rows:
            continue
        indices_l = torch.tensor(local_rows, dtype=torch.long, device=device)  # (S, L)

        rec: Dict[str, Any] = {
            "id": sample_id,          # base id（通常是 material_id）
            "gen_id": raw_id,         # 原始生成 id（例如 mp-31117__gen_0）
            "indices_g": indices_g.cpu(),
            "indices_l": indices_l.cpu(),
            "property_bin": property_bin,
            "tokens": tokens,
        }

        # 可选：计算 VQ 连续向量
        if vq_model is not None:
            with torch.no_grad():
                z_g = vq_model.get_att_agg_output_from_indices(
                    indices_g.to(device), is_global=True, use_aggregation=True
                )
                z_l = vq_model.get_att_agg_output_from_indices(
                    indices_l.to(device), is_global=False, use_aggregation=True
                )
            rec["z_g"] = z_g.cpu()
            rec["z_l"] = z_l.cpu()

        results.append(rec)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(results, str(out_path))
    print(
        f"[Saved] parsed indices for {len(results)} GPT samples "
        f"-> {out_path} (source={tokens_path})"
    )
    if skipped_ids:
        # 去重但保留顺序
        seen = set()
        uniq = []
        for x in skipped_ids:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        print(f"[Stats] skipped_invalid={len(uniq)}")
        if args.save_skipped_ids:
            p = Path(args.save_skipped_ids)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(uniq) + "\n", encoding="utf-8")
            print(f"[Saved] skipped_ids -> {p}")


if __name__ == "__main__":
    # python scripts/gpt_tokens_to_vq_indices.py --tokens_path dataset/mpts_52/gen_DSS/test_generated.pkl.gz --out_path dataset/mpts_52/DSS2indices/test.pkl.gz --expected_global_codes 2 --expected_local_codes 4 --skip_invalid --save_skipped_ids dataset/mpts_52/DSS2indices/test_skipped_ids.txt
    # # 或者附带 VQ 连续向量：
    # # --vq_checkpoint ckpts/mpts_52/attn_rvq/best_model.pth \
    # # --vq_config config/data/train_cryvqvae_mpts_52.yaml

    # python scripts/gpt_tokens_to_vq_indices.py --tokens_path dataset/mp_20/gen_DSS/train_generated.pkl.gz --out_path dataset/mp_20/DSS2indices/train.pkl.gz --expected_global_codes 2 --expected_local_codes 4 --skip_invalid --save_skipped_ids dataset/mp_20/DSS2indices/train_skipped_ids.txt
    # python scripts/gpt_tokens_to_vq_indices.py --tokens_path dataset/mp_20/gen_DSS/val_generated.pkl.gz --out_path dataset/mp_20/DSS2indices/val.pkl.gz --expected_global_codes 2 --expected_local_codes 4 --skip_invalid --save_skipped_ids dataset/mp_20/DSS2indices/val_skipped_ids.txt


    # python scripts/gpt_tokens_to_vq_indices.py --tokens_path dataset/perov_5/gen_DSS/val_generated.pkl.gz --out_path dataset/perov_5/DSS2indices/val.pkl.gz --expected_global_codes 2 --expected_local_codes 4 --skip_invalid --save_skipped_ids dataset/perov_5/DSS2indices/val_skipped_ids.txt
    # python scripts/gpt_tokens_to_vq_indices.py --tokens_path dataset/perov_5/gen_DSS/train_generated.pkl.gz --out_path dataset/perov_5/DSS2indices/train.pkl.gz --expected_global_codes 2 --expected_local_codes 4 --skip_invalid --save_skipped_ids dataset/perov_5/DSS2indices/train_skipped_ids.txt
    # python scripts/gpt_tokens_to_vq_indices.py --tokens_path dataset/perov_5/gen_DSS/test_generated.pkl.gz --out_path dataset/perov_5/DSS2indices/test.pkl.gz --expected_global_codes 2 --expected_local_codes 4 --skip_invalid --save_skipped_ids dataset/perov_5/DSS2indices/test_skipped_ids.txt
    main()

