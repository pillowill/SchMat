import argparse
import gzip
import json
import pickle
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm


def save_pickle(obj: Any, path: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "wb"
    with opener(path, mode) as f:
        pickle.dump(obj, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Load pretrained crystal_token_gpt and generate DSS-seq from prompts or random hyperparameter-controlled prompts.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_cry2seq.pt")
    parser.add_argument("--prompts_path", type=str, default="", help="Prompt file path (.jsonl) from make_prompts_from_csv.py")
    parser.add_argument("--out_path", type=str, default="results/gpt_tokens/generated_from_prompts.pkl.gz", help="Output pkl(.gz)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate after prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling (0 disables)")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="How many sequences per prompt")
    parser.add_argument("--num_generations", type=int, default=0, help="If >0, generate this exact number of DSS-seqs (useful in --random_mode).")
    parser.add_argument("--max_prompts", type=int, default=0, help="If >0, limit number of prompts (debug)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--random_mode", action="store_true", help="Enable random generation mode without prompt file.")
    parser.add_argument("--conditioning_mode", type=str, default="none", choices=["none", "property"], help="Conditioning mode for random prompts.")
    parser.add_argument("--property_name", type=str, default="band_gap", help="Property name token suffix when conditioning_mode=property.")
    parser.add_argument("--property_bin", type=int, default=-1, help="Discrete property bin id when conditioning_mode=property.")
    parser.add_argument("--property_value", type=float, default=None, help="Continuous property value; if set, converted to bin by bin_size/origin.")
    parser.add_argument("--property_bin_size", type=float, default=0.5, help="Bin size when converting property_value to property_bin.")
    parser.add_argument("--property_bin_origin", type=float, default=0.0, help="Bin origin when converting property_value to property_bin.")
    parser.add_argument("--allow_unk", action="store_true", help="Allow sampling <UNK>. Default is to suppress <UNK> for cleaner DSS outputs.")
    parser.add_argument("--early_stop_eos", action="store_true", help="Stop early when all sampled sequences hit <EOS>.")
    parser.add_argument("--use_smact_filter", action="store_true", help="Filter generated sequences by SMACT charge neutrality.")
    return parser.parse_args()


def sanitize_token_name(name: str) -> str:
    s = str(name).strip().upper()
    out = []
    prev_underscore = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    return ("".join(out).strip("_") or "UNKNOWN")


def to_property_bin(value: float, bin_size: float, origin: float) -> int:
    if bin_size <= 0:
        raise ValueError(f"property_bin_size must be > 0, got {bin_size}")
    idx = int(math.floor((float(value) - float(origin)) / float(bin_size)))
    return max(0, idx)


def build_random_prompt_tokens(args) -> List[str]:
    tokens = ["<BOS>", "<CRYSTAL>"]
    if args.conditioning_mode == "property":
        if args.property_value is not None:
            prop_bin = to_property_bin(args.property_value, args.property_bin_size, args.property_bin_origin)
        elif args.property_bin >= 0:
            prop_bin = int(args.property_bin)
        else:
            raise ValueError("conditioning_mode=property requires --property_bin>=0 or --property_value")

        prop_name = sanitize_token_name(args.property_name)
        tokens.extend([
            "<PROPERTY>",
            f"<PROPERTY_NAME_{prop_name}>",
            f"<PROPERTY_BIN_{prop_bin}>",
            "</PROPERTY>",
        ])
    return tokens


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    project_root = Path(__file__).resolve().parent.parent
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from model.crystal_token_gpt import GPT, GPTConfig
    from scripts.tokenizer import CrystalVocabConfig, CrystalVocab, CrystalTokenizer

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    vocab_cfg = CrystalVocabConfig(**ckpt["vocab_config"])
    vocab = CrystalVocab(vocab_cfg)
    tokenizer = CrystalTokenizer(vocab)

    from scripts.evaluate_dss_composition_validity import parse_formula_from_tokens, reduce_stoich, smact_validity

    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    unk_id = vocab.token_to_id.get("<UNK>", None)
    eos_id = vocab.token_to_id.get("<EOS>", None)
    forbidden_ids = [] if args.allow_unk else ([unk_id] if unk_id is not None else [])

    top_k: Optional[int] = args.top_k if args.top_k and args.top_k > 0 else None

    results: List[Dict[str, Any]] = []
    n_read = 0

    if args.random_mode:
        prompt_tokens = build_random_prompt_tokens(args)
        for t in prompt_tokens:
            if t not in vocab.token_to_id:
                raise ValueError(f"Token '{t}' is not in checkpoint vocabulary. Please use a compatible cry2seq checkpoint.")

        target_n = args.num_generations if args.num_generations > 0 else 10000
        batch_size = max(1, int(args.num_samples_per_prompt))
        
        pbar = tqdm(desc="Generating (random_mode)", total=target_n, unit="gen")
        batch_idx = 0
        while len(results) < target_n:
            prompt_id = f"random_prompt_{batch_idx}"
            prompt_ids = tokenizer.encode(prompt_tokens)
            prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

            gen_ids = model.generate(
                idx=prompt_ids_t,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                num_gens=1,
                forbidden_token_ids=forbidden_ids,
                eos_token_id=eos_id,
                early_stop_eos=args.early_stop_eos,
            )

            for j in range(gen_ids.size(0)):
                ids_j = gen_ids[j].tolist()
                tokens_j = tokenizer.decode(ids_j)
                if "<EOS>" in tokens_j:
                    tokens_j = tokens_j[: tokens_j.index("<EOS>") + 1]

                if args.use_smact_filter:
                    comp = parse_formula_from_tokens(tokens_j)
                    comp = reduce_stoich(comp)
                    elems = [e for e, _ in comp]
                    counts = [int(c) for _, c in comp]
                    if not elems:
                        continue
                    try:
                        is_valid = smact_validity(elems, counts, use_pauling_test=True, include_alloys=True)
                    except Exception:
                        is_valid = False
                    if not is_valid:
                        continue

                results.append(
                    {
                        "id": f"{prompt_id}__gen_{j}",
                        "prompt_id": prompt_id,
                        "formula": None,
                        "prompt_tokens": prompt_tokens,
                        "tokens": tokens_j,
                    }
                )

                pbar.update(1)
                if len(results) >= target_n:
                    break
            
            n_read += batch_size
            batch_idx += 1

        pbar.close()

    else:
        prompt_records: List[Dict[str, Any]] = []
        prompts_path = Path(args.prompts_path)
        assert prompts_path.is_file(), f"Prompts file not found: {prompts_path}"
        with prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompt_tokens = rec.get("prompt_tokens", None)
                if not isinstance(prompt_tokens, list) or len(prompt_tokens) == 0:
                    continue
                prompt_records.append(rec)
                if args.max_prompts and args.max_prompts > 0 and len(prompt_records) >= args.max_prompts:
                    break

        total_recs = len(prompt_records)
        pbar = tqdm(prompt_records, total=total_recs, desc="Generating", unit="prompt")
        
        for rec in pbar:
            prompt_tokens = rec.get("prompt_tokens", None)
            if not isinstance(prompt_tokens, list) or len(prompt_tokens) == 0:
                continue

            prompt_id = str(rec.get("id", f"prompt_{n_read}"))
            formula = rec.get("formula", None)

            prompt_ids = tokenizer.encode(prompt_tokens)
            prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)

            gen_ids = model.generate(
                idx=prompt_ids_t,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                num_gens=args.num_samples_per_prompt,
                forbidden_token_ids=forbidden_ids,
                eos_token_id=eos_id,
                early_stop_eos=args.early_stop_eos,
            )

            for j in range(gen_ids.size(0)):
                ids_j = gen_ids[j].tolist()
                tokens_j = tokenizer.decode(ids_j)
                if "<EOS>" in tokens_j:
                    tokens_j = tokens_j[: tokens_j.index("<EOS>") + 1]
                
                if args.use_smact_filter:
                    comp = parse_formula_from_tokens(tokens_j)
                    comp = reduce_stoich(comp)
                    elems = [e for e, _ in comp]
                    counts = [int(c) for _, c in comp]
                    if not elems:
                        continue
                    try:
                        is_valid = smact_validity(elems, counts, use_pauling_test=True, include_alloys=True)
                    except Exception:
                        is_valid = False
                    if not is_valid:
                        continue

                results.append(
                    {
                        "id": f"{prompt_id}__gen_{j}",
                        "prompt_id": prompt_id,
                        "formula": formula,
                        "prompt_tokens": prompt_tokens,
                        "tokens": tokens_j,
                    }
                )

            n_read += 1
            pbar.set_postfix(
                prompts=str(n_read),
                gens=str(len(results)),
                prompt_len=str(len(prompt_tokens)),
            )

            if args.num_generations and args.num_generations > 0 and len(results) >= args.num_generations:
                break

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(results, str(out_path))
    # 避免某些终端编码导致 Unicode 输出异常
    print(f"[Saved] {len(results)} generations -> {out_path} (from {n_read} prompts)")


if __name__ == "__main__":
    # num_samples_per_prompt 决定“每次并行生成多少条”；num_generations 决定“总共要多少条”
    # 随机生成 10000 条 DSS-seq（无成分、无属性）：
    # CUDA_VISIBLE_DEVICES=2 python scripts/generate_tokens_from_prompts.py --checkpoint ckpts/mp_20/ar_model/mp20/best_cry2seq.pt --random_mode --conditioning_mode none --num_generations 10000 --num_samples_per_prompt 512 --max_new_tokens 256 --temperature 0.2 --top_k 5 --out_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new1.pkl.gz --device cuda
    # 随机生成带目标属性条件的 DSS-seq：
    # CUDA_VISIBLE_DEVICES=0 python scripts/generate_tokens_from_prompts.py --checkpoint ckpts/mp_20/ar_model/mp20_bandgap/best_cry2seq.pt --random_mode --conditioning_mode property --property_name band_gap --property_bin 0 --num_generations 10000 --num_samples_per_prompt 64 --max_new_tokens 256 --temperature 1 --top_k 10 --out_path dataset/mp_20/gen_random_DSS-seqs-bandgap/random_property_bg0_10000.pkl.gz --device cuda
    # python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/mp_20/cry2seq/best_cry2seq.pt" --prompts_path "dataset/mp_20/test_prompts.jsonl" --out_path "dataset/mpts_52/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/mpts_52/cry2seq/best_cry2seq.pt" --prompts_path "dataset/mpts_52/prompts/test_prompts.jsonl" --out_path "dataset/mpts_52/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda

    # CUDA_VISIBLE_DEVICES=1 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/carbon_24/cry2seq/best_cry2seq.pt" --prompts_path "dataset/carbon_24/train_prompts.jsonl" --out_path "dataset/carbon_24/gen_DSS/train_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=2 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/carbon_24/cry2seq/best_cry2seq.pt" --prompts_path "dataset/carbon_24/val_prompts.jsonl" --out_path "dataset/carbon_24/gen_DSS/val_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=0 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/carbon_24/cry2seq/best_cry2seq.pt" --prompts_path "dataset/carbon_24/test_prompts.jsonl" --out_path "dataset/carbon_24/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda

    # CUDA_VISIBLE_DEVICES=3 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/train_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/train_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=4 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/val_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/val_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=5 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/test_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda

    # CUDA_VISIBLE_DEVICES=5 python scripts/generate_tokens_from_prompts.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/test_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    """
    bg
CUDA_VISIBLE_DEVICES=0 python scripts/generate_tokens_from_prompts.py \
  --checkpoint ckpts/mp_20/ar_model/mp20_bandgap/best_cry2seq.pt \
  --random_mode \
  --conditioning_mode property \
  --property_name band_gap \
  --property_bin 0 \
  --num_generations 10000 \
  --num_samples_per_prompt 64 \
  --use_smact_filter \
  --out_path dataset/mp_20/gen_random_DSS-seqs-bg/random_property_bg6_10000.pkl.gz \
  --device cuda
    """

    """
    fe 
CUDA_VISIBLE_DEVICES=3 python scripts/generate_tokens_from_prompts.py \
  --checkpoint ckpts/mp_20/ar_model/mp20_fe/best_cry2seq.pt \
  --random_mode \
  --conditioning_mode property \
  --property_name FORM_ENERGY \
  --property_bin 12 \
  --num_generations 10000 \
  --num_samples_per_prompt 64 \
  --use_smact_filter \
  --out_path dataset/mp_20/gen_random_DSS-seqs-fe/random_property_fe12_10000.pkl.gz \
  --device cuda
    
    """
    main()

