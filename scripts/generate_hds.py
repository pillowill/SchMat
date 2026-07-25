"""Generate HDS sequences from trained token models and prompts."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import pickle
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from tqdm import tqdm


def save_pickle(obj: Any, path: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "wb"
    with opener(path, mode) as f:
        pickle.dump(obj, f)


def generation_manifest_path(output_path: Path) -> Path:
    """Keep shard provenance adjacent to, but outside of, the legacy pickle format."""
    return Path(f"{output_path}.manifest.json")


def load_pickle(path: Path) -> Any:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def file_sha256(path: Path) -> str:
    """Return a content identity for artifacts shared by parallel shards."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_file_signature(path_value: str) -> Optional[Dict[str, str]]:
    """Describe an optional analysis artifact without silently accepting a missing file."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Configured analysis artifact does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _validate_shard_arguments(args) -> None:
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard_index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    if args.random_mode and args.num_shards != 1:
        raise ValueError("Prompt sharding is only defined for --prompts_path generation, not --random_mode")


def merge_generation_shards(shard_paths: List[str], output_path: str, metrics_path: str = "") -> None:
    """Merge a complete, compatible set of prompt shards with provenance checks.

    The output remains a plain list of HDS records, so downstream diffusion code
    can keep reading it unchanged.  The sidecar manifest carries validation data.
    """
    if not shard_paths:
        raise ValueError("--merge_shards requires at least one shard path")

    records: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    for raw_path in shard_paths:
        shard_path = Path(raw_path)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard output does not exist: {shard_path}")
        manifest_path = generation_manifest_path(shard_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Shard manifest does not exist: {manifest_path}")
        shard_records = load_pickle(shard_path)
        if not isinstance(shard_records, list):
            raise ValueError(f"Shard is not a list of HDS records: {shard_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise ValueError(f"Unsupported shard manifest format: {manifest_path}")
        if int(manifest.get("result_count", -1)) != len(shard_records):
            raise ValueError(f"Shard record count does not match manifest: {shard_path}")
        prompt_ids = manifest.get("prompt_ids", [])
        if not isinstance(prompt_ids, list) or len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError(f"Shard manifest has invalid or duplicate prompt IDs: {manifest_path}")
        prompt_id_set = set(prompt_ids)
        record_keys = set()
        per_prompt_indices: Dict[str, set] = {prompt_id: set() for prompt_id in prompt_ids}
        for record in shard_records:
            if not isinstance(record, dict):
                raise ValueError(f"Shard contains a non-dictionary HDS record: {shard_path}")
            prompt_id = str(record.get("prompt_id", ""))
            if prompt_id not in prompt_id_set:
                raise ValueError(f"Shard record has an undeclared prompt ID {prompt_id!r}: {shard_path}")
            candidate_index = int(record.get("candidate_index", 0))
            record_key = (prompt_id, candidate_index)
            if record_key in record_keys:
                raise ValueError(f"Shard repeats candidate {record_key}: {shard_path}")
            record_keys.add(record_key)
            per_prompt_indices[prompt_id].add(candidate_index)
        expected_candidates = int(manifest.get("expected_candidates_per_prompt", 0))
        if expected_candidates > 0:
            expected_indices = set(range(expected_candidates))
            incomplete = [
                prompt_id for prompt_id, indices in per_prompt_indices.items()
                if indices != expected_indices
            ]
            if incomplete:
                preview = ", ".join(incomplete[:10])
                raise ValueError(
                    f"Shard lacks the complete candidate set for {len(incomplete)} prompts "
                    f"(first: {preview}): {shard_path}"
                )
        records.extend(shard_records)
        manifests.append(manifest)

    first = manifests[0]
    expected_shards = int(first.get("num_shards", 0))
    if expected_shards < 1 or len(manifests) != expected_shards:
        raise ValueError(
            f"Expected exactly {expected_shards} shards, received {len(manifests)}. "
            "Do not merge a partial generation."
        )
    comparable_keys = (
        "generation_mode",
        "checkpoint",
        "prompts_path",
        "generation_config",
        "total_prompt_count",
        "expected_candidates_per_prompt",
        "checkpoint_sha256",
        "prompts_sha256",
        "analysis_artifacts",
    )
    for manifest in manifests[1:]:
        if int(manifest.get("num_shards", 0)) != expected_shards:
            raise ValueError("Shard manifests disagree on --num_shards")
        for key in comparable_keys:
            if manifest.get(key) != first.get(key):
                raise ValueError(f"Shard manifests disagree on {key}; refusing to mix generation runs")

    shard_indices = [int(manifest.get("shard_index", -1)) for manifest in manifests]
    if sorted(shard_indices) != list(range(expected_shards)):
        raise ValueError(
            f"Shard indices must be exactly 0..{expected_shards - 1}, got {sorted(shard_indices)}"
        )

    seen_prompt_ids = set()
    for manifest in manifests:
        for prompt_id in manifest.get("prompt_ids", []):
            if prompt_id in seen_prompt_ids:
                raise ValueError(f"Prompt {prompt_id!r} appears in more than one shard")
            seen_prompt_ids.add(prompt_id)
    if len(seen_prompt_ids) != int(first.get("total_prompt_count", -1)):
        raise ValueError(
            "Merged prompt IDs do not cover the source prompt set; refusing incomplete generation"
        )

    records.sort(
        key=lambda rec: (str(rec.get("prompt_id", rec.get("id", ""))), int(rec.get("candidate_index", 0)))
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(records, str(output))

    merged_manifest = {
        "format_version": 1,
        "merged": True,
        "source_shards": [str(Path(path)) for path in shard_paths],
        "num_shards": expected_shards,
        "prompt_count": len(seen_prompt_ids),
        "result_count": len(records),
        "generation_mode": first.get("generation_mode"),
        "checkpoint": first.get("checkpoint"),
        "checkpoint_sha256": first.get("checkpoint_sha256"),
        "prompts_path": first.get("prompts_path"),
        "prompts_sha256": first.get("prompts_sha256"),
        "generation_config": first.get("generation_config"),
        "analysis_artifacts": first.get("analysis_artifacts"),
    }
    merged_metric_totals = None
    if first.get("metric_totals") is not None:
        merged_metric_totals = {key: 0.0 for key in first["metric_totals"]}
        for manifest in manifests:
            metric_totals = manifest.get("metric_totals")
            if not isinstance(metric_totals, dict) or set(metric_totals) != set(merged_metric_totals):
                raise ValueError("Shard manifests have incompatible metric accumulators")
            for key, value in metric_totals.items():
                merged_metric_totals[key] += float(value)
        metrics = finalize_csp_metrics(merged_metric_totals)
        metrics.update({
            "generation_mode": first["generation_mode"],
            **first["generation_config"],
        })
        final_metrics_path = Path(metrics_path) if metrics_path else output.with_suffix(".metrics.json")
        final_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        final_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        merged_manifest["metrics_path"] = str(final_metrics_path)
        merged_manifest["metric_totals"] = merged_metric_totals
    generation_manifest_path(output).write_text(
        json.dumps(merged_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[Merged] {len(shard_paths)} shards, {len(records)} HDS records -> {output}")
    if merged_metric_totals is not None:
        print(f"[Merged] analysis metrics -> {merged_manifest['metrics_path']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load pretrained crystal_token_gpt and generate DSS-seq from prompts or random hyperparameter-controlled prompts.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to best_cry2seq.pt")
    parser.add_argument("--prompts_path", type=str, default="", help="Prompt file path (.jsonl) from make_prompts_from_csv.py")
    parser.add_argument("--out_path", type=str, default="results/gpt_tokens/generated_from_prompts.pkl.gz", help="Output pkl(.gz)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate after prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 disables; E2 uses top-p by default)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold; E2 default is 0.95")
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
    parser.add_argument("--num_candidates", type=int, default=0, help="Keep this many canonical candidates per prompt (0 keeps all samples).")
    parser.add_argument("--candidate_oversample", type=int, default=8, help="Generate this multiple before canonical de-duplication (E2 default: 8).")
    parser.add_argument("--canonicalize_candidates", action="store_true", help="Canonicalize formula/site order before writing candidates.")
    parser.add_argument("--deduplicate_candidates", action="store_true", help="Drop duplicate canonical HDS candidates per prompt.")
    parser.add_argument(
        "--mmr_lambda",
        type=float,
        default=0.7,
        help="Likelihood/diversity balance for canonical candidate selection (E2 default: 0.7).",
    )
    parser.add_argument(
        "--generation_mode",
        choices=["auto", "legacy", "constrained_csp"],
        default="auto",
        help="auto selects constrained CSP decoding for E2 checkpoints and legacy decoding otherwise.",
    )
    parser.add_argument(
        "--beta_aux",
        type=float,
        default=1.0,
        help="Auxiliary-head logit weight at code slots. E2 main setting uses 1.0.",
    )
    parser.add_argument(
        "--reference_hds_path",
        type=str,
        default="",
        help="Optional direct-HDS pkl(.gz) used only for code prediction analysis.",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="",
        help="Optional JSON output for schema and CSP code-prediction analysis metrics.",
    )
    parser.add_argument(
        "--vq_checkpoint",
        type=str,
        default="",
        help="Optional Attn-RVQ checkpoint for index-space continuous cosine analysis.",
    )
    parser.add_argument(
        "--vq_config",
        type=str,
        default="",
        help="Optional Attn-RVQ YAML paired with --vq_checkpoint.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Deterministically partition prompt records into this many disjoint shards.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based prompt shard index handled by this process.",
    )
    parser.add_argument(
        "--merge_shards",
        nargs="+",
        default=None,
        help="Merge generated shard pkl(.gz) files. This mode does not require --checkpoint.",
    )
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


def passes_smact_filter(tokens: List[str]) -> bool:
    """Evaluate charge validity without depending on a removed DSS helper script."""
    from pymatgen.core.periodic_table import Element
    from scripts.eval_utils import smact_validity

    composition: List[Tuple[str, int]] = []
    in_formula = False
    pending_atom: Optional[str] = None
    for token in tokens:
        if token == "<FORMULA>":
            in_formula = True
            continue
        if token == "</FORMULA>":
            break
        if not in_formula:
            continue
        atom_match = re.fullmatch(r"<ATOM_([A-Za-z]{1,2})>", token)
        count_match = re.fullmatch(r"<COUNT_(\d+)>", token)
        if atom_match:
            pending_atom = atom_match.group(1)
        elif count_match and pending_atom is not None:
            composition.append((pending_atom, int(count_match.group(1))))
            pending_atom = None
    if not composition:
        return False
    atomic_numbers = [int(Element(symbol).Z) for symbol, _ in composition]
    counts = [count for _, count in composition]
    return bool(smact_validity(atomic_numbers, counts, use_pauling_test=True, include_alloys=True))


def build_stage_maps(vocab) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create embedding and output-head stage routes from the checkpoint vocab."""
    input_stages = torch.zeros(len(vocab), dtype=torch.long)
    head_stages = torch.zeros(len(vocab), dtype=torch.long)
    from scripts.tokenizer import hds_stage_names

    cfg = vocab.config
    stage_to_id = {
        name: index
        for index, name in enumerate(
            hds_stage_names(cfg.num_quantizers_global, cfg.num_quantizers_local),
            start=1,
        )
    }
    for token, token_id in vocab.token_to_id.items():
        match = re.fullmatch(r"<(?:STAGE_)?(GLOBAL|LOCAL)_Q(\d+)>", token)
        if not match:
            match = re.fullmatch(r"<(GLOBAL|LOCAL)_CODE_Q(\d+)_\d+>", token)
        if not match:
            continue
        stage_id = stage_to_id.get(f"{match.group(1)}_Q{match.group(2)}", 0)
        input_stages[token_id] = stage_id
        if token.startswith("<STAGE_"):
            # A stage marker predicts the following code; ordinary code tokens
            # predict syntax and therefore keep the shared output head.
            head_stages[token_id] = stage_id
    return input_stages, head_stages


def canonicalize_candidate(tokens: List[str]) -> List[str]:
    """Remove training-only markers and apply the shared canonical HDS order."""
    from train_ar_model import canonicalize_hds_tokens

    without_markers = [token for token in tokens if not token.startswith("<STAGE_")]
    return canonicalize_hds_tokens(without_markers)


def select_candidates(
    candidates: List[Dict[str, Any]],
    requested: int,
    canonicalize: bool,
    deduplicate: bool,
    mmr_lambda: float = 1.0,
) -> List[Dict[str, Any]]:
    """Apply canonical de-duplication followed by likelihood/diversity MMR."""
    prepared = []
    seen = set()
    for candidate in candidates:
        tokens = canonicalize_candidate(candidate["tokens"]) if canonicalize else candidate["tokens"]
        key = tuple(tokens)
        if deduplicate and key in seen:
            continue
        seen.add(key)
        updated = dict(candidate)
        updated["tokens"] = tokens
        prepared.append(updated)
    # Prefer model likelihood when present; lexical order is a stable tie-break.
    prepared.sort(key=lambda item: (-float(item.get("code_log_probability", 0.0)), tuple(item["tokens"])))
    if requested <= 0 or len(prepared) <= requested or mmr_lambda >= 1.0:
        return prepared[:requested] if requested > 0 else prepared
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be in [0, 1]")

    def code_vector(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
        if "global_codes" not in candidate or "local_codes" not in candidate:
            return tuple(candidate["tokens"])
        return tuple(candidate["global_codes"]) + tuple(
            code for row in candidate["local_codes"] for code in row
        )

    def distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
        left_codes = code_vector(left)
        right_codes = code_vector(right)
        width = max(len(left_codes), len(right_codes), 1)
        mismatches = sum(a != b for a, b in zip(left_codes, right_codes)) + abs(len(left_codes) - len(right_codes))
        return mismatches / width

    scores = [float(item.get("code_log_probability", 0.0)) for item in prepared]
    score_min, score_max = min(scores), max(scores)
    if score_max > score_min:
        relevance = [(score - score_min) / (score_max - score_min) for score in scores]
    else:
        relevance = [1.0] * len(scores)
    selected_indices = [0]
    remaining = set(range(1, len(prepared)))
    while remaining and len(selected_indices) < requested:
        best_index = max(
            remaining,
            key=lambda index: (
                mmr_lambda * relevance[index]
                + (1.0 - mmr_lambda) * min(
                    distance(prepared[index], prepared[selected]) for selected in selected_indices
                ),
                relevance[index],
                tuple(prepared[index]["tokens"]),
            ),
        )
        selected_indices.append(best_index)
        remaining.remove(best_index)
    return [prepared[index] for index in selected_indices]


def extract_property_tokens(tokens: List[str]) -> List[str]:
    """Return the optional complete property block from one prompt."""
    if "<PROPERTY>" not in tokens:
        return []
    start = tokens.index("<PROPERTY>")
    try:
        end = tokens.index("</PROPERTY>", start + 1)
    except ValueError as exc:
        raise ValueError("Prompt contains an unclosed PROPERTY block") from exc
    return tokens[start:end + 1]


def apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample codebook indices and return their filtered log probabilities."""
    if temperature is None or temperature <= 0:
        selected = logits.argmax(dim=-1)
        log_probs = F.log_softmax(logits, dim=-1).gather(1, selected.unsqueeze(1)).squeeze(1)
        return selected, log_probs

    filtered = logits / float(temperature)
    if top_k is not None and top_k > 0:
        keep = min(int(top_k), filtered.size(-1))
        threshold = torch.topk(filtered, keep, dim=-1).values[:, [-1]]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove_mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        filtered = filtered.masked_fill(remove_mask, float("-inf"))
    log_probs = F.log_softmax(filtered, dim=-1)
    selected = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(1)
    return selected, log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def generate_constrained_csp_candidates(
    model,
    tokenizer,
    compiler,
    composition_counts: List[Tuple[str, int]],
    property_tokens: List[str],
    num_samples: int,
    args,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Generate a batch of HDS candidates while sampling code slots only."""
    from scripts.tokenizer import parse_hds_code_token

    num_sites = sum(count for _, count in composition_counts)
    skeleton = compiler.compile_from_composition(
        composition_counts=composition_counts,
        global_codes=[0] * compiler.num_global,
        local_codes=[[0] * compiler.num_local for _ in range(num_sites)],
        property_tokens=property_tokens,
    )
    if len(skeleton.tokens) - 1 > model.config.block_size:
        raise ValueError(
            f"Compiled HDS requires {len(skeleton.tokens) - 1} causal positions, "
            f"checkpoint block_size={model.config.block_size}"
        )

    global_token_ids = [
        torch.tensor(
            [tokenizer.vocab.token_to_id[f"<GLOBAL_CODE_Q{q}_{code}>"] for code in range(compiler.global_size)],
            dtype=torch.long,
            device=device,
        )
        for q in range(compiler.num_global)
    ]
    local_token_ids = [
        torch.tensor(
            [tokenizer.vocab.token_to_id[f"<LOCAL_CODE_Q{q}_{code}>"] for code in range(compiler.local_size)],
            dtype=torch.long,
            device=device,
        )
        for q in range(compiler.num_local)
    ]
    stage_id_by_token, head_stage_id_by_token = build_stage_maps(tokenizer.vocab)
    stage_id_by_token = stage_id_by_token.to(device)
    head_stage_id_by_token = head_stage_id_by_token.to(device)

    generated = torch.empty((num_samples, 0), dtype=torch.long, device=device)
    code_log_probability = torch.zeros(num_samples, dtype=torch.float32, device=device)
    for token in skeleton.tokens:
        parsed = parse_hds_code_token(token)
        if parsed is None:
            token_id = tokenizer.vocab.token_to_id[token]
            next_ids = torch.full((num_samples, 1), token_id, dtype=torch.long, device=device)
            generated = torch.cat([generated, next_ids], dim=1)
            continue

        kind, quantizer, _ = parsed
        assert quantizer is not None
        input_stage_ids = stage_id_by_token[generated]
        output_stage_ids = head_stage_id_by_token[generated]
        ar_logits, _, hidden = model(
            generated,
            stage_ids=input_stage_ids,
            head_stage_ids=output_stage_ids,
            return_hidden=True,
        )
        allowed_ids = global_token_ids[quantizer] if kind == "global" else local_token_ids[quantizer]
        fused_logits = ar_logits[:, -1, allowed_ids]
        if model.config.auxiliary_code_heads:
            fused_logits = fused_logits + float(args.beta_aux) * model.auxiliary_logits(
                hidden[:, -1, :], kind=kind, quantizer=quantizer
            )
        selected, selected_log_probability = apply_sampling_filters(
            fused_logits,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if 0.0 < args.top_p < 1.0 else None,
        )
        code_log_probability += selected_log_probability
        generated = torch.cat([generated, allowed_ids[selected].unsqueeze(1)], dim=1)

    candidates: List[Dict[str, Any]] = []
    for row in range(num_samples):
        raw_tokens = tokenizer.decode(generated[row].tolist())
        raw_schema = assess_raw_constrained_hds(raw_tokens, skeleton.tokens, compiler)
        # Same-element site rows are canonicalized after their local tuples are
        # known; strict parsing then verifies the complete emitted schema.
        compiled = compiler.compile_legacy_record({"tokens": raw_tokens, "id": f"candidate-{row}"})
        compiler.parse_strict_record(compiled.tokens)
        candidates.append({
            "tokens": compiled.tokens,
            "composition_counts": [
                {"symbol": symbol, "count": count} for symbol, count in compiled.composition_counts
            ],
            "global_codes": compiled.global_codes,
            "atom_symbols": compiled.atom_symbols,
            "local_codes": compiled.local_codes,
            "canonical_atom_permutation": compiled.canonical_atom_permutation,
            "inverse_canonical_atom_permutation": compiled.inverse_canonical_atom_permutation,
            "unpadded_length": len(compiled.tokens),
            "code_log_probability": float(code_log_probability[row].item()),
            "schema_valid": raw_schema["schema_valid"],
            "formula_site_consistent": raw_schema["formula_site_consistent"],
            "site_count_match": raw_schema["site_count_match"],
        })
    return candidates


def assess_raw_constrained_hds(
    raw_tokens: List[str],
    template_tokens: List[str],
    compiler,
) -> Dict[str, bool]:
    """Assess emitted syntax before canonical site reordering normalizes it."""
    from scripts.tokenizer import parse_hds_code_token

    schema_valid = len(raw_tokens) == len(template_tokens)
    if schema_valid:
        for actual, template in zip(raw_tokens, template_tokens):
            expected_code = parse_hds_code_token(template)
            if expected_code is None:
                if actual != template:
                    schema_valid = False
                    break
                continue
            actual_code = parse_hds_code_token(actual)
            if actual_code is None or actual_code[:2] != expected_code[:2]:
                schema_valid = False
                break

    try:
        composition_counts = compiler.parse_composition_tokens(raw_tokens)
    except ValueError:
        composition_counts = []

    site_atoms: List[str] = []
    site_ids: List[int] = []
    inside_site = False
    for token in raw_tokens:
        opening = re.fullmatch(r"<SITE_(\d+)>", token)
        closing = re.fullmatch(r"</SITE_(\d+)>", token)
        if opening:
            site_ids.append(int(opening.group(1)))
            inside_site = True
            continue
        if closing:
            inside_site = False
            continue
        if inside_site:
            atom = re.fullmatch(r"<ATOM_([A-Za-z]{1,2})>", token)
            if atom:
                site_atoms.append(atom.group(1))

    expected_sites = sum(count for _, count in composition_counts)
    formula_site_consistent = bool(composition_counts) and (
        Counter(site_atoms) == Counter(dict(composition_counts))
    )
    num_site_tokens = [
        int(match.group(1))
        for token in raw_tokens
        if (match := re.fullmatch(r"<NUM_SITES_(\d+)>", token))
    ]
    site_count_match = (
        len(num_site_tokens) == 1
        and num_site_tokens[0] == expected_sites == len(site_atoms)
        and site_ids == list(range(1, expected_sites + 1))
    )
    return {
        "schema_valid": bool(schema_valid),
        "formula_site_consistent": bool(formula_site_consistent),
        "site_count_match": bool(site_count_match),
    }


def load_reference_hds(path: str, compiler) -> Dict[str, Any]:
    """Load direct HDS records by base material id for optional analysis."""
    if not path:
        return {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as handle:
        records = pickle.load(handle)
    reference = {}
    for row, record in enumerate(records):
        record_id = str(record.get("id", row))
        reference[record_id] = compiler.compile_legacy_record(record)
    return reference


def _metric_file_signature(path: str) -> Dict[str, Any]:
    artifact = Path(path)
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"available": True, "size": int(artifact.stat().st_size), "sha256": digest.hexdigest()}


def _provenance_path_matches(declared: Optional[str], actual: str) -> bool:
    if not declared:
        return True
    declared_norm = Path(declared).as_posix().lower()
    actual_norm = Path(actual).as_posix().lower()
    return actual_norm.endswith(declared_norm) or declared_norm.endswith(actual_norm)


def validate_metric_vq_provenance(
    provenance: Dict[str, Any],
    config_path: str,
    checkpoint_path: str,
) -> None:
    """Bind optional cosine metrics to the same artifacts as the HDS model."""
    if not provenance:
        raise ValueError("VQ cosine analysis requires Attn-RVQ provenance in the HDS checkpoint")
    pairs = (
        ("vq_checkpoint", "vq_checkpoint_signature", checkpoint_path),
        ("vq_config", "vq_config_signature", config_path),
    )
    for path_field, signature_field, actual in pairs:
        if not _provenance_path_matches(provenance.get(path_field), actual):
            raise ValueError(f"{path_field} does not match the HDS checkpoint provenance")
        declared_signature = provenance.get(signature_field)
        if not declared_signature or not declared_signature.get("available"):
            raise ValueError(f"HDS checkpoint lacks a usable {signature_field}")
        if declared_signature != _metric_file_signature(actual):
            raise ValueError(f"{signature_field} does not match the supplied artifact")


def load_vq_model_for_metrics(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    provenance: Dict[str, Any],
):
    """Load Attn-RVQ only when continuous code-space analysis is requested."""
    import yaml
    from model.attn_rvq import CrystalVQVAE

    validate_metric_vq_provenance(provenance, config_path, checkpoint_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model = CrystalVQVAE(
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 4),
        codebook_size=config.get("codebook_size", 512),
        num_space_groups=config.get("num_space_groups", 230),
        max_cn=config.get("max_cn", 20),
        num_elements=config.get("num_elements", 100),
        commitment_weight=config.get("commitment_weight", 0.0),
        loss_weights=config.get("loss_weights", None),
        num_quantizers=config.get("num_quantizers", 4),
        num_quantizers_global=config.get("num_quantizers_global", None),
        num_quantizers_local=config.get("num_quantizers_local", None),
        sample_codebook_temp=config.get("sample_codebook_temp", 0.1),
        vq_type=config.get("vq_type", "residual"),
        aggregator_dropout=config.get("aggregator_dropout", 0.1),
        aggregator_attn_dropout=config.get("aggregator_attn_dropout", 0.0),
        aggregator_use_positional=config.get("aggregator_use_positional", True),
        aggregator_positional_scale=config.get("aggregator_positional_scale", 1.0),
        residual_use_outer_aggregator=config.get("residual_use_outer_aggregator", True),
        residual_attn_use_outer_aggregator=config.get("residual_attn_use_outer_aggregator", False),
        stacked_vq_use_outer_aggregator=config.get("stacked_vq_use_outer_aggregator", True),
        enable_residual_codebook_params=config.get("enable_residual_codebook_params", False),
        residual_codebook_params=config.get("residual_codebook_params", None),
    ).to(device)
    model_state = model.state_dict()
    critical_prefixes = ("vq_global.", "vq_local.", "global_aggregator.", "local_aggregator.")
    missing_critical = []
    mismatched_critical = []
    for key, expected in model_state.items():
        if not key.startswith(critical_prefixes):
            continue
        if key not in state_dict:
            missing_critical.append(key)
        elif tuple(state_dict[key].shape) != tuple(expected.shape):
            mismatched_critical.append(
                f"{key}: checkpoint={tuple(state_dict[key].shape)}, model={tuple(expected.shape)}"
            )
    if missing_critical or mismatched_critical:
        raise ValueError(
            "Attn-RVQ checkpoint/config mismatch for VQ metrics; "
            f"missing={missing_critical[:5]}, shape_mismatch={mismatched_critical[:5]}"
        )
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model.load_state_dict(compatible_state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def accumulate_vq_cosines(
    totals: Dict[str, float],
    candidate: Dict[str, Any],
    reference,
    vq_model,
    device: torch.device,
) -> None:
    """Compare top-candidate and reference HDS in deterministic VQ codebook space.

    Alpha-weighted Attn-RVQ gates depend on encoder residuals, which are absent
    from an HDS. The model helper therefore uses only the selected codebook
    vectors for both sides rather than attempting to recreate unavailable gates.
    """
    predicted_global = torch.tensor([candidate["global_codes"]], dtype=torch.long, device=device)
    reference_global = torch.tensor([reference.global_codes], dtype=torch.long, device=device)
    predicted_local = torch.tensor(candidate["local_codes"], dtype=torch.long, device=device)
    reference_local = torch.tensor(reference.local_codes, dtype=torch.long, device=device)

    predicted_global_vq = vq_model.get_hds_metric_embedding_from_indices(predicted_global, is_global=True)
    reference_global_vq = vq_model.get_hds_metric_embedding_from_indices(reference_global, is_global=True)
    predicted_local_vq = vq_model.get_hds_metric_embedding_from_indices(predicted_local, is_global=False)
    reference_local_vq = vq_model.get_hds_metric_embedding_from_indices(reference_local, is_global=False)

    global_cosine = F.cosine_similarity(predicted_global_vq, reference_global_vq, dim=-1).mean()
    local_cosine = F.cosine_similarity(predicted_local_vq, reference_local_vq, dim=-1).mean()
    predicted_hds_vq = torch.cat([predicted_global_vq.flatten(), predicted_local_vq.flatten()])
    reference_hds_vq = torch.cat([reference_global_vq.flatten(), reference_local_vq.flatten()])
    hds_cosine = F.cosine_similarity(
        predicted_hds_vq.unsqueeze(0), reference_hds_vq.unsqueeze(0), dim=-1
    ).mean()

    totals["global_vq_cosine_sum"] += float(global_cosine.item())
    totals["local_vq_cosine_sum"] += float(local_cosine.item())
    totals["hds_vq_cosine_sum"] += float(hds_cosine.item())
    totals["vq_cosine_prompts"] += 1


def update_csp_metrics(
    totals: Dict[str, float],
    candidates: List[Dict[str, Any]],
    reference,
    vq_model=None,
    device: Optional[torch.device] = None,
) -> None:
    """Accumulate schema and direct-code comparison metrics without gating output."""
    totals["prompts"] += 1
    totals["candidates"] += len(candidates)
    totals["schema_valid"] += sum(bool(candidate.get("schema_valid")) for candidate in candidates)
    totals["formula_site_consistent"] += sum(
        bool(candidate.get("formula_site_consistent")) for candidate in candidates
    )
    totals["site_count_match"] += sum(bool(candidate.get("site_count_match")) for candidate in candidates)
    if reference is None or not candidates:
        return
    totals["reference_prompts"] += 1
    top = candidates[0]
    reference_composition = [
        {"symbol": symbol, "count": count} for symbol, count in reference.composition_counts
    ]
    if top["composition_counts"] != reference_composition:
        raise ValueError(
            f"Prompt/reference composition mismatch: prompt={top['composition_counts']}, "
            f"reference={reference_composition}"
        )
    ref_global = reference.global_codes
    ref_local = reference.local_codes
    totals["global_slots"] += len(ref_global)
    totals["global_slot_correct"] += sum(a == b for a, b in zip(top["global_codes"], ref_global))
    flat_top_local = [code for row in top["local_codes"] for code in row]
    flat_ref_local = [code for row in ref_local for code in row]
    totals["local_slots"] += len(flat_ref_local)
    totals["local_slot_correct"] += sum(a == b for a, b in zip(flat_top_local, flat_ref_local))
    totals["global_exact"] += int(top["global_codes"] == ref_global)
    totals["local_exact"] += int(top["local_codes"] == ref_local)
    totals["topk_full_coverage"] += int(any(
        candidate["global_codes"] == ref_global and candidate["local_codes"] == ref_local
        for candidate in candidates
    ))
    if vq_model is not None:
        if device is None:
            raise ValueError("device is required when VQ cosine analysis is enabled")
        accumulate_vq_cosines(totals, top, reference, vq_model, device)


def finalize_csp_metrics(totals: Dict[str, float]) -> Dict[str, float]:
    """Finalize analysis metrics; no value is treated as an acceptance threshold."""
    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(totals[denominator], 1)

    metrics = {
        "schema_valid_rate": ratio("schema_valid", "candidates"),
        "formula_site_composition_consistency": ratio("formula_site_consistent", "candidates"),
        "site_count_match_rate": ratio("site_count_match", "candidates"),
        "candidate_count": totals["candidates"],
        "prompt_count": totals["prompts"],
    }
    if totals["reference_prompts"]:
        metrics.update({
            "global_code_match_rate": ratio("global_slot_correct", "global_slots"),
            "local_code_match_rate": ratio("local_slot_correct", "local_slots"),
            "global_exact_match_rate": ratio("global_exact", "reference_prompts"),
            "local_exact_match_rate": ratio("local_exact", "reference_prompts"),
            "candidate_top_k_coverage": ratio("topk_full_coverage", "reference_prompts"),
            "reference_prompt_count": totals["reference_prompts"],
        })
    if totals["vq_cosine_prompts"]:
        metrics.update({
            "global_vq_cosine": ratio("global_vq_cosine_sum", "vq_cosine_prompts"),
            "local_vq_cosine": ratio("local_vq_cosine_sum", "vq_cosine_prompts"),
            "hds_vq_cosine": ratio("hds_vq_cosine_sum", "vq_cosine_prompts"),
            "vq_cosine_prompt_count": int(totals["vq_cosine_prompts"]),
        })
    return metrics


def main():
    args = parse_args()
    if args.merge_shards is not None:
        merge_generation_shards(args.merge_shards, args.out_path, args.metrics_path)
        return
    _validate_shard_arguments(args)
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --merge_shards is used")
    if args.num_shards > 1 and args.num_generations > 0:
        raise ValueError(
            "--num_generations is incompatible with prompt sharding because it can create "
            "an incomplete merged dataset"
        )
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    project_root = Path(__file__).resolve().parent.parent
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from model.crystal_token_gpt import GPT, GPTConfig
    from scripts.tokenizer import CrystalVocabConfig, CrystalVocab, CrystalTokenizer, HDSCompiler

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    vocab_cfg = CrystalVocabConfig(**ckpt["vocab_config"])
    vocab = CrystalVocab(vocab_cfg)
    tokenizer = CrystalTokenizer(vocab)

    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    is_e2_checkpoint = bool(ckpt.get("e2_config", {}).get("enabled", gpt_cfg.auxiliary_code_heads))
    generation_mode = args.generation_mode
    if generation_mode == "auto":
        generation_mode = "constrained_csp" if is_e2_checkpoint and not args.random_mode else "legacy"
    if generation_mode == "constrained_csp" and args.random_mode:
        raise ValueError(
            "constrained_csp requires full-cell composition prompts; "
            "de novo formula generation remains an E4 follow-up"
        )
    if generation_mode == "constrained_csp" and not gpt_cfg.auxiliary_code_heads:
        raise ValueError("constrained_csp E2 decoding requires a checkpoint with auxiliary_code_heads=True")
    compiler = HDSCompiler(vocab_cfg)
    reference_hds = load_reference_hds(args.reference_hds_path, compiler) if generation_mode == "constrained_csp" else {}
    if bool(args.vq_checkpoint) != bool(args.vq_config):
        raise ValueError("--vq_checkpoint and --vq_config must be provided together")
    if (args.vq_checkpoint or args.vq_config) and not reference_hds:
        raise ValueError("VQ cosine analysis also requires --reference_hds_path")
    vq_model = (
        load_vq_model_for_metrics(
            args.vq_config,
            args.vq_checkpoint,
            device,
            provenance=ckpt.get("attn_rvq_provenance", {}),
        )
        if args.vq_checkpoint
        else None
    )
    metric_totals = {
        "prompts": 0,
        "candidates": 0,
        "schema_valid": 0,
        "formula_site_consistent": 0,
        "site_count_match": 0,
        "reference_prompts": 0,
        "global_slots": 0,
        "global_slot_correct": 0,
        "local_slots": 0,
        "local_slot_correct": 0,
        "global_exact": 0,
        "local_exact": 0,
        "topk_full_coverage": 0,
        "global_vq_cosine_sum": 0.0,
        "local_vq_cosine_sum": 0.0,
        "hds_vq_cosine_sum": 0.0,
        "vq_cosine_prompts": 0,
    }
    print(f"Generation mode: {generation_mode}")

    unk_id = vocab.token_to_id.get("<UNK>", None)
    eos_id = vocab.token_to_id.get("<EOS>", None)
    forbidden_ids = [] if args.allow_unk else ([unk_id] if unk_id is not None else [])

    top_k: Optional[int] = args.top_k if args.top_k and args.top_k > 0 else None
    top_p: Optional[float] = args.top_p if 0.0 < args.top_p < 1.0 else None
    stage_id_by_token, head_stage_id_by_token = build_stage_maps(vocab)
    samples_per_prompt = args.num_samples_per_prompt
    if args.num_candidates > 0:
        samples_per_prompt = args.num_candidates * max(1, args.candidate_oversample)

    results: List[Dict[str, Any]] = []
    n_read = 0

    if args.random_mode:
        prompt_tokens = build_random_prompt_tokens(args)
        for t in prompt_tokens:
            if t not in vocab.token_to_id:
                raise ValueError(f"Token '{t}' is not in checkpoint vocabulary. Please use a compatible cry2seq checkpoint.")

        target_n = args.num_generations if args.num_generations > 0 else 10000
        batch_size = max(1, int(samples_per_prompt))
        
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
                top_p=top_p,
                num_gens=1,
                forbidden_token_ids=forbidden_ids,
                eos_token_id=eos_id,
                early_stop_eos=args.early_stop_eos,
                stage_id_by_token=stage_id_by_token,
                head_stage_id_by_token=head_stage_id_by_token,
            )

            for j in range(gen_ids.size(0)):
                ids_j = gen_ids[j].tolist()
                tokens_j = tokenizer.decode(ids_j)
                if "<EOS>" in tokens_j:
                    tokens_j = tokens_j[: tokens_j.index("<EOS>") + 1]

                if args.use_smact_filter:
                    try:
                        is_valid = passes_smact_filter(tokens_j)
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

        total_prompt_count = len(prompt_records)
        indexed_prompt_records = list(enumerate(prompt_records))
        all_prompt_ids = [
            str(rec.get("id", f"prompt_{source_index}"))
            for source_index, rec in indexed_prompt_records
        ]
        if len(all_prompt_ids) != len(set(all_prompt_ids)):
            raise ValueError("Prompt IDs must be globally unique before deterministic sharding")
        indexed_prompt_records = indexed_prompt_records[args.shard_index::args.num_shards]
        shard_prompt_ids = [
            str(rec.get("id", f"prompt_{source_index}"))
            for source_index, rec in indexed_prompt_records
        ]
        if len(shard_prompt_ids) != len(set(shard_prompt_ids)):
            raise ValueError("Prompt IDs must be unique before deterministic sharding")
        if generation_mode == "constrained_csp" and reference_hds:
            missing_reference_ids = [prompt_id for prompt_id in shard_prompt_ids if prompt_id not in reference_hds]
            if missing_reference_ids:
                preview = ", ".join(missing_reference_ids[:10])
                raise ValueError(
                    f"{len(missing_reference_ids)} prompt IDs have no direct-HDS reference "
                    f"(first: {preview}). Refuse to generate an unaligned CSP split."
                )
        total_recs = len(indexed_prompt_records)
        print(
            f"[Shard] {args.shard_index + 1}/{args.num_shards}: "
            f"{total_recs}/{total_prompt_count} prompt records"
        )
        pbar = tqdm(indexed_prompt_records, total=total_recs, desc="Generating", unit="prompt")
        
        for source_index, rec in pbar:
            prompt_tokens = rec.get("prompt_tokens", None)
            if not isinstance(prompt_tokens, list) or len(prompt_tokens) == 0:
                continue

            prompt_id = str(rec.get("id", f"prompt_{source_index}"))
            formula = rec.get("formula", None)

            if generation_mode == "constrained_csp":
                composition_counts = compiler.parse_composition_tokens(prompt_tokens)
                property_tokens = extract_property_tokens(prompt_tokens)
                requested_candidates = int(
                    args.num_candidates
                    if args.num_candidates > 0
                    else ckpt.get("num_candidates", 5)
                )
                if requested_candidates <= 0:
                    requested_candidates = 5
                per_attempt = requested_candidates * max(1, int(args.candidate_oversample))
                raw_candidates: List[Dict[str, Any]] = []
                selected_candidates: List[Dict[str, Any]] = []
                for _ in range(3):
                    raw_candidates.extend(generate_constrained_csp_candidates(
                        model=model,
                        tokenizer=tokenizer,
                        compiler=compiler,
                        composition_counts=composition_counts,
                        property_tokens=property_tokens,
                        num_samples=per_attempt,
                        args=args,
                        device=device,
                    ))
                    selected_candidates = select_candidates(
                        raw_candidates,
                        requested=requested_candidates,
                        canonicalize=False,
                        deduplicate=True,
                        mmr_lambda=args.mmr_lambda,
                    )
                    if len(selected_candidates) >= requested_candidates:
                        break
                if len(selected_candidates) != requested_candidates:
                    raise ValueError(
                        f"Prompt {prompt_id} produced only {len(selected_candidates)} unique canonical HDS "
                        f"after {len(raw_candidates)} samples; requested {requested_candidates}. "
                        "Increase temperature/top-p or candidate_oversample."
                    )

                for candidate_index, candidate in enumerate(selected_candidates):
                    candidate.update({
                        "id": prompt_id,
                        "candidate_id": f"{prompt_id}__hds_{candidate_index}",
                        "candidate_index": candidate_index,
                        "prompt_id": prompt_id,
                        "formula": formula,
                        "prompt_tokens": prompt_tokens,
                        "source": "predicted_e2_csp",
                        "beta_aux": float(args.beta_aux),
                        "attn_rvq_provenance": ckpt.get("attn_rvq_provenance", {}),
                    })
                results.extend(selected_candidates)
                update_csp_metrics(
                    metric_totals,
                    selected_candidates,
                    reference_hds.get(prompt_id),
                    vq_model=vq_model,
                    device=device,
                )
                n_read += 1
                pbar.set_postfix(
                    prompts=str(n_read),
                    gens=str(len(results)),
                    prompt_len=str(len(prompt_tokens)),
                )
                if args.num_generations and args.num_generations > 0 and len(results) >= args.num_generations:
                    break
                continue

            prompt_ids = tokenizer.encode(prompt_tokens)
            prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)

            gen_ids = model.generate(
                idx=prompt_ids_t,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                top_p=top_p,
                num_gens=samples_per_prompt,
                forbidden_token_ids=forbidden_ids,
                eos_token_id=eos_id,
                early_stop_eos=args.early_stop_eos,
                stage_id_by_token=stage_id_by_token,
                head_stage_id_by_token=head_stage_id_by_token,
            )

            prompt_candidates: List[Dict[str, Any]] = []
            for j in range(gen_ids.size(0)):
                ids_j = gen_ids[j].tolist()
                tokens_j = tokenizer.decode(ids_j)
                if "<EOS>" in tokens_j:
                    tokens_j = tokens_j[: tokens_j.index("<EOS>") + 1]
                
                if args.use_smact_filter:
                    try:
                        is_valid = passes_smact_filter(tokens_j)
                    except Exception:
                        is_valid = False
                    if not is_valid:
                        continue

                prompt_candidates.append(
                    {
                        "id": f"{prompt_id}__gen_{j}",
                        "prompt_id": prompt_id,
                        "formula": formula,
                        "prompt_tokens": prompt_tokens,
                        "tokens": tokens_j,
                    }
                )

            results.extend(
                select_candidates(
                    prompt_candidates,
                    requested=args.num_candidates,
                    canonicalize=args.canonicalize_candidates,
                    deduplicate=args.deduplicate_candidates,
                )
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
    if not args.random_mode:
        effective_num_candidates = int(
            args.num_candidates if args.num_candidates > 0 else ckpt.get("num_candidates", 5)
        )
        if effective_num_candidates <= 0:
            effective_num_candidates = 5
        generation_config = {
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "num_candidates": int(args.num_candidates),
            "effective_num_candidates": effective_num_candidates,
            "candidate_oversample": int(args.candidate_oversample),
            "mmr_lambda": float(args.mmr_lambda),
            "beta_aux": float(args.beta_aux),
            "max_new_tokens": int(args.max_new_tokens),
            "num_samples_per_prompt": int(args.num_samples_per_prompt),
            "canonicalize_candidates": bool(args.canonicalize_candidates),
            "deduplicate_candidates": bool(args.deduplicate_candidates),
            "allow_unk": bool(args.allow_unk),
            "early_stop_eos": bool(args.early_stop_eos),
            "use_smact_filter": bool(args.use_smact_filter),
        }
        shard_manifest = {
            "format_version": 1,
            "merged": False,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "total_prompt_count": int(total_prompt_count),
            "prompt_ids": shard_prompt_ids,
            "result_count": len(results),
            "generation_mode": generation_mode,
            "checkpoint": str(ckpt_path.resolve()),
            "checkpoint_sha256": file_sha256(ckpt_path),
            "prompts_path": str(Path(args.prompts_path).resolve()),
            "prompts_sha256": file_sha256(Path(args.prompts_path)),
            "generation_config": generation_config,
            "expected_candidates_per_prompt": (
                effective_num_candidates if generation_mode == "constrained_csp" else 0
            ),
            "analysis_artifacts": {
                "reference_hds": optional_file_signature(args.reference_hds_path),
                "vq_checkpoint": optional_file_signature(args.vq_checkpoint),
                "vq_config": optional_file_signature(args.vq_config),
            },
            "metric_totals": metric_totals if generation_mode == "constrained_csp" else None,
        }
        generation_manifest_path(out_path).write_text(
            json.dumps(shard_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[Shard] manifest -> {generation_manifest_path(out_path)}")
    if generation_mode == "constrained_csp":
        metrics = finalize_csp_metrics(metric_totals)
        metrics.update({
            "generation_mode": generation_mode,
            "beta_aux": float(args.beta_aux),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "mmr_lambda": float(args.mmr_lambda),
        })
        metrics_path = Path(args.metrics_path) if args.metrics_path else out_path.with_suffix(".metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[Metrics] {json.dumps(metrics, sort_keys=True)}")
        print(f"[Saved] analysis metrics -> {metrics_path}")
    # 避免某些终端编码导致 Unicode 输出异常
    print(f"[Saved] {len(results)} generations -> {out_path} (from {n_read} prompts)")


if __name__ == "__main__":
    # num_samples_per_prompt 决定“每次并行生成多少条”；num_generations 决定“总共要多少条”
    # 随机生成 10000 条 DSS-seq（无成分、无属性）：
    # CUDA_VISIBLE_DEVICES=2 python scripts/generate_hds.py --checkpoint ckpts/mp_20/ar_model/mp20/best_cry2seq.pt --random_mode --conditioning_mode none --num_generations 10000 --num_samples_per_prompt 512 --max_new_tokens 256 --temperature 0.2 --top_k 5 --out_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new1.pkl.gz --device cuda
    # 随机生成带目标属性条件的 DSS-seq：
    # CUDA_VISIBLE_DEVICES=0 python scripts/generate_hds.py --checkpoint ckpts/mp_20/ar_model/mp20_bandgap/best_cry2seq.pt --random_mode --conditioning_mode property --property_name band_gap --property_bin 0 --num_generations 10000 --num_samples_per_prompt 64 --max_new_tokens 256 --temperature 1 --top_k 10 --out_path dataset/mp_20/gen_random_DSS-seqs-bandgap/random_property_bg0_10000.pkl.gz --device cuda
    # python scripts/generate_hds.py --checkpoint "ckpts/mp_20/cry2seq/best_cry2seq.pt" --prompts_path "dataset/mp_20/test_prompts.jsonl" --out_path "dataset/mpts_52/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # python scripts/generate_hds.py --checkpoint "ckpts/mpts_52/cry2seq/best_cry2seq.pt" --prompts_path "dataset/mpts_52/prompts/test_prompts.jsonl" --out_path "dataset/mpts_52/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda

    # CUDA_VISIBLE_DEVICES=1 python scripts/generate_hds.py --checkpoint "ckpts/carbon_24/cry2seq/best_cry2seq.pt" --prompts_path "dataset/carbon_24/train_prompts.jsonl" --out_path "dataset/carbon_24/gen_DSS/train_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=2 python scripts/generate_hds.py --checkpoint "ckpts/carbon_24/cry2seq/best_cry2seq.pt" --prompts_path "dataset/carbon_24/val_prompts.jsonl" --out_path "dataset/carbon_24/gen_DSS/val_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=0 python scripts/generate_hds.py --checkpoint "ckpts/carbon_24/cry2seq/best_cry2seq.pt" --prompts_path "dataset/carbon_24/test_prompts.jsonl" --out_path "dataset/carbon_24/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda

    # CUDA_VISIBLE_DEVICES=3 python scripts/generate_hds.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/train_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/train_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=4 python scripts/generate_hds.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/val_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/val_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    # CUDA_VISIBLE_DEVICES=5 python scripts/generate_hds.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/test_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda

    # CUDA_VISIBLE_DEVICES=5 python scripts/generate_hds.py --checkpoint "ckpts/perov_5/cry2seq/best_cry2seq.pt" --prompts_path "dataset/perov_5/test_prompts.jsonl" --out_path "dataset/perov_5/gen_DSS/test_generated.pkl.gz" --max_new_tokens 512 --temperature 0.3 --top_k 0 --num_samples_per_prompt 1 --device cuda
    """
    bg
CUDA_VISIBLE_DEVICES=0 python scripts/generate_hds.py \
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
CUDA_VISIBLE_DEVICES=3 python scripts/generate_hds.py \
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

