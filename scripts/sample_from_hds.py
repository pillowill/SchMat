import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from scripts.tokenizer import CrystalVocabConfig, HDSCompiler, parse_hds_code_token

SITE_START_RE = re.compile(r"^<SITE_(\d+)>$")
SITE_END_RE = re.compile(r"^</SITE_(\d+)>$")
ATOM_RE = re.compile(r"^<ATOM_([A-Za-z]{1,2})>$")
COUNT_RE = re.compile(r"^<COUNT_(\d+)>$")
PROPERTY_BIN_RE = re.compile(r"^<PROPERTY_BIN_(\d+)>$")
NUM_SITES_RE = re.compile(r"^<NUM_SITES_(\d+)>$")
HDS_SHUFFLE_MODES = {"matched", "global", "local", "both"}


def parse_structure_from_cif_str(cif_str: str):
    """Load the optional metrics dependency only when structure analysis is requested."""
    from scripts.hds_eval_metrics import parse_structure_from_cif_str as parse_impl

    return parse_impl(cif_str)


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    assert p.is_file(), f"Input not found: {p}"

    if p.suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rb") as f:
        return pickle.load(f)


def save_results_csv(results: List[Dict[str, Any]], path: str):
    rows: List[Dict[str, Any]] = []
    max_gen = 0
    for rec in results:
        max_gen = max(max_gen, len(rec.get("generated_cifs", [])))

    for rec in results:
        row = {
            "id": rec.get("id"),
            "formula_composition": json.dumps(rec.get("formula_composition", []), ensure_ascii=False),
            "indices_g": json.dumps(rec.get("indices_g").tolist() if rec.get("indices_g") is not None else None),
            "indices_l": json.dumps(rec.get("indices_l").tolist() if rec.get("indices_l") is not None else None),
            "property_bin": rec.get("property_bin"),
            "used_property_bin": rec.get("used_property_bin"),
            "hds_conditions": rec.get("hds_conditions", 1),
            "diffusion_samples_per_hds": rec.get("diffusion_samples_per_hds"),
            "parallel_diffusion_batch_size": rec.get("parallel_diffusion_batch_size"),
            "attn_rvq_provenance": json.dumps(rec.get("attn_rvq_provenance", {}), ensure_ascii=False),
            "hds_shuffle_mode": rec.get("hds_shuffle_mode", "matched"),
            "hds_shuffle_seed": rec.get("hds_shuffle_seed"),
            "hds_shuffle_manifest_sha256": rec.get("hds_shuffle_manifest_sha256"),
            "sampling_seed": rec.get("sampling_seed"),
            "sample_seed": rec.get("sample_seed"),
            "global_donor_id": rec.get("global_donor_id"),
            "local_donor_ids": json.dumps(rec.get("local_donor_ids", []), ensure_ascii=False),
            "local_rank_fallbacks": rec.get("local_rank_fallbacks", 0),
            "local_unchanged_no_donor": rec.get("local_unchanged_no_donor", 0),
        }
        gen_cifs = rec.get("generated_cifs", [])
        for i in range(max_gen):
            row[f"gen_structure_{i}"] = gen_cifs[i] if i < len(gen_cifs) else None

        if "E_unrelaxed" in rec:
            row["E_unrelaxed"] = rec.get("E_unrelaxed")
        if "E_relaxed" in rec:
            row["E_relaxed"] = rec.get("E_relaxed")
        if "BandGap" in rec:
            row["BandGap"] = rec.get("BandGap")
        if "BandGap_unrelaxed" in rec:
            row["BandGap_unrelaxed"] = rec.get("BandGap_unrelaxed")
        if "BandGap_relaxed" in rec:
            row["BandGap_relaxed"] = rec.get("BandGap_relaxed")
        if "BandGap_relaxed_cif_path" in rec:
            row["BandGap_relaxed_cif_path"] = rec.get("BandGap_relaxed_cif_path")
        if "match_rate" in rec:
            row["match_rate"] = rec.get("match_rate")
        if "rms_dist" in rec:
            row["rms_dist"] = rec.get("rms_dist")

        rows.append(row)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def validate_attn_rvq_provenance(
    records: List[Dict[str, Any]],
    vq_checkpoint: str,
    vq_config: str,
) -> Dict[str, Any]:
    """Reject mixed E2 HDS sources before constructing diffusion conditions."""
    provenances = {
        json.dumps(record.get("attn_rvq_provenance", {}), sort_keys=True)
        for record in records
        if record.get("attn_rvq_provenance")
    }
    if not provenances:
        print("[Provenance] Input HDS has no Attn-RVQ provenance (legacy data).")
        return {}
    if len(provenances) != 1:
        raise ValueError("Input HDS records mix different Attn-RVQ provenance values")
    provenance = json.loads(next(iter(provenances)))

    def same_path(declared: Optional[str], actual: str) -> bool:
        if not declared:
            return True
        declared_norm = Path(declared).as_posix().lower()
        actual_norm = Path(actual).as_posix().lower()
        return actual_norm.endswith(declared_norm) or declared_norm.endswith(actual_norm)

    def file_signature(actual: str) -> Dict[str, Any]:
        path = Path(actual)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "available": True,
            "size": int(path.stat().st_size),
            "sha256": digest.hexdigest(),
        }

    def validate_signature(field: str, actual: str) -> None:
        declared = provenance.get(field)
        # Legacy E2 outputs may have only path provenance. New outputs carry a
        # content digest and must match the exact Attn-RVQ artifact used here.
        if not declared or not declared.get("available"):
            return
        actual_signature = file_signature(actual)
        if (
            declared.get("size") != actual_signature["size"]
            or declared.get("sha256") != actual_signature["sha256"]
        ):
            raise ValueError(f"HDS {field} differs from the supplied artifact")

    if not same_path(provenance.get("vq_checkpoint"), vq_checkpoint):
        raise ValueError(
            f"HDS VQ checkpoint provenance differs from --vq_checkpoint: "
            f"{provenance.get('vq_checkpoint')} vs {vq_checkpoint}"
        )
    if not same_path(provenance.get("vq_config"), vq_config):
        raise ValueError(
            f"HDS VQ config provenance differs from --vq_config: "
            f"{provenance.get('vq_config')} vs {vq_config}"
        )
    validate_signature("vq_checkpoint_signature", vq_checkpoint)
    validate_signature("vq_config_signature", vq_config)
    if provenance.get("index_reconstructible_from_hds") is False:
        print(
            "[Provenance] alpha-weighted Attn-RVQ is not exactly reconstructible from indices; "
            "diffusion uses the configured index-derived approximation."
        )
    return provenance


def parse_vq_indices_from_tokens(tokens: List[str]) -> Tuple[
    List[int], Dict[int, List[int]], Optional[int], Optional[int]]:
    global_codes: List[int] = []
    local_codes_per_site: Dict[int, List[int]] = {}
    property_bin = None
    num_sites = None

    for tok in tokens:
        m_prop = PROPERTY_BIN_RE.match(tok)
        if m_prop:
            property_bin = int(m_prop.group(1))
            break

    for tok in tokens:
        m_ns = NUM_SITES_RE.match(tok)
        if m_ns:
            num_sites = int(m_ns.group(1))
            break

    in_global = False
    for tok in tokens:
        if tok == "<GLOBAL>":
            in_global = True
            continue
        if tok == "</GLOBAL>":
            in_global = False
            continue
        if in_global:
            parsed = parse_hds_code_token(tok)
            if parsed and parsed[0] == "global":
                _, quantizer, code = parsed
                if quantizer is not None and quantizer != len(global_codes):
                    raise ValueError(
                        f"Global Q stages must be consecutive: got Q{quantizer} at position {len(global_codes)}"
                    )
                global_codes.append(code)

    current_site = None
    for tok in tokens:
        m_start = SITE_START_RE.match(tok)
        if m_start:
            current_site = int(m_start.group(1))
            local_codes_per_site.setdefault(current_site, [])
            continue
        m_end = SITE_END_RE.match(tok)
        if m_end:
            current_site = None
            continue
        if current_site is not None:
            parsed = parse_hds_code_token(tok)
            if parsed and parsed[0] == "local":
                _, quantizer, code = parsed
                expected_q = len(local_codes_per_site[current_site])
                if quantizer is not None and quantizer != expected_q:
                    raise ValueError(
                        f"Site {current_site} local Q stages must be consecutive: "
                        f"got Q{quantizer} at position {expected_q}"
                    )
                local_codes_per_site[current_site].append(code)

    return global_codes, local_codes_per_site, property_bin, num_sites


def parse_formula_from_tokens(tokens: List[str]) -> List[Tuple[str, int]]:
    if tokens.count("<FORMULA>") != 1 or tokens.count("</FORMULA>") != 1:
        raise ValueError("HDS must contain exactly one complete FORMULA block")
    in_formula = False
    parsed: List[Tuple[str, int]] = []
    pending_atom: Optional[str] = None

    for tok in tokens:
        if tok == "<FORMULA>":
            in_formula = True
            continue
        if tok == "</FORMULA>":
            break
        if not in_formula:
            continue

        m_atom = ATOM_RE.match(tok)
        if m_atom:
            pending_atom = m_atom.group(1)
            continue

        m_cnt = COUNT_RE.match(tok)
        if m_cnt and pending_atom is not None:
            count = int(m_cnt.group(1))
            if count <= 0:
                raise ValueError(f"Formula count must be positive for {pending_atom}: {count}")
            parsed.append((pending_atom, count))
            pending_atom = None

    if pending_atom is not None:
        raise ValueError("FORMULA ends with an ATOM token without COUNT")
    return parsed


def parse_site_atoms(tokens: List[str]) -> List[str]:
    """Read the explicit site atom skeleton; no formula-derived repair is allowed."""
    atoms_by_site: Dict[int, str] = {}
    cursor = 0
    while cursor < len(tokens):
        opening = SITE_START_RE.fullmatch(tokens[cursor])
        if opening is None:
            cursor += 1
            continue
        site_index = int(opening.group(1))
        closing = f"</SITE_{site_index}>"
        try:
            end = tokens.index(closing, cursor + 1)
        except ValueError as exc:
            raise ValueError(f"Unclosed site block {site_index}") from exc
        matches = [ATOM_RE.fullmatch(token) for token in tokens[cursor + 1:end]]
        symbols = [match.group(1) for match in matches if match is not None]
        if len(symbols) != 1:
            raise ValueError(f"Site {site_index} must contain exactly one ATOM token")
        if site_index in atoms_by_site:
            raise ValueError(f"Duplicate site block {site_index}")
        atoms_by_site[site_index] = symbols[0]
        cursor = end + 1
    expected_ids = list(range(1, len(atoms_by_site) + 1))
    if sorted(atoms_by_site) != expected_ids:
        raise ValueError("Site ids must be consecutive from 1 to NUM_SITES")
    return [atoms_by_site[index] for index in expected_ids]


def atom_symbols_to_numbers(atom_symbols: List[str]) -> torch.Tensor:
    """Convert validated site atoms to atomic numbers without a fallback element."""
    from pymatgen.core.periodic_table import Element

    if not atom_symbols:
        raise ValueError("HDS contains no site atom skeleton")
    return torch.tensor([int(Element(symbol).Z) for symbol in atom_symbols], dtype=torch.long)


def group_hds_records_by_id(
        records: List[Dict[str, Any]],
        expected_hds_per_condition: int,
) -> List[Dict[str, Any]]:
    """Group candidate HDS rows by material id in their first-seen order.

    Evaluation must never split the five candidates of one material between GPU
    shards.  The strict count check also catches incomplete generation outputs
    before any expensive diffusion sampling starts.
    """
    if expected_hds_per_condition <= 0:
        raise ValueError(
            "--group_hds_by_id requires --hds_per_condition to be greater than zero"
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row_index, record in enumerate(records):
        record_id = str(record.get("prompt_id", record.get("id", f"sample_{row_index}")))
        grouped.setdefault(record_id, []).append(record)

    invalid_counts = {
        record_id: len(rows)
        for record_id, rows in grouped.items()
        if len(rows) != expected_hds_per_condition
    }
    if invalid_counts:
        examples = list(invalid_counts.items())[:8]
        raise ValueError(
            f"expected exactly {expected_hds_per_condition} HDS rows per condition, "
            f"but {len(invalid_counts)} ids differ; examples={examples}"
        )

    return [
        {"id": record_id, "_grouped_hds_records": rows}
        for record_id, rows in grouped.items()
    ]


def _candidate_index(record: Dict[str, Any], fallback: int) -> int:
    """Return the generated HDS candidate rank, falling back to row order."""
    value = record.get("candidate_index", fallback)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid HDS candidate_index={value!r}") from exc


def _record_code_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Read cached real HDS tuples, with token parsing as a legacy fallback."""
    global_codes = record.get("global_codes")
    local_codes = record.get("local_codes")
    atom_symbols = record.get("atom_symbols")

    if isinstance(global_codes, torch.Tensor):
        global_codes = global_codes.detach().cpu().tolist()
    if isinstance(local_codes, torch.Tensor):
        local_codes = local_codes.detach().cpu().tolist()
    if isinstance(atom_symbols, torch.Tensor):
        atom_symbols = atom_symbols.detach().cpu().tolist()

    if global_codes is None or local_codes is None or atom_symbols is None:
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("HDS record has neither cached code tuples nor parseable tokens")
        parsed_global, local_by_site, _, _ = parse_vq_indices_from_tokens(tokens)
        site_ids = sorted(local_by_site)
        global_codes = parsed_global
        local_codes = [local_by_site[site_id] for site_id in site_ids]
        atom_symbols = parse_site_atoms(tokens)

    global_codes = [int(code) for code in global_codes]
    local_codes = [[int(code) for code in row] for row in local_codes]
    atom_symbols = [str(symbol) for symbol in atom_symbols]
    if len(local_codes) != len(atom_symbols):
        raise ValueError(
            "Cached HDS local-code/site mismatch: "
            f"local_rows={len(local_codes)}, atom_symbols={len(atom_symbols)}"
        )
    return {
        "global_codes": global_codes,
        "local_codes": local_codes,
        "atom_symbols": atom_symbols,
    }


def _index_grouped_hds_records(
        grouped_records: List[Dict[str, Any]],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Index every material's real HDS candidates by candidate rank."""
    lookup: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for group in grouped_records:
        sample_id = str(group["id"])
        rows = group.get("_grouped_hds_records")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Grouped HDS entry {sample_id!r} has no candidate rows")
        rank_map: Dict[int, Dict[str, Any]] = {}
        for fallback_rank, row in enumerate(rows):
            rank = _candidate_index(row, fallback_rank)
            if rank in rank_map:
                raise ValueError(f"Duplicate candidate_index={rank} for id={sample_id}")
            rank_map[rank] = row
        lookup[sample_id] = rank_map
    return lookup


def compute_hds_shuffle_source_signature(
        grouped_records: List[Dict[str, Any]],
) -> str:
    """Fingerprint the exact real-code population used by a shuffle manifest."""
    lookup = _index_grouped_hds_records(grouped_records)
    digest = hashlib.sha256()
    for sample_id in sorted(lookup):
        for rank in sorted(lookup[sample_id]):
            payload = _record_code_payload(lookup[sample_id][rank])
            serialized = json.dumps(
                [sample_id, rank, payload],
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(serialized.encode("utf-8"))
    return digest.hexdigest()


def build_real_hds_shuffle_manifest(
        grouped_records: List[Dict[str, Any]],
        seed: int,
) -> Dict[str, Any]:
    """Build deterministic donors using only complete tuples observed in real HDS.

    Global donors are a sample-level cyclic derangement. Local donors are drawn
    from other samples with the same element and, when possible, candidate rank.
    No RVQ tuple is assembled stage by stage.
    """
    lookup = _index_grouped_hds_records(grouped_records)
    sample_ids = sorted(lookup)
    if len(sample_ids) < 2:
        raise ValueError("Shuffled HDS requires at least two material ids")

    rng = random.Random(int(seed))
    cycle = list(sample_ids)
    rng.shuffle(cycle)
    global_donors = {
        target_id: cycle[(position + 1) % len(cycle)]
        for position, target_id in enumerate(cycle)
    }

    rank_element_bank: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    element_bank: Dict[str, List[Dict[str, Any]]] = {}
    payloads: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for sample_id in sample_ids:
        payloads[sample_id] = {}
        for rank, record in lookup[sample_id].items():
            payload = _record_code_payload(record)
            payloads[sample_id][rank] = payload
            for site_index, atom_symbol in enumerate(payload["atom_symbols"]):
                donor = {
                    "sample_id": sample_id,
                    "candidate_index": int(rank),
                    "site_index": int(site_index),
                }
                rank_element_bank.setdefault((rank, atom_symbol), []).append(donor)
                element_bank.setdefault(atom_symbol, []).append(donor)

    local_donors: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rank_fallbacks = 0
    unchanged_no_donor = 0
    for target_id in sample_ids:
        local_donors[target_id] = {}
        global_donor_id = global_donors[target_id]
        target_ranks = set(lookup[target_id])
        donor_ranks = set(lookup[global_donor_id])
        if not target_ranks.issubset(donor_ranks):
            raise ValueError(
                f"Global donor {global_donor_id} lacks candidate ranks for target {target_id}: "
                f"missing={sorted(target_ranks.difference(donor_ranks))}"
            )

        for rank in sorted(lookup[target_id]):
            target_payload = payloads[target_id][rank]
            site_donors: List[Dict[str, Any]] = []
            for atom_symbol in target_payload["atom_symbols"]:
                candidates = [
                    donor
                    for donor in rank_element_bank.get((rank, atom_symbol), [])
                    if donor["sample_id"] != target_id
                ]
                used_rank_fallback = False
                if not candidates:
                    candidates = [
                        donor
                        for donor in element_bank.get(atom_symbol, [])
                        if donor["sample_id"] != target_id
                    ]
                    used_rank_fallback = True
                if not candidates:
                    # Rare elements can occur in only one test material. Keep
                    # that real tuple rather than introducing an invalid code.
                    chosen = {
                        "sample_id": target_id,
                        "candidate_index": int(rank),
                        "site_index": int(len(site_donors)),
                        "unchanged_no_donor": True,
                    }
                    unchanged_no_donor += 1
                else:
                    chosen = dict(rng.choice(candidates))
                chosen["atom_symbol"] = atom_symbol
                chosen["rank_fallback"] = used_rank_fallback
                rank_fallbacks += int(used_rank_fallback)
                site_donors.append(chosen)
            local_donors[target_id][str(rank)] = site_donors

    return {
        "version": 1,
        "seed": int(seed),
        "source_signature": compute_hds_shuffle_source_signature(grouped_records),
        "sample_count": len(sample_ids),
        "global_donors": global_donors,
        "local_donors": local_donors,
        "local_rank_fallbacks": rank_fallbacks,
        "local_unchanged_no_donor": unchanged_no_donor,
    }


def save_hds_shuffle_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Atomically store a reproducible shuffle manifest."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    opener = gzip.open if str(output).endswith(".gz") else open
    with opener(temporary, "wb") as handle:
        pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)


def load_hds_shuffle_manifest(path: str) -> Dict[str, Any]:
    input_path = Path(path)
    opener = gzip.open if str(input_path).endswith(".gz") else open
    with opener(input_path, "rb") as handle:
        manifest = pickle.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid HDS shuffle manifest: {input_path}")
    return manifest


def apply_real_hds_shuffle(
        *,
        sample_id: str,
        hds_records: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]],
        grouped_record_lookup: Dict[str, Dict[int, Dict[str, Any]]],
        manifest: Dict[str, Any],
        mode: str,
        device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Replace global/local conditions with complete real tuples from donors."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in HDS_SHUFFLE_MODES:
        raise ValueError(f"Unknown hds_shuffle_mode={mode!r}")

    shuffled = [
        {
            **condition,
            "indices_g": condition["indices_g"].clone(),
            "indices_l": condition["indices_l"].clone(),
        }
        for condition in conditions
    ]
    metadata: Dict[str, Any] = {
        "mode": normalized_mode,
        "global_donor_id": None,
        "local_donor_ids": [],
        "local_rank_fallbacks": 0,
        "local_unchanged_no_donor": 0,
    }
    if normalized_mode == "matched":
        return shuffled, metadata

    use_global = normalized_mode in {"global", "both"}
    use_local = normalized_mode in {"local", "both"}
    global_donor_id = manifest["global_donors"][sample_id]
    if use_global:
        metadata["global_donor_id"] = global_donor_id

    local_ids = set()
    for fallback_rank, (record, condition) in enumerate(zip(hds_records, shuffled)):
        rank = _candidate_index(record, fallback_rank)
        target_payload = _record_code_payload(record)
        if use_global:
            donor_record = grouped_record_lookup[global_donor_id][rank]
            donor_global = _record_code_payload(donor_record)["global_codes"]
            if len(donor_global) != int(condition["indices_g"].shape[-1]):
                raise ValueError(
                    f"Global donor tuple length mismatch for id={sample_id}, rank={rank}"
                )
            condition["indices_g"] = torch.tensor(
                donor_global, dtype=torch.long, device=device
            ).unsqueeze(0)

        if use_local:
            donor_specs = manifest["local_donors"][sample_id][str(rank)]
            if len(donor_specs) != int(condition["indices_l"].shape[0]):
                raise ValueError(
                    f"Local donor/site count mismatch for id={sample_id}, rank={rank}"
                )
            replacement_rows = []
            for site_index, donor_spec in enumerate(donor_specs):
                target_symbol = target_payload["atom_symbols"][site_index]
                if donor_spec["atom_symbol"] != target_symbol:
                    raise ValueError(
                        f"Local donor element mismatch for id={sample_id}, rank={rank}, "
                        f"site={site_index}: {donor_spec['atom_symbol']} vs {target_symbol}"
                    )
                donor_id = str(donor_spec["sample_id"])
                unchanged_no_donor = bool(donor_spec.get("unchanged_no_donor", False))
                if donor_id == sample_id and not unchanged_no_donor:
                    raise ValueError("Local shuffle donor must differ from the target sample")
                donor_rank = int(donor_spec["candidate_index"])
                donor_site = int(donor_spec["site_index"])
                donor_record = grouped_record_lookup[donor_id][donor_rank]
                donor_payload = _record_code_payload(donor_record)
                if donor_payload["atom_symbols"][donor_site] != target_symbol:
                    raise ValueError("Manifest local donor no longer matches its declared element")
                replacement_rows.append(donor_payload["local_codes"][donor_site])
                if donor_id != sample_id:
                    local_ids.add(donor_id)
                metadata["local_rank_fallbacks"] += int(donor_spec.get("rank_fallback", False))
                metadata["local_unchanged_no_donor"] += int(unchanged_no_donor)
            replacement = torch.tensor(replacement_rows, dtype=torch.long, device=device)
            if replacement.shape != condition["indices_l"].shape:
                raise ValueError(
                    f"Local donor tuple shape mismatch for id={sample_id}, rank={rank}: "
                    f"{tuple(replacement.shape)} vs {tuple(condition['indices_l'].shape)}"
                )
            condition["indices_l"] = replacement

    metadata["local_donor_ids"] = sorted(local_ids)
    return shuffled, metadata


def derive_sampling_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable per-material seed shared by every shuffle condition."""
    digest = hashlib.sha256(f"{int(base_seed)}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2 ** 31)


def build_hds_sampling_condition(
        tokens: List[str],
        *,
        qg: int,
        ql: int,
        strict_code_shape: bool,
        device: torch.device,
) -> Dict[str, Any]:
    """Parse one HDS row into an atom skeleton and aligned VQ index tensors."""
    global_codes, local_codes_per_site, property_bin, num_sites_tok = (
        parse_vq_indices_from_tokens(tokens)
    )
    num_site_tokens = [token for token in tokens if NUM_SITES_RE.fullmatch(token)]
    if len(num_site_tokens) != 1:
        raise ValueError(f"Expected exactly one NUM_SITES token, got {len(num_site_tokens)}")
    if not global_codes or not local_codes_per_site:
        raise ValueError("Missing GLOBAL/LOCAL code tokens")

    if len(global_codes) != qg:
        raise ValueError(
            f"Global code length mismatch: got {len(global_codes)}, expected {qg}"
        )

    site_ids = sorted(local_codes_per_site.keys())
    if num_sites_tok is None or num_sites_tok <= 0:
        raise ValueError("Missing or invalid NUM_SITES token")
    n_sites = num_sites_tok
    if site_ids != list(range(1, n_sites + 1)):
        raise ValueError(
            f"Site blocks must be exactly 1..{n_sites}, got {site_ids[:10]}"
        )
    local_rows = []
    for site_id in range(1, n_sites + 1):
        row = local_codes_per_site.get(site_id, [])
        if len(row) != ql:
            raise ValueError(
                f"Local code length mismatch at site {site_id}: got {len(row)}, expected {ql}"
            )
        local_rows.append(list(row))
    if not local_rows:
        raise ValueError("No valid local rows")

    parsed_formula = parse_formula_from_tokens(tokens)
    atom_symbols = parse_site_atoms(tokens)
    if len(atom_symbols) != n_sites:
        raise ValueError(
            f"Site atom count mismatch: atoms={len(atom_symbols)}, NUM_SITES={n_sites}"
        )
    from collections import Counter

    formula_counter = Counter({symbol: count for symbol, count in parsed_formula})
    if len(formula_counter) != len(parsed_formula):
        raise ValueError("FORMULA contains duplicate elements")
    if formula_counter != Counter(atom_symbols):
        raise ValueError(
            f"FORMULA/site composition mismatch: formula={dict(formula_counter)}, "
            f"sites={dict(Counter(atom_symbols))}"
        )

    # Q-tagged E2 output must equal the compiler's canonical form exactly.
    if any(token.startswith("<STAGE_") or "_CODE_Q" in token for token in tokens):
        compiler = HDSCompiler(CrystalVocabConfig(
            num_quantizers_global=qg,
            codebooksize_global=max(global_codes) + 1,
            num_quantizers_local=ql,
            codebooksize_local=max(code for row in local_rows for code in row) + 1,
            max_sites=max(128, n_sites),
            max_formula_count=max(100, n_sites),
            include_hds_stage_tokens=True,
        ))
        compiled = compiler.parse_strict_record(tokens)
        parsed_formula = compiled.composition_counts
        atom_symbols = compiled.atom_symbols
        global_codes = compiled.global_codes
        local_rows = compiled.local_codes

    atom_types = atom_symbols_to_numbers(atom_symbols)
    return {
        "formula_composition": parsed_formula,
        "atom_symbols": atom_symbols,
        "property_bin": property_bin,
        "atom_types": atom_types,
        "indices_g": torch.tensor(global_codes, dtype=torch.long, device=device).unsqueeze(0),
        "indices_l": torch.tensor(local_rows, dtype=torch.long, device=device),
    }


def parse_record_range(record_range: str, total_records: int) -> Tuple[int, int]:
    """
    Parse an inclusive record range for selecting DSS records.

    Supported formats:
      - "0,999"
      - "[0,999]"
      - "0:999"

    Returns:
        (start_idx, end_idx), where end_idx is inclusive.
    """
    s = str(record_range).strip()
    if not s:
        raise ValueError("empty record_range")

    # Allow bracketed or parenthesized forms, e.g. [0,999] or (0,999).
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()

    # Also allow Python-slice-like syntax with an inclusive end index.
    s = s.replace(":", ",")

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("record_range must be like 'start,end', '[start,end]', or 'start:end'")

    try:
        start_idx = int(parts[0])
        end_idx = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"record_range indices must be integers, got: {record_range!r}") from exc

    if total_records <= 0:
        raise ValueError("no records available")
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid range: start={start_idx}, end={end_idx}")
    if start_idx >= total_records:
        raise ValueError(f"start_idx={start_idx} exceeds total records={total_records}")

    if end_idx >= total_records:
        print(
            f"Warning: end_idx={end_idx} exceeds total records={total_records}, "
            f"clamping to {total_records - 1}"
        )
        end_idx = total_records - 1

    return start_idx, end_idx


def load_models(
        cond_config_path: str,
        cond_checkpoint_path: str,
        vq_checkpoint_path: str,
        vq_config_path: str,
        device: torch.device,
):
    project_root = Path(__file__).resolve().parent.parent
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import yaml
    from model.cond_diffusion import CondDiffusion
    from model.attn_rvq import CrystalVQVAE

    def load_yaml_config_local(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def build_model_from_config_local(config: dict, checkpoint: dict):
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
        filtered_state = {
            k: v for k, v in state_dict.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
        }
        model.load_state_dict(filtered_state, strict=False)
        return model

    with open(cond_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", cfg)

    cond_model = CondDiffusion(model_cfg).to(device)
    cond_ckpt = torch.load(cond_checkpoint_path, map_location=device, weights_only=False)
    cond_state = cond_ckpt.get("model_state_dict", cond_ckpt.get("state_dict", cond_ckpt))
    cond_model.load_state_dict(cond_state, strict=False)
    cond_model.eval()

    vq_cfg = load_yaml_config_local(vq_config_path)
    vq_ckpt = torch.load(vq_checkpoint_path, map_location=device, weights_only=False)
    vq_model = build_model_from_config_local(vq_cfg, vq_ckpt)
    vq_model.eval()

    cond_model.pretrained_vqvae_model = vq_model
    return cond_model, model_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load DSS-seq generated by nanoGPT, parse formula + discrete VQ indices, "
            "and sample atom coordinates with pretrained CondDiffusion."
        )
    )
    parser.add_argument("--dss_path", type=str, default="", help="Input DSS sequences (pkl/pkl.gz/jsonl).")
    parser.add_argument("--cond_config", type=str, required=True, help="CondDiffusion config yaml.")
    parser.add_argument("--cond_checkpoint", type=str, required=True, help="Pretrained CondDiffusion checkpoint.")
    parser.add_argument("--vq_checkpoint", type=str, required=True, help="Pretrained CrystalVQVAE checkpoint.")
    parser.add_argument("--vq_config", type=str, required=True, help="CrystalVQVAE training config yaml.")
    parser.add_argument("--out_path", type=str, default="results/dss_samples/sampled_from_dss_1.csv")
    parser.add_argument("--n_samples", type=int, default=5, help="Diffusion samples per HDS condition.")
    parser.add_argument(
        "--group_hds_by_id",
        action="store_true",
        help=(
            "Group every HDS row sharing an id and diffuse all HDS/sample pairs "
            "in one parallel batch. In this mode --record_range selects material ids."
        ),
    )
    parser.add_argument(
        "--hds_per_condition",
        type=int,
        default=0,
        help="Required number of candidate HDS rows per id in grouped mode.",
    )
    parser.add_argument(
        "--candidate_rank",
        type=int,
        default=-1,
        help=(
            "If non-negative, keep only HDS rows whose candidate_index equals this rank "
            "before grouped sampling. Useful for rank-0 1x20 comparisons."
        ),
    )
    parser.add_argument(
        "--hds_shuffle_mode",
        choices=sorted(HDS_SHUFFLE_MODES),
        default="matched",
        help=(
            "Evaluation-time HDS correspondence: matched, shuffled global tuple, "
            "element-matched shuffled local tuples, or both."
        ),
    )
    parser.add_argument(
        "--hds_shuffle_seed",
        type=int,
        default=42,
        help="Seed used only to construct the deterministic real-tuple donor manifest.",
    )
    parser.add_argument(
        "--hds_shuffle_manifest",
        type=str,
        default="",
        help=(
            "Shared .pkl/.pkl.gz donor manifest. For shuffled modes it is loaded "
            "when present or deterministically created from the complete input HDS."
        ),
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=-1,
        help=(
            "If non-negative, derive a stable per-material diffusion seed. Use the "
            "same value across matched/global/local/both for paired sampling noise."
        ),
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help=(
            "Do not write generated CIF strings to the CSV. Requires evaluation "
            "metrics and writes an aggregate .metrics.json sidecar."
        ),
    )
    parser.add_argument("--cond_scale", type=float, default=2.0)
    parser.add_argument("--step_lr", type=float, default=1e-5)
    parser.add_argument("--max_records", type=int, default=0, help="If >0, only process first N DSS records.")
    parser.add_argument(
        "--record_range",
        type=str,
        default="",
        help=(
            "Process records in an inclusive index range. "
            "Supports 'start,end', '[start,end]', or 'start:end'. "
            "For example, '[0,999]' processes the first 1000 records. "
            "Overrides --max_records if set."
        ),
    )
    parser.add_argument("--skip_invalid", action="store_true", help="Skip malformed DSS records instead of raising.")
    parser.add_argument("--save_skipped_ids", type=str, default="",
                        help="If set, save skipped IDs to a txt file (one ID per line).")
    parser.add_argument("--save_skipped_reasons", type=str, default="",
                        help="If set, save skipped IDs and reasons (.csv or .txt).")
    parser.add_argument("--use_property_condition", action="store_true",
                        help="Use PROPERTY_BIN from DSS as extra condition.")
    parser.add_argument("--property_bin_override", type=int, default=-1,
                        help="If >=0, override DSS property bin for all records.")
    parser.add_argument("--expected_global_codes", type=int, default=0,
                        help="If >0, require exactly this many global codes; no padding/truncation is performed.")
    parser.add_argument("--expected_local_codes", type=int, default=0,
                        help="If >0, require exactly this many local codes per site; no repair is performed.")
    parser.add_argument("--strict_code_shape", action="store_true",
                        help="Deprecated compatibility flag; exact code shape is now always enforced.")
    parser.add_argument("--enable_online_metrics", action="store_true",
                        help="(Deprecated alias) Enable evaluation metrics during sampling.")
    parser.add_argument(
        "--compute_eval_metrics",
        action="store_true",
        help="Enable computation of evaluation metrics (validity/coverage/wdist) during sampling.",
    )
    parser.add_argument(
        "--compute_match_metrics",
        action="store_true",
        help="Compute per-reference match rate and minimum RMS over each sample's candidates.",
    )
    parser.add_argument(
        "--no_compute_eval_metrics",
        action="store_true",
        help="Force disable evaluation metrics even if --compute_eval_metrics/--enable_online_metrics is set.",
    )
    parser.add_argument("--reference_test_csv", type=str, default="",
                        help="Test-set CSV (with CIF column) for online coverage/property metrics.")
    parser.add_argument("--reference_test_cif_col", type=str, default="cif",
                        help="CIF column name in reference_test_csv.")
    parser.add_argument("--reference_test_id_col", type=str, default="material_id",
                        help="Reference-test ID column used by --compute_match_metrics.")
    parser.add_argument(
        "--merge_metric_states",
        nargs="+",
        default=None,
        help="Merge metric-state .npz files from disjoint sampling shards and exit.",
    )
    parser.add_argument(
        "--merge_metrics_output",
        type=str,
        default="",
        help="Output JSON path used with --merge_metric_states.",
    )
    parser.add_argument(
        "--merge_match_metric_states",
        nargs="+",
        default=None,
        help="Merge additive match/RMS state JSON files from disjoint sampling shards and exit.",
    )
    parser.add_argument(
        "--merge_match_metrics_output",
        type=str,
        default="",
        help="Output JSON path used with --merge_match_metric_states.",
    )
    parser.add_argument("--cov_struc_cutoff", type=float, default=0.4, help="Coverage structural cutoff for dS.")
    parser.add_argument("--cov_comp_cutoff", type=float, default=10.0, help="Coverage compositional cutoff for dC.")
    parser.add_argument("--metrics_update_interval", type=int, default=20,
                        help="Update heavy online metrics every N processed samples.")
    parser.add_argument(
        "--suppress_pymatgen_warnings",
        action="store_true",
        help="Suppress verbose pymatgen UserWarning logs during CIF parsing/metrics initialization.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--calc_formation_energy", action="store_true",
                        help="Calculate formation energy using CHGNet during sampling.")
    parser.add_argument("--calc_bandgap", action="store_true", help="Calculate bandgap using MEGNet during sampling.")
    parser.add_argument(
        "--bandgap_relax",
        action="store_true",
        help="Also compute bandgap after CHGNet relaxation when possible.",
    )
    parser.add_argument(
        "--bandgap_relax_fmax",
        type=float,
        default=0.02,
        help="CHGNet relaxation force convergence for bandgap calc.",
    )
    parser.add_argument(
        "--bandgap_relax_steps",
        type=int,
        default=300,
        help="CHGNet relaxation max steps for bandgap calc.",
    )
    parser.add_argument(
        "--bandgap_relax_save_dir",
        type=str,
        default="",
        help="If set, save CHGNet-relaxed structures used for bandgap.",
    )
    parser.add_argument("--mp_api_key", type=str, default="", help="Materials Project API key for FE calc via CHGNet.")
    return parser.parse_args()


def merge_metric_states(args) -> None:
    """Recompute global metrics from lossless generated-side shard states."""
    if not args.reference_test_csv:
        raise ValueError("--merge_metric_states requires --reference_test_csv")
    if not args.merge_metrics_output:
        raise ValueError("--merge_metrics_output is required with --merge_metric_states")

    from scripts.hds_eval_metrics import OnlineDssMetrics

    ref_df = pd.read_csv(args.reference_test_csv)
    if args.reference_test_cif_col not in ref_df.columns:
        raise ValueError(f"Missing CIF column '{args.reference_test_cif_col}' in {args.reference_test_csv}")
    ref_structs = [
        structure for structure in (
            parse_structure_from_cif_str(value)
            for value in ref_df[args.reference_test_cif_col].tolist()
        )
        if structure is not None
    ]
    metrics = OnlineDssMetrics(
        reference_structs=ref_structs,
        struc_cutoff=args.cov_struc_cutoff,
        comp_cutoff=args.cov_comp_cutoff,
    )
    required = set(metrics.export_generated_state())
    states = []
    for state_path in args.merge_metric_states:
        with np.load(state_path) as loaded:
            missing = required.difference(loaded.files)
            if missing:
                raise ValueError(f"Metric state {state_path} is missing keys: {sorted(missing)}")
            states.append({key: np.asarray(loaded[key]) for key in required})
    metrics.import_merged_generated_states(states)

    output = Path(args.merge_metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = metrics.summary()
    summary.update({
        "metric_state_shards": [str(Path(path)) for path in args.merge_metric_states],
        "generated_structures_saved": False,
    })
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Merged] {len(states)} metric states -> {output}")


def merge_match_metric_states(args) -> None:
    """Merge additive match/RMS statistics emitted by disjoint sampling shards."""
    if not args.merge_match_metrics_output:
        raise ValueError("--merge_match_metrics_output is required with --merge_match_metric_states")

    total = {"evaluated": 0, "matched": 0, "rms_sum": 0.0, "rms_count": 0}
    experiment_metadata = None
    for state_path in args.merge_match_metric_states:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        state_metadata = {
            "hds_shuffle_mode": state.get("hds_shuffle_mode", "matched"),
            "hds_shuffle_seed": state.get("hds_shuffle_seed"),
            "hds_shuffle_manifest_sha256": state.get("hds_shuffle_manifest_sha256"),
            "sampling_seed": state.get("sampling_seed"),
            "local_unchanged_no_donor": state.get("local_unchanged_no_donor", 0),
        }
        if experiment_metadata is None:
            experiment_metadata = state_metadata
        elif state_metadata != experiment_metadata:
            raise ValueError(
                "Cannot merge match metric states from different HDS shuffle/sampling settings"
            )
        for key in ("evaluated", "matched", "rms_count"):
            total[key] += int(state.get(key, 0))
        total["rms_sum"] += float(state.get("rms_sum", 0.0))

    summary = {
        "match_rate": float(total["matched"] / total["evaluated"]) if total["evaluated"] else 0.0,
        "rms_dist": float(total["rms_sum"] / total["rms_count"]) if total["rms_count"] else None,
        "evaluated_condition_ids": total["evaluated"],
        "matched_condition_ids": total["matched"],
        "rms_count": total["rms_count"],
        "metric_state_shards": [str(Path(path)) for path in args.merge_match_metric_states],
        "generated_structures_saved": False,
        **(experiment_metadata or {}),
    }
    output = Path(args.merge_match_metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Merged] {len(args.merge_match_metric_states)} match metric states -> {output}")


def main():
    args = parse_args()
    if args.merge_metric_states:
        merge_metric_states(args)
        return
    if args.merge_match_metric_states:
        merge_match_metric_states(args)
        return
    if not args.dss_path:
        raise ValueError("--dss_path is required unless --merge_metric_states is used")
    compute_eval_metrics = bool(
        (args.compute_eval_metrics or args.enable_online_metrics) and (not args.no_compute_eval_metrics))
    compute_match_metrics = bool(args.compute_match_metrics)
    if args.metrics_only and not (compute_eval_metrics or compute_match_metrics):
        raise ValueError("--metrics_only requires --compute_eval_metrics or --compute_match_metrics")
    if args.group_hds_by_id and args.n_samples <= 0:
        raise ValueError("--n_samples must be greater than zero")
    if args.hds_shuffle_mode != "matched":
        if not args.group_hds_by_id:
            raise ValueError("Shuffled HDS evaluation requires --group_hds_by_id")
        if not args.hds_shuffle_manifest:
            raise ValueError("Shuffled HDS evaluation requires --hds_shuffle_manifest")

    if args.suppress_pymatgen_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\\..*")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    chgnet_model = None
    if args.calc_formation_energy:
        process_dir = str(Path(project_root) / "results" / "random_dss_samples")
        if process_dir not in sys.path:
            sys.path.insert(0, process_dir)
        try:
            import process_chgnet
            if args.mp_api_key:
                process_chgnet.API_KEY = args.mp_api_key
            print("[PropCalc] Loading CHGNet formation energy model...")
            chgnet_model = process_chgnet.CHGNet.load()
            process_chgnet.preload_all_ref_energies(chgnet_model)
        except Exception as e:
            print(f"[PropCalc] Failed to load CHGNet formation energy calculation: {e}")
            chgnet_model = None

    megnet_model = None
    if args.calc_bandgap:
        print("[PropCalc] Loading MEGNet band gap model...")
        import os
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        try:
            try:
                import tensorflow.keras.models as kmodels
                original_load_model = kmodels.load_model

                def patched_load_model(*args, **kwargs):
                    kwargs["compile"] = False
                    return original_load_model(*args, **kwargs)

                kmodels.load_model = patched_load_model
                import keras.src.saving.saving_api as saving_api
                original_saving_api_load = saving_api.load_model

                def patched_saving_api_load_model(*args, **kwargs):
                    kwargs["compile"] = False
                    return original_saving_api_load(*args, **kwargs)

                saving_api.load_model = patched_saving_api_load_model
            except Exception:
                pass
            from megnet.utils.models import load_model
            megnet_model = load_model("Bandgap_MP_2018")
        except Exception as e:
            print(f"[PropCalc] Failed to load MEGNet band gap model: {e}")
            megnet_model = None

    bandgap_relaxer = None
    if args.calc_bandgap and args.bandgap_relax:
        try:
            from chgnet.model.dynamics import StructOptimizer
            from chgnet.model.model import CHGNet

            print("[PropCalc] Loading CHGNet relaxer for band gap...")
            bandgap_model = CHGNet.load()
            bandgap_relaxer = StructOptimizer(model=bandgap_model)
        except Exception as e:
            print(f"[PropCalc] Failed to load CHGNet relaxer for band gap: {e}")
            bandgap_relaxer = None

    bandgap_relax_save_dir = None
    if args.bandgap_relax_save_dir:
        bandgap_relax_save_dir = Path(args.bandgap_relax_save_dir)
        bandgap_relax_save_dir.mkdir(parents=True, exist_ok=True)

    cond_model, model_cfg = load_models(
        cond_config_path=args.cond_config,
        cond_checkpoint_path=args.cond_checkpoint,
        vq_checkpoint_path=args.vq_checkpoint,
        vq_config_path=args.vq_config,
        device=device,
    )

    records = load_records(args.dss_path)
    if args.candidate_rank >= 0:
        raw_count = len(records)
        records = [
            record
            for record in records
            if int(record.get("candidate_index", -1)) == args.candidate_rank
        ]
        if not records:
            raise ValueError(
                f"No HDS records with candidate_index={args.candidate_rank} in {args.dss_path}"
            )
        print(
            f"Selected candidate rank {args.candidate_rank}: "
            f"{len(records)}/{raw_count} HDS rows"
        )
    hds_attn_rvq_provenance = validate_attn_rvq_provenance(
        records,
        vq_checkpoint=args.vq_checkpoint,
        vq_config=args.vq_config,
    )
    raw_hds_record_count = len(records)
    if args.group_hds_by_id:
        records = group_hds_records_by_id(
            records,
            expected_hds_per_condition=args.hds_per_condition,
        )
        print(
            f"Grouped {raw_hds_record_count} HDS rows into {len(records)} material ids "
            f"({args.hds_per_condition} HDS x {args.n_samples} diffusion samples each)"
        )

    grouped_record_lookup = None
    shuffle_manifest = None
    shuffle_manifest_sha256 = None
    if args.hds_shuffle_mode != "matched":
        grouped_record_lookup = _index_grouped_hds_records(records)
        source_signature = compute_hds_shuffle_source_signature(records)
        manifest_path = Path(args.hds_shuffle_manifest)
        if manifest_path.is_file():
            shuffle_manifest = load_hds_shuffle_manifest(str(manifest_path))
            print(f"[HDS Shuffle] Loaded donor manifest: {manifest_path}")
        else:
            shuffle_manifest = build_real_hds_shuffle_manifest(
                records,
                seed=args.hds_shuffle_seed,
            )
            save_hds_shuffle_manifest(shuffle_manifest, str(manifest_path))
            print(f"[HDS Shuffle] Created donor manifest: {manifest_path}")
        if int(shuffle_manifest.get("version", -1)) != 1:
            raise ValueError("Unsupported HDS shuffle manifest version")
        if int(shuffle_manifest.get("seed", -1)) != int(args.hds_shuffle_seed):
            raise ValueError(
                "HDS shuffle manifest seed differs from --hds_shuffle_seed: "
                f"{shuffle_manifest.get('seed')} vs {args.hds_shuffle_seed}"
            )
        if shuffle_manifest.get("source_signature") != source_signature:
            raise ValueError("HDS shuffle manifest was built from a different HDS population")
        if int(shuffle_manifest.get("sample_count", -1)) != len(records):
            raise ValueError("HDS shuffle manifest sample count differs from the complete input")
        shuffle_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        print(
            "[HDS Shuffle] "
            f"mode={args.hds_shuffle_mode}, seed={args.hds_shuffle_seed}, "
            f"samples={len(records)}, local_rank_fallbacks="
            f"{shuffle_manifest.get('local_rank_fallbacks', 0)}, "
            f"local_unchanged_no_donor="
            f"{shuffle_manifest.get('local_unchanged_no_donor', 0)}, "
            f"manifest_sha256={shuffle_manifest_sha256}"
        )

    # 处理记录范围：--record_range 优先于 --max_records，end_idx 为闭区间。
    if args.record_range:
        try:
            start_idx, end_idx = parse_record_range(args.record_range, total_records=len(records))
            records = records[start_idx:end_idx + 1]  # inclusive end_idx
            print(f"Processing records [{start_idx}, {end_idx}] ({len(records)} records)")
        except Exception as e:
            raise ValueError(f"Invalid record_range '{args.record_range}': {e}")
    elif args.max_records and args.max_records > 0:
        records = records[: args.max_records]
        print(f"Processing first {len(records)} records")
    else:
        print(f"Processing all {len(records)} records")

    selected_condition_count = len(records)
    selected_hds_record_count = (
        selected_condition_count * args.hds_per_condition
        if args.group_hds_by_id
        else selected_condition_count
    )

    qg = args.expected_global_codes
    ql = args.expected_local_codes
    if qg <= 0:
        qg = int(getattr(cond_model.pretrained_vqvae_model, "num_quantizers_global", 0) or 0)
        if qg <= 0:
            qg = int(getattr(cond_model.pretrained_vqvae_model, "num_quantizers", 1) or 1)
    if ql <= 0:
        ql = int(getattr(cond_model.pretrained_vqvae_model, "num_quantizers_local", 0) or 0)
        if ql <= 0:
            ql = int(getattr(cond_model.pretrained_vqvae_model, "num_quantizers", 1) or 1)

    # Optional conversion to CIF strings.
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor

        def to_cif_str(x):
            if isinstance(x, Structure):
                return x.to(fmt="cif")
            if hasattr(x, "get_chemical_symbols"):
                return AseAtomsAdaptor.get_structure(x).to(fmt="cif")
            return None

    except Exception:
        def to_cif_str(x):
            return None

    results: List[Dict[str, Any]] = []
    skipped: List[str] = []
    skipped_reasons: List[Dict[str, str]] = []

    online_metrics_total = None
    online_metrics_window = None
    if compute_eval_metrics:
        from scripts.hds_eval_metrics import OnlineDssMetrics

        t0 = time.time()
        if not args.reference_test_csv:
            raise ValueError("--compute_eval_metrics requires --reference_test_csv")
        print(f"[OnlineMetrics] Loading reference CSV: {args.reference_test_csv}")
        ref_df = pd.read_csv(args.reference_test_csv)
        if args.reference_test_cif_col not in ref_df.columns:
            raise ValueError(f"Missing CIF column '{args.reference_test_cif_col}' in {args.reference_test_csv}")
        cif_values = ref_df[args.reference_test_cif_col].tolist()
        print(f"[OnlineMetrics] Parsing reference CIFs: {len(cif_values)} rows")
        ref_structs = []
        parse_iter = tqdm(cif_values, total=len(cif_values), desc="Parsing reference CIFs", unit="cif", leave=False)
        for s in parse_iter:
            st = parse_structure_from_cif_str(s)
            if st is not None:
                ref_structs.append(st)
        print(f"[OnlineMetrics] Parsed reference structures: {len(ref_structs)}/{len(cif_values)}")
        print("[OnlineMetrics] Building feature cache...")
        online_metrics_total = OnlineDssMetrics(
            reference_structs=ref_structs,
            struc_cutoff=args.cov_struc_cutoff,
            comp_cutoff=args.cov_comp_cutoff,
            show_feature_cache_progress=True,
            feature_cache_desc="[OnlineMetrics] Building feature cache (total)",
        )
        online_metrics_window = OnlineDssMetrics(
            reference_structs=ref_structs,
            struc_cutoff=args.cov_struc_cutoff,
            comp_cutoff=args.cov_comp_cutoff,
            show_feature_cache_progress=False,
        )
        print(f"[OnlineMetrics] Ready in {time.time() - t0:.1f}s")

    match_reference_by_id = {}
    match_metric_state = None
    matcher = None
    diffusion_steps = int(model_cfg.get("timesteps", 1000))
    if compute_match_metrics:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from scripts.hds_eval_metrics import structure_validity

        if not args.reference_test_csv:
            raise ValueError("--compute_match_metrics requires --reference_test_csv")
        ref_df = pd.read_csv(args.reference_test_csv)
        required_cols = {args.reference_test_id_col, args.reference_test_cif_col}
        missing_cols = required_cols.difference(ref_df.columns)
        if missing_cols:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing_cols)}")
        for _, row in ref_df.iterrows():
            reference = parse_structure_from_cif_str(row[args.reference_test_cif_col])
            if reference is not None:
                match_reference_by_id[str(row[args.reference_test_id_col])] = reference
        if not match_reference_by_id:
            raise ValueError("No parseable reference structures were found for match metrics")
        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
        match_metric_state = {"evaluated": 0, "matched": 0, "rms_sum": 0.0, "rms_count": 0}
        print(f"[MatchMetrics] Loaded {len(match_reference_by_id)} reference structures")

    pbar = tqdm(enumerate(records), total=len(records), desc="Sampling from DSS", unit="sample")
    for idx, rec in pbar:
        sample_id = str(rec.get("id", rec.get("prompt_id", f"sample_{idx}")))

        try:
            hds_records = rec.get("_grouped_hds_records", [rec])
            conditions = []
            for hds_index, hds_record in enumerate(hds_records):
                tokens = hds_record.get("tokens", None)
                if not isinstance(tokens, list) or not tokens:
                    raise ValueError(f"Invalid or empty tokens at HDS candidate {hds_index}")
                conditions.append(
                    build_hds_sampling_condition(
                        tokens,
                        qg=qg,
                        ql=ql,
                        strict_code_shape=args.strict_code_shape,
                        device=device,
                    )
                )

            # Candidate HDS must share the material's composition and site order.
            first_condition = conditions[0]
            parsed_formula = first_condition["formula_composition"]
            parsed_property_bin = first_condition["property_bin"]
            atom_types = first_condition["atom_types"]
            for hds_index, condition in enumerate(conditions[1:], start=1):
                if condition["formula_composition"] != parsed_formula:
                    raise ValueError(f"HDS candidate {hds_index} has a different formula for id={sample_id}")
                if not torch.equal(condition["atom_types"], atom_types):
                    raise ValueError(f"HDS candidate {hds_index} has a different atom skeleton for id={sample_id}")
                if condition["indices_l"].shape != first_condition["indices_l"].shape:
                    raise ValueError(f"HDS candidate {hds_index} has a different local-code shape for id={sample_id}")

            shuffle_metadata = {
                "mode": "matched",
                "global_donor_id": None,
                "local_donor_ids": [],
                "local_rank_fallbacks": 0,
                "local_unchanged_no_donor": 0,
            }
            if args.hds_shuffle_mode != "matched":
                conditions, shuffle_metadata = apply_real_hds_shuffle(
                    sample_id=sample_id,
                    hds_records=hds_records,
                    conditions=conditions,
                    grouped_record_lookup=grouped_record_lookup,
                    manifest=shuffle_manifest,
                    mode=args.hds_shuffle_mode,
                    device=device,
                )
                first_condition = conditions[0]

            grouped_mode = bool(args.group_hds_by_id)
            diffusion_repeats = args.n_samples
            if grouped_mode:
                # HDS-major layout: HDS_0 x 4, HDS_1 x 4, ... HDS_4 x 4.
                # Passing n_samples=1 preserves this pre-expanded 5 x 4 batch.
                data_list: List[Data] = []
                global_blocks: List[torch.Tensor] = []
                local_blocks: List[torch.Tensor] = []
                for condition in conditions:
                    data = Data(
                        atom_types=condition["atom_types"],
                        natoms=torch.tensor([condition["atom_types"].shape[0]], dtype=torch.long),
                        num_nodes=int(condition["atom_types"].shape[0]),
                    )
                    data_list.extend(data.clone() for _ in range(diffusion_repeats))
                    global_blocks.append(condition["indices_g"].repeat(diffusion_repeats, 1))
                    local_blocks.append(condition["indices_l"].repeat(diffusion_repeats, 1))
                batch = Batch.from_data_list(data_list).to(device)
                expanded_indices_g = torch.cat(global_blocks, dim=0)
                expanded_indices_l = torch.cat(local_blocks, dim=0)
                indices_g = torch.cat([condition["indices_g"] for condition in conditions], dim=0)
                indices_l = torch.stack([condition["indices_l"] for condition in conditions], dim=0)
                sample_call_n = 1
            else:
                data = Data(
                    atom_types=atom_types,
                    natoms=torch.tensor([atom_types.shape[0]], dtype=torch.long),
                    num_nodes=int(atom_types.shape[0]),
                )
                batch = Batch.from_data_list([data]).to(device)
                expanded_indices_g = first_condition["indices_g"]
                expanded_indices_l = first_condition["indices_l"]
                indices_g = expanded_indices_g
                indices_l = expanded_indices_l
                sample_call_n = diffusion_repeats

            cond_inputs: List[torch.Tensor] = [expanded_indices_g, expanded_indices_l]
            used_property_bin = None
            if args.use_property_condition:
                property_bins = []
                for condition in conditions:
                    prop_bin = condition["property_bin"]
                    if args.property_bin_override >= 0:
                        prop_bin = int(args.property_bin_override)
                    property_bins.append(prop_bin)
                if any(value is not None and value >= 0 for value in property_bins):
                    if not all(value is not None and value >= 0 for value in property_bins):
                        raise ValueError(f"HDS candidates have inconsistent property bins for id={sample_id}")
                    used_property_bin = int(property_bins[0])
                    if grouped_mode:
                        expanded_property_bins = torch.tensor(
                            property_bins, dtype=torch.long, device=device
                        ).repeat_interleave(diffusion_repeats)
                    else:
                        expanded_property_bins = torch.tensor(
                            [used_property_bin], dtype=torch.long, device=device
                        )
                    cond_inputs.append(expanded_property_bins)

            sample_seed = None
            if args.sampling_seed >= 0:
                sample_seed = derive_sampling_seed(args.sampling_seed, sample_id)
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)

            sampled = cond_model.sample(
                batch=batch,
                cond_inputs=cond_inputs,
                n_samples=sample_call_n,
                cond_scale=args.cond_scale,
                step_lr=args.step_lr,
                return_trajectory=False,
                stream=False,
            )

            sampled_list = sampled if isinstance(sampled, list) else [sampled]
            expected_sample_count = len(conditions) * diffusion_repeats
            if len(sampled_list) != expected_sample_count:
                raise ValueError(
                    f"diffusion returned {len(sampled_list)} structures, expected {expected_sample_count}"
                )
            cif_list = [c for c in (to_cif_str(x) for x in sampled_list) if c is not None]

            res_record = {
                "id": sample_id,
                "formula_composition": parsed_formula,
                "indices_g": indices_g.cpu(),
                "indices_l": indices_l.cpu(),
                "property_bin": parsed_property_bin,
                "used_property_bin": used_property_bin,
                "hds_conditions": len(conditions),
                "diffusion_samples_per_hds": diffusion_repeats,
                "parallel_diffusion_batch_size": expected_sample_count,
                "attn_rvq_provenance": hds_attn_rvq_provenance,
                "hds_shuffle_mode": args.hds_shuffle_mode,
                "hds_shuffle_seed": (
                    args.hds_shuffle_seed if args.hds_shuffle_mode != "matched" else None
                ),
                "hds_shuffle_manifest_sha256": shuffle_manifest_sha256,
                "sampling_seed": args.sampling_seed if args.sampling_seed >= 0 else None,
                "sample_seed": sample_seed,
                "global_donor_id": shuffle_metadata["global_donor_id"],
                "local_donor_ids": shuffle_metadata["local_donor_ids"],
                "local_rank_fallbacks": shuffle_metadata["local_rank_fallbacks"],
                "local_unchanged_no_donor": shuffle_metadata["local_unchanged_no_donor"],
            }
            if not args.metrics_only:
                res_record["generated_cifs"] = cif_list

            calc_info = {}
            # Optional Online Property Calculation
            if len(cif_list) > 0:
                first_cif = cif_list[0]
                struct_for_prop = parse_structure_from_cif_str(first_cif)
                if struct_for_prop is not None:
                    if chgnet_model is not None:
                        try:
                            import process_chgnet
                            fe_res = process_chgnet.chgnet_calc_eform(struct_for_prop, chgnet_model)
                            res_record["E_unrelaxed"] = fe_res.get("E_unrelaxed")
                            res_record["E_relaxed"] = fe_res.get("E_relaxed")
                            if res_record["E_unrelaxed"] is not None:
                                calc_info["E_unrel"] = f"{res_record['E_unrelaxed']:.3f}"
                            if res_record["E_relaxed"] is not None:
                                calc_info["E_rel"] = f"{res_record['E_relaxed']:.3f}"
                        except Exception:
                            pass
                    if megnet_model is not None:
                        try:
                            bg_pred = megnet_model.predict_structure(struct_for_prop)
                            bg_val = float(np.ravel(bg_pred)[0])
                            res_record["BandGap"] = bg_val
                            res_record["BandGap_unrelaxed"] = bg_val
                            calc_info["BG"] = f"{bg_val:.3f}"
                        except Exception:
                            pass
                    if megnet_model is not None and bandgap_relaxer is not None:
                        try:
                            relax_info = bandgap_relaxer.relax(
                                struct_for_prop,
                                fmax=args.bandgap_relax_fmax,
                                steps=args.bandgap_relax_steps,
                                verbose=False,
                            )
                            relaxed_struct = relax_info["final_structure"]
                            bg_pred_rel = megnet_model.predict_structure(relaxed_struct)
                            bg_val_rel = float(np.ravel(bg_pred_rel)[0])
                            res_record["BandGap_relaxed"] = bg_val_rel
                            calc_info["BG_rel"] = f"{bg_val_rel:.3f}"
                            if bandgap_relax_save_dir is not None:
                                out_name = f"{sample_id}_bg_relaxed.cif"
                                out_path = bandgap_relax_save_dir / out_name
                                out_path.write_text(relaxed_struct.to(fmt="cif"), encoding="utf-8")
                                res_record["BandGap_relaxed_cif_path"] = str(out_path)
                        except Exception:
                            pass

            if calc_info:
                pbar.set_postfix(calc_info)

            if online_metrics_total is not None and len(cif_list) > 0:
                if online_metrics_window is not None:
                    online_metrics_window.reset_generated_state()
                for cif_s in cif_list:
                    st = parse_structure_from_cif_str(cif_s)
                    if st is not None:
                        online_metrics_total.update_with_structure(st)
                        if online_metrics_window is not None:
                            online_metrics_window.update_with_structure(st)

            if match_metric_state is not None:
                reference = match_reference_by_id.get(sample_id)
                if reference is None:
                    raise ValueError(f"No reference structure for generated HDS id={sample_id!r}")
                rms_candidates = []
                for cif_s in cif_list:
                    generated = parse_structure_from_cif_str(cif_s)
                    if generated is None or not structure_validity(generated):
                        continue
                    try:
                        rms = matcher.get_rms_dist(generated, reference)
                    except Exception:
                        rms = None
                    if rms is not None:
                        rms_candidates.append(float(rms[0]))

                current_match = float(bool(rms_candidates))
                current_rms = min(rms_candidates) if rms_candidates else None
                match_metric_state["evaluated"] += 1
                match_metric_state["matched"] += int(current_match)
                if current_rms is not None:
                    match_metric_state["rms_sum"] += float(current_rms)
                    match_metric_state["rms_count"] += 1
                res_record["match_rate"] = current_match
                res_record["rms_dist"] = current_rms

                total_match = match_metric_state["matched"] / match_metric_state["evaluated"]
                total_rms = (
                    match_metric_state["rms_sum"] / match_metric_state["rms_count"]
                    if match_metric_state["rms_count"] else None
                )
                current_rms_text = f"{current_rms:.4f}" if current_rms is not None else "N/A"
                total_rms_text = f"{total_rms:.4f}" if total_rms is not None else "N/A"
                pbar.write(
                    "[MatchMetrics] "
                    f"sample={idx + 1}, id={sample_id}, diffusion_steps={diffusion_steps}, "
                    f"current_match_rate={current_match:.4f}, current_rms_dist={current_rms_text}, "
                    f"total_match_rate={total_match:.4f}, total_rms_dist={total_rms_text}"
                )

            results.append(res_record)

            if online_metrics_total is not None:
                sm_cur = online_metrics_window.summary() if online_metrics_window is not None else {
                    "structure_valid_rate": float("nan"),
                    "composition_valid_rate": float("nan"),
                    "cov_recall": float("nan"),
                    "cov_precision": float("nan"),
                    "wdist_rho": float("nan"),
                    "wdist_nel": float("nan"),
                }
                sm_tot = online_metrics_total.summary()
                cur_wd_rho = f"{sm_cur['wdist_rho']:.3f}" if not np.isnan(sm_cur["wdist_rho"]) else "nan"
                tot_wd_rho = f"{sm_tot['wdist_rho']:.3f}" if not np.isnan(sm_tot["wdist_rho"]) else "nan"
                cur_wd_nel = f"{sm_cur['wdist_nel']:.3f}" if not np.isnan(sm_cur["wdist_nel"]) else "nan"
                tot_wd_nel = f"{sm_tot['wdist_nel']:.3f}" if not np.isnan(sm_tot["wdist_nel"]) else "nan"
                pbar.write(
                    "[Metrics] "
                    f"sample={idx + 1}, id={sample_id}, "
                    f"cur_sv={sm_cur['structure_valid_rate']:.3f}, tot_sv={sm_tot['structure_valid_rate']:.3f}, "
                    f"cur_cv={sm_cur['composition_valid_rate']:.3f}, tot_cv={sm_tot['composition_valid_rate']:.3f}, "
                    f"cur_covR={sm_cur['cov_recall']:.3f}, tot_covR={sm_tot['cov_recall']:.3f}, "
                    f"cur_covP={sm_cur['cov_precision']:.3f}, tot_covP={sm_tot['cov_precision']:.3f}, "
                    f"cur_wd_rho={cur_wd_rho}, tot_wd_rho={tot_wd_rho}, "
                    f"cur_wd_nel={cur_wd_nel}, tot_wd_nel={tot_wd_nel}"
                )
                if online_metrics_window is not None:
                    online_metrics_window.reset_generated_state()
            else:
                pbar.write(f"[Sample {idx + 1}] id={sample_id} done | saved={len(results)} | skipped={len(skipped)}")
        except Exception as e:
            if args.skip_invalid:
                skipped.append(sample_id)
                skipped_reasons.append({"id": sample_id, "reason": str(e)})
                print(f"[Sample {idx + 1}] id={sample_id} skipped ({e})")
                continue
            raise RuntimeError(f"Failed on id={sample_id}: {e}") from e

    out_path = Path(args.out_path)
    save_results_csv(results, str(out_path))

    print(f"[Saved] {len(results)} sampled records -> {out_path}")
    if online_metrics_total is not None:
        sm = online_metrics_total.summary()
        print(
            "[OnlineMetrics] "
            f"structure_valid_rate={sm['structure_valid_rate']:.4f}, "
            f"composition_valid_rate={sm['composition_valid_rate']:.4f}, "
            f"cov_recall={sm['cov_recall']:.4f}, cov_precision={sm['cov_precision']:.4f}, "
            f"wdist_rho={sm['wdist_rho']:.4f}, wdist_nel={sm['wdist_nel']:.4f}"
        )
        if args.metrics_only:
            metrics_path = out_path.with_suffix(".metrics.json")
            metrics_summary = {
                **sm,
                "input_hds_records": selected_hds_record_count,
                "input_condition_ids": selected_condition_count,
                "sampled_condition_ids": len(results),
                "skipped_records": len(skipped),
                "hds_per_condition": args.hds_per_condition if args.group_hds_by_id else 1,
                "diffusion_samples_per_hds": args.n_samples,
                "parallel_hds_sampling": bool(args.group_hds_by_id),
                "generated_structures_saved": False,
            }
            metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
            print(f"[Saved] online metric summary -> {metrics_path}")
            state_path = out_path.with_suffix(".metrics_state.npz")
            np.savez_compressed(state_path, **online_metrics_total.export_generated_state())
            print(f"[Saved] mergeable metric state -> {state_path}")
    if match_metric_state is not None:
        match_summary = {
            "match_rate": (
                match_metric_state["matched"] / match_metric_state["evaluated"]
                if match_metric_state["evaluated"] else 0.0
            ),
            "rms_dist": (
                match_metric_state["rms_sum"] / match_metric_state["rms_count"]
                if match_metric_state["rms_count"] else None
            ),
            **match_metric_state,
            "diffusion_steps": diffusion_steps,
            "generated_structures_saved": False,
            "hds_shuffle_mode": args.hds_shuffle_mode,
            "hds_shuffle_seed": (
                args.hds_shuffle_seed if args.hds_shuffle_mode != "matched" else None
            ),
            "hds_shuffle_manifest_sha256": shuffle_manifest_sha256,
            "sampling_seed": args.sampling_seed if args.sampling_seed >= 0 else None,
            "local_unchanged_no_donor": (
                int(shuffle_manifest.get("local_unchanged_no_donor", 0))
                if shuffle_manifest is not None else 0
            ),
        }
        match_state_path = out_path.with_suffix(".match_metrics_state.json")
        match_state_path.write_text(json.dumps(match_summary, indent=2), encoding="utf-8")
        print(f"[Saved] match/RMS metric state -> {match_state_path}")
    if skipped:
        print(f"[Skipped] {len(skipped)} invalid DSS records")
        if args.save_skipped_ids:
            # Deduplicate while preserving original order.
            seen = set()
            uniq = []
            for sid in skipped:
                if sid in seen:
                    continue
                seen.add(sid)
                uniq.append(sid)

            skipped_path = Path(args.save_skipped_ids)
            skipped_path.parent.mkdir(parents=True, exist_ok=True)
            skipped_path.write_text("\n".join(uniq) + "\n", encoding="utf-8")
            print(f"[Saved] skipped IDs -> {skipped_path}")

        if args.save_skipped_reasons and skipped_reasons:
            reasons_path = Path(args.save_skipped_reasons)
            reasons_path.parent.mkdir(parents=True, exist_ok=True)
            seen_pair = set()
            rows = []
            for item in skipped_reasons:
                pair = (item.get("id", ""), item.get("reason", ""))
                if pair in seen_pair:
                    continue
                seen_pair.add(pair)
                rows.append({"id": pair[0], "reason": pair[1]})
            if reasons_path.suffix.lower() == ".csv":
                pd.DataFrame(rows).to_csv(reasons_path, index=False)
            else:
                reasons_path.write_text("\n".join([f"{r['id']}\t{r['reason']}" for r in rows]) + "\n", encoding="utf-8")
            print(f"[Saved] skipped reasons -> {reasons_path}")


if __name__ == "__main__":
    # 无属性条件：
    # python scripts/sample_from_hds.py --dss_path dataset/mp_20/gen_DSS/random_10000.pkl.gz --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml --n_samples 5 --skip_invalid --out_path results/dss_samples/random_10000_samples.pkl.gz
    # CUDA_VISIBLE_DEVICES=3 python scripts/sample_from_hds.py --dss_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new.pkl.gz --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml --skip_invalid --save_skipped_ids results/random_dss_samples/mp_20/skipped_ids_new.txt --out_path results/random_dss_samples/mp_20/sampled_from_dss_new.csv --n_samples 1 --expected_global_codes 2 --expected_local_codes 3 --save_skipped_reasons results/random_dss_samples/mp_20/skipped_reasons_new.csv
    """
# sample时实时显示：结构有效性、成分有效性、coverage recall/precision、wdist(rho)/wdist(N_el)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=2 python scripts/sample_from_hds.py \
  --dss_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new2.pkl.gz \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  # --compute_eval_metrics \
  # --enable_online_metrics \
  --reference_test_csv dataset/mp_20/test.csv \
  --reference_test_cif_col cif \
  --cov_struc_cutoff 0.4 \
  --cov_comp_cutoff 10.0 \
  # --metrics_update_interval 1 \
  --save_skipped_ids results/random_dss_samples/mp_20/skipped_ids_new_all1.txt \
  --save_skipped_reasons results/random_dss_samples/mp_20/skipped_reasons_new_all1.csv \
  --out_path results/random_dss_samples/mp_20/sampled_from_dss_new_all1.csv

    """

    # 针对带隙(bg0)属性的运行命令：
    """
    # 3070B上需要export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=3 python scripts/sample_from_hds.py \
  --dss_path dataset/mp_20/gen_random_DSS-seqs-bg/random_property_bg7_10000.pkl.gz \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion-new-bg/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --use_property_condition \
  --calc_bandgap \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  --no_compute_eval_metrics \
  --save_skipped_ids results/random_dss_samples/bg/mp_20_bg7/skipped_ids.txt \
  --save_skipped_reasons results/random_dss_samples/bg/mp_20_bg7/skipped_reasons.csv \
  --out_path results/random_dss_samples/bg/mp_20_bg7/sampled_from_dss_1.csv \
  --bandgap_relax \
  --bandgap_relax_save_dir results/random_dss_samples/bg/mp_20_bg7/bandgap_relaxed_cifs \
  --max_records 1000
    """

    # 针对形成能(fe0)属性的运行命令：
    """
CUDA_VISIBLE_DEVICES=3 python scripts/sample_from_hds.py \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion_fe/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --use_property_condition \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  --no_compute_eval_metrics \
  --dss_path dataset/mp_20/gen_random_DSS-seqs-fe/random_property_fe10_10000.pkl.gz \
  --save_skipped_ids results/random_dss_samples/fe/mp_20_fe10/skipped_ids.txt \
  --save_skipped_reasons results/random_dss_samples/fe/mp_20_fe10/skipped_reasons.csv \
  --out_path results/random_dss_samples/fe/mp_20_fe10/sampled_from_dss_1.csv \
  --calc_formation_energy \
  --max_records 1000 \
    """

    # 使用记录范围（新特性）：
    """
# 示例1：处理前1000个记录（索引 0-999）
CUDA_VISIBLE_DEVICES=2 python scripts/sample_from_hds.py \
  --dss_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new.pkl.gz \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --record_range 0,999 \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  --out_path results/random_dss_samples/mp_20/sampled_range_0_999.csv

# 示例2：处理第1000-1999个记录（索引 1000-1999）
CUDA_VISIBLE_DEVICES=2 python scripts/sample_from_hds.py \
  --dss_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new.pkl.gz \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --record_range 1000,1999 \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  --out_path results/random_dss_samples/mp_20/sampled_range_1000_1999.csv

# 示例3：并行处理多个批次（在不同GPU上）
# GPU 0: 处理 0-2499
CUDA_VISIBLE_DEVICES=0 python scripts/sample_from_hds.py \
  --dss_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new.pkl.gz \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --record_range 0,2499 \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  --out_path results/random_dss_samples/mp_20/batch_0.csv &

# GPU 1: 处理 2500-4999
CUDA_VISIBLE_DEVICES=1 python scripts/sample_from_hds.py \
  --dss_path dataset/mp_20/gen_random_DSS-seqs/random_10000_new.pkl.gz \
  --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml \
  --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth \
  --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --record_range 2500,4999 \
  --n_samples 1 \
  --expected_global_codes 2 \
  --expected_local_codes 3 \
  --skip_invalid \
  --out_path results/random_dss_samples/mp_20/batch_1.csv &

wait  # 等待所有后台任务完成

# 合并结果
python -c "
import pandas as pd
import glob
dfs = [pd.read_csv(f) for f in sorted(glob.glob('results/random_dss_samples/mp_20/batch_*.csv'))]
pd.concat(dfs, ignore_index=True).to_csv('results/random_dss_samples/mp_20/sampled_merged.csv', index=False)
print('Merged', len(pd.concat(dfs)), 'records')
"
    """

    # python scripts/sample_from_hds.py --dss_path dataset/mp_20/gen_DSS/random_10000_new.pkl.gz --cond_config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --cond_checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth --vq_checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth --vq_config config/attn_rvq/train_cryvqvae_mp_20.yaml --use_property_condition --n_samples 5 --skip_invalid --out_path results/dss_samples/random_property_samples.pkl.gz
    main()

