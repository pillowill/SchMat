import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def structure_formula_from_cif(cif_str: str) -> str:
    from pymatgen.io.cif import CifParser
    from io import StringIO

    parser = CifParser(StringIO(cif_str))
    # 新版 pymatgen 推荐 parse_structures（避免 get_structures 的弃用告警）
    try:
        st = parser.parse_structures(primitive=True)[0]
    except Exception:
        st = parser.get_structures()[0]
    # CSP skeleton must describe every site in the represented cell.  Using a
    # reduced formula here would recreate the formula/site mismatch fixed by E2.
    return st.composition.formula


def build_formula_tokens(formula: str) -> List[str]:
    """
    将化学式字符串转换为 FORMULA token 片段：
      <FORMULA> <ATOM_X> <COUNT_n> ... </FORMULA>
    """
    from pymatgen.core import Composition

    comp = Composition(formula).get_el_amt_dict()
    from pymatgen.core.periodic_table import Element

    elems = sorted(comp, key=lambda symbol: int(Element(symbol).Z))
    counts = []
    for elem in elems:
        amount = float(comp[elem])
        count = int(round(amount))
        if count <= 0 or abs(amount - count) > 1e-6:
            raise ValueError(f"Full-cell composition for {elem} is not a positive integer: {amount}")
        counts.append(count)

    tokens: List[str] = ["<BOS>", "<CRYSTAL>", "<FORMULA>"]
    for e, c in zip(elems, counts):
        tokens.append(f"<ATOM_{e}>")
        tokens.append(f"<COUNT_{int(c)}>")
    tokens.append("</FORMULA>")
    return tokens


def add_property_prompt_tokens(
    prompt_tokens: List[str],
    value: Any,
    property_name: str,
    property_scheme: str,
) -> tuple[List[str], int | None]:
    """Insert the independent property condition immediately after CRYSTAL."""
    if property_scheme == "none":
        return prompt_tokens, None
    if property_scheme != "formation_energy_10":
        raise ValueError(f"Unsupported property scheme: {property_scheme}")
    if value is None or pd.isna(value):
        raise ValueError("Missing formation-energy value for a conditional CSP prompt")
    from scripts.tokenizer import bin_formation_energy_10, sanitize_token_name

    property_bin = bin_formation_energy_10(float(value))
    property_tokens = [
        "<PROPERTY>",
        f"<PROPERTY_NAME_{sanitize_token_name(property_name)}>",
        f"<PROPERTY_BIN_{property_bin}>",
        "</PROPERTY>",
    ]
    updated = list(prompt_tokens)
    updated[2:2] = property_tokens
    return updated, property_bin


def composition_counts_from_prompt(prompt_tokens: List[str]) -> List[Dict[str, Any]]:
    """Return structured full-cell counts from the already validated FORMULA block."""
    return [
        {"symbol": token[6:-1], "count": int(prompt_tokens[index + 1][7:-1])}
        for index, token in enumerate(prompt_tokens)
        if token.startswith("<ATOM_") and index + 1 < len(prompt_tokens)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Create prompt file (jsonl) from CSV containing CIF strings.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file path (must contain a 'cif' column).")
    parser.add_argument(
        "--id_col",
        type=str,
        default="material_id",
        help="ID column name (fallback to 'id' then row index).",
    )
    parser.add_argument(
        "--cif_col",
        type=str,
        default="cif",
        help="CIF string column name (default: cif).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/prompts/prompts.jsonl",
        help="Output prompt file path (.jsonl).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="If >0, only process first N rows (debug).",
    )
    parser.add_argument(
        "--debug_rows",
        type=int,
        default=0,
        help="If >0, print basic info for first N rows (type/len/head of cif field).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, do NOT swallow exceptions. Stop at first error with full traceback (for debugging).",
    )
    parser.add_argument(
        "--print_errors",
        type=int,
        default=0,
        help="If >0 and not --strict, print traceback for first N errors.",
    )
    parser.add_argument(
        "--property_scheme",
        choices=["none", "formation_energy_10"],
        default="none",
        help="Optional independent conditional branch for strict one-based formation-energy bins 1..10.",
    )
    parser.add_argument("--property_col", type=str, default="formation_energy_per_atom")
    parser.add_argument("--property_name", type=str, default="FORMATION_ENERGY_PER_ATOM")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    assert csv_path.is_file(), f"CSV not found: {csv_path}"

    # NOTE:
    # 不同平台/版本的 pandas 对超长字段和 NA 的推断可能不一致。
    # 这里强制 cif 列按字符串读取，并关闭默认 NA 解析，避免整列被读成 NaN 导致全部 skip。
    df = pd.read_csv(
        csv_path,
        dtype={args.cif_col: "string"},
        keep_default_na=False,
        na_filter=False,
    )
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    assert args.cif_col in df.columns, f"Missing cif column '{args.cif_col}' in CSV."

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_ok = 0
    num_skip = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            cif_val = row.get(args.cif_col, "")
            # pandas StringDtype 可能给出 <NA> 或 numpy 标量；统一转 str
            cif_str = "" if cif_val is None else str(cif_val)
            if cif_str.lower() == "nan":
                cif_str = ""

            if args.debug_rows and args.debug_rows > 0 and i < args.debug_rows:
                head = cif_str[:80].replace("\n", "\\n")
                print(f"[Debug] row={i} cif_len={len(cif_str)} cif_head={head}")

            if len(cif_str.strip()) == 0:
                num_skip += 1
                continue

            sample_id = row.get(args.id_col, None)
            if sample_id is None or (isinstance(sample_id, float) and pd.isna(sample_id)):
                sample_id = row.get("id", None)
            if sample_id is None or (isinstance(sample_id, float) and pd.isna(sample_id)):
                sample_id = f"row_{i}"
            sample_id = str(sample_id)

            if args.strict:
                # 直接抛异常，便于定位问题
                formula = structure_formula_from_cif(cif_str)
                prompt_tokens = build_formula_tokens(formula)
                prompt_tokens, property_bin = add_property_prompt_tokens(
                    prompt_tokens,
                    row.get(args.property_col, None),
                    args.property_name,
                    args.property_scheme,
                )
                record: Dict[str, Any] = {
                    "id": sample_id,
                    "formula": formula,
                    "prompt_tokens": prompt_tokens,
                    "composition_counts": composition_counts_from_prompt(prompt_tokens),
                    "property_bin": property_bin,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_ok += 1
            else:
                try:
                    formula = structure_formula_from_cif(cif_str)
                    prompt_tokens = build_formula_tokens(formula)
                    prompt_tokens, property_bin = add_property_prompt_tokens(
                        prompt_tokens,
                        row.get(args.property_col, None),
                        args.property_name,
                        args.property_scheme,
                    )
                    record: Dict[str, Any] = {
                        "id": sample_id,
                        "formula": formula,
                        "prompt_tokens": prompt_tokens,
                        "composition_counts": composition_counts_from_prompt(prompt_tokens),
                        "property_bin": property_bin,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_ok += 1
                except Exception as e:
                    if args.property_scheme == "formation_energy_10":
                        raise
                    num_skip += 1
                    if args.print_errors and args.print_errors > 0 and num_skip <= args.print_errors:
                        import traceback
                        print(f"[Error] row={i} id={sample_id} err={repr(e)}")
                        traceback.print_exc()
                    continue

    # Windows PowerShell 默认编码可能是 GBK，避免打印 Unicode 符号导致崩溃
    print(f"[Saved] prompts -> {out_path}")
    print(f"[Stats] ok={num_ok}, skipped={num_skip}")


if __name__ == "__main__":
    # python scripts/make_prompts_from_csv.py --csv_path "dataset/mpts_52/test.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/mpts_52/test_prompts.jsonl"
    # python scripts/make_prompts_from_csv.py --csv_path "dataset/mp_20/test.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/mp_20/test_prompts.jsonl"

    # python scripts/make_prompts_from_csv.py --csv_path "dataset/carbon_24/train.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/carbon_24/train_prompts.jsonl"
    # python scripts/make_prompts_from_csv.py --csv_path "dataset/carbon_24/val.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/carbon_24/val_prompts.jsonl"
    # python scripts/make_prompts_from_csv.py --csv_path "dataset/carbon_24/test.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/carbon_24/test_prompts.jsonl"

    # python scripts/make_prompts_from_csv.py --csv_path "dataset/perov_5/train.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/perov_5/train_prompts.jsonl"
    # python scripts/make_prompts_from_csv.py --csv_path "dataset/perov_5/val.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/perov_5/val_prompts.jsonl"
    # python scripts/make_prompts_from_csv.py --csv_path "dataset/perov_5/test.csv" --id_col "material_id" --cif_col "cif" --out_path "dataset/perov_5/test_prompts.jsonl"
    main()

