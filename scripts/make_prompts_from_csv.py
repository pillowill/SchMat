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
    # reduced_formula 更稳定、可读（如 SrTiO3）
    return st.composition.reduced_formula


def build_formula_tokens(formula: str) -> List[str]:
    """
    将化学式字符串转换为 FORMULA token 片段：
      <FORMULA> <ATOM_X> <COUNT_n> ... </FORMULA>
    """
    from pymatgen.core import Composition

    comp = Composition(formula)
    elems = sorted(comp.get_el_amt_dict().keys())
    counts = [int(round(float(comp[e]))) for e in elems]
    # 约简为最简整数比
    from math import gcd

    g = counts[0] if counts else 1
    for c in counts[1:]:
        g = gcd(g, c)
    g = g if g > 0 else 1
    counts = [max(1, c // g) for c in counts]

    tokens: List[str] = ["<BOS>", "<CRYSTAL>", "<FORMULA>"]
    for e, c in zip(elems, counts):
        tokens.append(f"<ATOM_{e}>")
        tokens.append(f"<COUNT_{int(c)}>")
    tokens.append("</FORMULA>")
    return tokens


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
                record: Dict[str, Any] = {
                    "id": sample_id,
                    "formula": formula,
                    "prompt_tokens": prompt_tokens,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_ok += 1
            else:
                try:
                    formula = structure_formula_from_cif(cif_str)
                    prompt_tokens = build_formula_tokens(formula)
                    record: Dict[str, Any] = {
                        "id": sample_id,
                        "formula": formula,
                        "prompt_tokens": prompt_tokens,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_ok += 1
                except Exception as e:
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

