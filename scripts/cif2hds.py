# -*- coding: utf-8 -*-
"""
@Time ： 2025/12/23 10:36
@Auth ： He Yu
@File ：cif2hds.py
@IDE ：PyCharm

目标：
1. 读取 CIF/CSV/LMDB 数据。
2. 加载训练好的 CrystalVQVAE/Attn-RVQ，将结构编码为全局与局部离散 code。
3. 按与 tokenizer 对齐的简化 DSS-seq 模板输出 token。

默认模板不包含 <SPACE_GROUP_*>, <CRYSTAL_SYSTEM_*> 和 <COORD_*>，避免产生 tokenizer 词表之外的 token。
如需兼容旧实验，可以通过 --include_space_group、--include_crystal_system 或 --include_coord 显式打开。

简化模板示例：
<BOS>
<CRYSTAL>
  <FORMULA> <ATOM_X> <COUNT_n> ... </FORMULA>
  <GLOBAL>
    <GLOBAL_CODE_12>
    <NUM_SITES_5>
  </GLOBAL>
  <SITES>
    <SITE_1> <ATOM_Sr> <LOCAL_CODE_3> </SITE_1>
    <SITE_2> <ATOM_Ti> <LOCAL_CODE_17> </SITE_2>
  </SITES>
</CRYSTAL>
<EOS>
"""
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
from io import StringIO

import torch
from torch_geometric.data import Batch

import sys
import os
from pathlib import Path
import argparse
import yaml
import pickle
import gzip
import random
import numpy as np
from math import floor

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 将工作目录切换到项目根目录，简化相对路径处理
os.chdir(project_root)

# 使用新的数据处理逻辑
from data.data import CIFData, collate_pool
from utils.data_utils import atoms_to_pyg_data
from model.attn_rvq import CrystalVQVAE
from scripts.tokenizer import CrystalVocabConfig, HDSCompiler, bin_formation_energy_10


def sanitize_token_name(name: str) -> str:
    if name is None:
        return "UNKNOWN"
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


def parse_float_or_none(value):
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def bin_property_value(value: float, bin_size: float = 0.5, origin: float = 0.0,
                       max_bins: int | None = None, strict_range: bool = False) -> int | None:
    if value is None:
        return None
    if bin_size <= 0:
        raise ValueError(f"property_bin_size must be > 0, got {bin_size}")
    idx = int(floor((value - origin) / bin_size))
    if idx < 0 or (max_bins is not None and idx >= max_bins):
        if strict_range:
            upper = origin + bin_size * max_bins if max_bins is not None else None
            raise ValueError(
                f"Property value {value} is outside [{origin}, {upper})"
            )
        idx = max(idx, 0)
        if max_bins is not None:
            idx = min(idx, max_bins - 1)
    return idx


def build_property_tokens(
        property_value,
        property_name: str,
        bin_size: float = 0.5,
        bin_origin: float = 0.0,
        max_bins: int = 128,
        property_scheme: str = "generic",
):
    value = parse_float_or_none(property_value)
    if value is None:
        if property_scheme == "formation_energy_10":
            raise ValueError(
                "formation_energy_10 requires a finite formation-energy value; "
                f"got {property_value!r}"
            )
        return []
    if property_scheme == "formation_energy_10":
        if max_bins <= 10:
            raise ValueError("formation_energy_10 uses token ids 1..10 and requires property_max_bins >= 11")
        bin_idx = bin_formation_energy_10(value)
    elif property_scheme == "generic":
        bin_idx = bin_property_value(value, bin_size=bin_size, origin=bin_origin, max_bins=max_bins)
    else:
        raise ValueError(f"Unsupported property scheme: {property_scheme}")
    if bin_idx is None:
        return []
    name_token = sanitize_token_name(property_name)
    return [
        "<PROPERTY>",
        f"<PROPERTY_NAME_{name_token}>",
        f"<PROPERTY_BIN_{bin_idx}>",
        "</PROPERTY>",
    ]


def load_structure(cif_path: str) -> Structure:
    """从 CIF 文件加载 pymatgen Structure."""
    return Structure.from_file(cif_path)


def structure_to_pyg_data(structure: Structure):
    """
    将 pymatgen Structure 转换为 PyG Data 对象（与 data.py 中的处理方式一致）。

    返回:
        pyg_data: PyG Data 对象
    """
    # 进行 niggli reduction 和规范化（与 CIFData.__getitem__ 一致）
    reduced_st = structure.get_reduced_structure()
    canonical_st = Structure(
        lattice=Lattice.from_parameters(*reduced_st.lattice.parameters),
        species=reduced_st.species,
        coords=reduced_st.frac_coords,
        coords_are_cartesian=False,
    )
    atoms = canonical_st.to_ase_atoms()
    pyg_data = atoms_to_pyg_data(atoms)
    return pyg_data


@torch.no_grad()
def encode_structure_with_vq(
        model: CrystalVQVAE,
        structure: Structure,
        device: str = "cuda",
):
    """
    使用训练好的 CrystalVQVAE 模型，将 Structure 编码为离散 code + 语义属性。

    返回：
      global_code_ids: List[int] - 全局 codebook 索引
      global_attributes: dict - 全局属性（空间群、晶系等）
      local_code_ids: List[List[int]] - 每个原子的局部 codebook 索引
      local_attributes: List[dict] - 每个原子的属性（配位数等）
    """
    model.eval()

    # === 1. Structure → PyG Data ===
    pyg_data = structure_to_pyg_data(structure)
    batch = Batch.from_data_list([pyg_data]).to(device)

    # === 2. Encode + Quantize ===
    g, h = model.encode(
        atom_types=batch.atom_types,
        frac_coords=batch.frac_coords,
        lattice=batch.lattices,
        num_atoms=batch.natoms,
        node2batch=batch.batch,
    )

    z_g, z_l, indices_g, indices_l, _ = model.quantize(g, h)

    # === 3. GLOBAL code ===
    # indices_g: (batch_size, num_quantizers) -> (1, num_quantizers) for single structure
    # 转换为列表格式，每个元素是一个量化器的索引
    if indices_g.dim() == 1:
        # 旧格式：单个索引，扩展为多个量化器
        qg = getattr(model, "num_quantizers_global", getattr(model, "num_quantizers", 1))
        qg = int(qg) if qg is not None else 1
        global_code_ids = [int(indices_g[0].item())] * qg
    else:
        # ResidualVQ 格式：多个量化器索引
        global_code_ids = [int(idx.item()) for idx in indices_g[0]]

    # 解码全局语义
    global_pred = model.global_decoder(z_g)
    space_group_idx = torch.argmax(global_pred["space_group_logits"], dim=-1).item()

    # 使用 pymatgen 获取真实的晶系信息
    analyzer = SpacegroupAnalyzer(structure)
    crystal_system_name = analyzer.get_crystal_system().upper()
    num_sites = len(structure)

    global_attrs = {
        "num_sites": int(num_sites),
        "space_group": int(space_group_idx),
        "crystal_system": crystal_system_name,
    }

    # === 4. LOCAL code ===
    # indices_l: (total_atoms, num_quantizers) -> List[List[int]]
    if indices_l.dim() == 1:
        # 旧格式：单个索引，扩展为多个量化器
        ql = getattr(model, "num_quantizers_local", getattr(model, "num_quantizers", 1))
        ql = int(ql) if ql is not None else 1
        local_code_ids = [[int(idx.item())] * ql for idx in indices_l]
    else:
        # ResidualVQ 格式：多个量化器索引
        local_code_ids = [[int(idx.item()) for idx in atom_indices] for atom_indices in indices_l]

    # 解码局部语义
    local_pred = model.local_decoder(z_l)
    coord_nums = torch.argmax(local_pred["cn_logits"], dim=-1)  # (total_atoms,)

    local_attrs = []
    for atom_idx in range(len(local_code_ids)):
        coord_num = int(coord_nums[atom_idx].item())
        local_attrs.append({"coord_num": coord_num})

    return global_code_ids, global_attrs, local_code_ids, local_attrs


def _build_formula_tokens(structure: Structure):
    """
    从 pymatgen Structure 提取化学式，并构造公式相关 token 片段。

    使用当前表示晶胞的完整整数计数，不再约简化学式：
      <FORMULA> <ATOM_X> <COUNT_n> ... </FORMULA>
    """
    tokens = []
    tokens.append("<FORMULA>")

    comp = structure.composition.get_el_amt_dict()
    from pymatgen.core.periodic_table import Element

    for elem in sorted(comp, key=lambda symbol: int(Element(symbol).Z)):
        amount = float(comp[elem])
        count = int(round(amount))
        if count <= 0 or not np.isclose(amount, count):
            raise ValueError(f"Non-integer full-cell composition for {elem}: {amount}")
        tokens.extend([f"<ATOM_{elem}>", f"<COUNT_{count}>"])

    tokens.append("</FORMULA>")
    return tokens


def build_crystal_tokens(
        structure: Structure,
        global_code_ids: list,
        global_attrs: dict,
        local_code_ids: list,
        local_attrs: list,
        property_value=None,
        property_name: str | None = None,
        property_bin_size: float = 0.5,
        property_bin_origin: float = 0.0,
        property_max_bins: int = 128,
        property_scheme: str = "generic",
        include_space_group: bool = False,
        include_crystal_system: bool = False,
        include_coord: bool = False,
):
    """
    构建 canonical、Q-tagged、stage-marked HDS token 序列。

    默认不输出 <SPACE_GROUP_*>, <CRYSTAL_SYSTEM_*> 和 <COORD_*>。
    如果确实需要兼容旧模板，可通过 include_space_group/include_crystal_system/include_coord 打开。
    """
    if int(global_attrs.get("num_sites", len(structure))) != len(structure):
        raise ValueError(
            f"Attn-RVQ num_sites={global_attrs.get('num_sites')} differs from structure sites={len(structure)}"
        )
    if len(local_code_ids) != len(structure):
        raise ValueError(
            f"Attn-RVQ local rows={len(local_code_ids)} differs from structure sites={len(structure)}"
        )

    property_tokens = []
    if property_name is not None:
        property_tokens = build_property_tokens(
            property_value,
            property_name=property_name,
            bin_size=property_bin_size,
            bin_origin=property_bin_origin,
            max_bins=property_max_bins,
            property_scheme=property_scheme,
        )

    qg = len(global_code_ids)
    local_lengths = {len(row) for row in local_code_ids}
    if len(local_lengths) != 1:
        raise ValueError(f"Ragged local RVQ stages: {sorted(local_lengths)}")
    ql = next(iter(local_lengths))
    if qg <= 0 or ql <= 0:
        raise ValueError(f"Attn-RVQ must provide positive stage counts, got Qg={qg}, Ql={ql}")
    global_size = max(int(code) for code in global_code_ids) + 1
    local_size = max(int(code) for row in local_code_ids for code in row) + 1
    compiler = HDSCompiler(CrystalVocabConfig(
        num_quantizers_global=qg,
        codebooksize_global=max(global_size, 1),
        num_quantizers_local=ql,
        codebooksize_local=max(local_size, 1),
        max_sites=max(128, len(structure)),
        max_formula_count=max(100, len(structure)),
        include_hds_stage_tokens=True,
    ))
    compiled = compiler.compile_from_codes(
        global_codes=global_code_ids,
        atom_symbols=[site.specie.symbol for site in structure.sites],
        local_codes=local_code_ids,
        property_tokens=property_tokens,
    )
    tokens = list(compiled.tokens)

    # Legacy semantic attributes remain opt-in and are outside strict E2 HDS.
    num_sites_index = tokens.index(f"<NUM_SITES_{len(structure)}>")
    optional_global_tokens = []
    if include_space_group:
        optional_global_tokens.append(f"<SPACE_GROUP_{global_attrs.get('space_group', 'UNKNOWN')}>")
    if include_crystal_system:
        optional_global_tokens.append(f"<CRYSTAL_SYSTEM_{global_attrs.get('crystal_system', 'UNKNOWN')}>")
    if optional_global_tokens:
        tokens[num_sites_index + 1:num_sites_index + 1] = optional_global_tokens
    if include_coord:
        if len(local_attrs) != len(structure):
            raise ValueError("Coord tokens require one local_attrs row per structure site")
        canonical_attrs = [local_attrs[index] for index in compiled.canonical_atom_permutation]
        for site_index, attrs in enumerate(canonical_attrs, start=1):
            closing = tokens.index(f"</SITE_{site_index}>")
            tokens.insert(closing, f"<COORD_{attrs['coord_num']}>")
    return tokens


# ======================
# 最小示例：不加载模型版本
# ======================

def encode_structure_minimal(structure: Structure):
    """
    最小工作示例：
    - 不依赖训练好的 VQ 模型
    - 使用 pymatgen 提取几何语义信息
    - 用占位符 GLOBAL_CODE / LOCAL_CODE 构造完整的 token 结构

    返回:
      tokens: List[str]
    """
    # 全局属性：使用 pymatgen 分析
    analyzer = SpacegroupAnalyzer(structure)
    space_group = analyzer.get_space_group_number()
    crystal_system = analyzer.get_crystal_system().upper()
    num_sites = len(structure)

    # 构造与 encode_structure_with_vq 一致的 global_attrs 结构
    global_code_ids = [0]  # 占位符：真实场景由 VQ code 决定
    global_attrs = {
        "num_sites": int(num_sites),
        "space_group": int(space_group),
        "crystal_system": crystal_system,
    }

    # 局部属性：计算每个原子的配位数
    local_code_ids = []  # 占位符：这里统一用 0
    local_attrs = []
    for atom_idx in range(len(structure)):
        site = structure[atom_idx]
        neighbors = structure.get_neighbors(site, r=3.0)
        coord_num = len(neighbors)
        local_code_ids.append([0])
        local_attrs.append({"coord_num": coord_num})

    tokens = build_crystal_tokens(
        structure,
        global_code_ids=global_code_ids,
        global_attrs=global_attrs,
        local_code_ids=local_code_ids,
        local_attrs=local_attrs,
    )
    return tokens


def insert_property_block(tokens: list[str], property_tokens: list[str]) -> list[str]:
    """Insert optional property tokens right after <CRYSTAL> and before <FORMULA>."""
    if not property_tokens:
        return tokens
    try:
        crystal_idx = tokens.index("<CRYSTAL>")
    except ValueError:
        return [tokens[0], *property_tokens, *tokens[1:]] if tokens else property_tokens

    insert_at = crystal_idx + 1
    # If a FORMULA token exists, insert before it; otherwise keep after <CRYSTAL>.
    try:
        formula_idx = tokens.index("<FORMULA>")
        insert_at = formula_idx
    except ValueError:
        pass

    return tokens[:insert_at] + property_tokens + tokens[insert_at:]


def extract_property_from_record(record: dict, property_col: str | None, property_name: str | None = None):
    if not property_col:
        return None
    return record.get(property_col, None)


# =============================================================
#  批量处理：CSV 文件中的 CIF → token.pkl.gz
#  加载训练好的 VQ 模型，配置需与 yaml 一致
# =============================================================

def load_yaml_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)


def save_pickle(obj, path: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "wb"
    with opener(path, mode) as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load a token shard written by :func:`save_pickle`."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def parse_record_range(record_range: str, total_records: int) -> tuple[int, int]:
    """Parse an inclusive ``start,end`` (or ``start:end``) shard range."""
    value = str(record_range).strip()
    if not value:
        raise ValueError("empty record_range")
    if (value.startswith("[") and value.endswith("]")) or (
            value.startswith("(") and value.endswith(")")
    ):
        value = value[1:-1].strip()
    parts = [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("record_range must be 'start,end', '[start,end]', or 'start:end'")

    try:
        start_idx, end_idx = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"record_range indices must be integers: {record_range!r}") from exc

    if total_records <= 0:
        raise ValueError("cannot shard an empty split")
    if start_idx < 0 or end_idx < start_idx or start_idx >= total_records:
        raise ValueError(
            f"invalid record range [{start_idx}, {end_idx}] for {total_records} records"
        )
    return start_idx, min(end_idx, total_records - 1)


def merge_token_shards(shard_paths: list[str], output_path: str) -> int:
    """Merge token shards in source-row order and reject duplicate records."""
    if not shard_paths:
        raise ValueError("at least one --merge_shards path is required")

    merged = []
    source_indices = set()
    material_ids = set()
    for shard_path in shard_paths:
        records = load_pickle(shard_path)
        if not isinstance(records, list):
            raise ValueError(f"token shard is not a list: {shard_path}")
        for record in records:
            source_index = record.get("source_index")
            material_id = str(record.get("id", ""))
            if source_index is None:
                raise ValueError(f"missing source_index in shard: {shard_path}")
            if source_index in source_indices:
                raise ValueError(f"duplicate source_index {source_index} across token shards")
            if material_id in material_ids:
                raise ValueError(f"duplicate material id {material_id!r} across token shards")
            source_indices.add(source_index)
            material_ids.add(material_id)
            merged.append(record)

    merged.sort(key=lambda record: record["source_index"])
    save_pickle(merged, output_path)
    print(f"[Merged] {len(merged)} records from {len(shard_paths)} shards -> {output_path}")
    return len(merged)


def build_model_from_config(config: dict, checkpoint: dict, device: str):
    """
    从配置和检查点构建 CrystalVQVAE 模型。
    """
    # 提取 state_dict
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Could not find model_state_dict or state_dict in checkpoint")

    # 从配置获取参数
    def get_param(key, default):
        return config.get(key, default)

    # 统一使用 0.0.5 版本的 CrystalVQVAE（支持 residual / vector 两种 VQ）
    model_config = {
        "hidden_dim": get_param("hidden_dim", 256),
        "num_layers": get_param("num_layers", 4),
        "codebook_size": get_param("codebook_size", 512),
        "num_space_groups": get_param("num_space_groups", 230),
        "max_cn": get_param("max_cn", 20),
        "num_elements": get_param("num_elements", 118),
        "commitment_weight": get_param("commitment_weight", 0.0),
        "loss_weights": get_param("loss_weights", None),
        # ResidualVQ 的量化器数量：
        # - num_quantizers_global / num_quantizers_local 如果提供，则分别控制全局/局部量化器数
        # - 否则回退到 num_quantizers（向后兼容）
        "num_quantizers": get_param("num_quantizers", 4),
        "num_quantizers_global": get_param("num_quantizers_global", None),
        "num_quantizers_local": get_param("num_quantizers_local", None),
        "vq_type": get_param("vq_type", "residual"),
        "aggregator_dropout": get_param("aggregator_dropout", 0.1),
    }

    print(
        f"Creating CrystalVQVAE from config: hidden_dim={model_config['hidden_dim']}, "
        f"num_layers={model_config['num_layers']}, codebook_size={model_config['codebook_size']}, "
        f"num_quantizers={model_config['num_quantizers']}, "
        f"num_quantizers_global={model_config['num_quantizers_global']}, "
        f"num_quantizers_local={model_config['num_quantizers_local']}, "
        f"vq_type={model_config['vq_type']}"
    )

    model = CrystalVQVAE(**model_config).to(device)

    # 加载 state dict：统一用 strict=False 兼容不同版本 checkpoint
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        # 如果 strict=True 失败，退回 strict=False，但打印详细提示
        print(f"[Warning] strict load_state_dict failed: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys when loading state_dict: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            print(f"  Missing: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            print(f"  Unexpected: {unexpected_keys}")

    return model


# OC20 LMDB helpers removed. Use CSV/ROOT_DIR processing functions instead.


def process_csv_to_tokens(
        root_dir: str,
        output_dir: str,
        checkpoint_path: str = None,
        config_path: str = None,
        train_csv: str = 'train.csv',
        val_csv: str = 'val.csv',
        test_csv: str = 'test.csv',
        max_length: int | None = None,
        device: str = "cuda",
        use_model: bool = True,
        process_test: bool = True,
        splits: list[str] | None = None,
        record_range: str = "",
        include_property: bool = False,
        property_col: str | None = None,
        property_name: str | None = None,
        property_bin_size: float = 0.5,
        property_bin_origin: float = 0.0,
        property_max_bins: int = 128,
        property_scheme: str = "generic",
        include_space_group: bool = False,
        include_crystal_system: bool = False,
        include_coord: bool = False,
):
    """
    从 CSV 文件批量处理 CIF 转换为 token（与 train_cond_diff.py 的数据格式一致）。

    参数:
        root_dir: 数据根目录，包含 train.csv, val.csv, test.csv
        output_dir: 输出目录，将保存 train_tokens.pkl.gz, val_tokens.pkl.gz, test_tokens.pkl.gz
        checkpoint_path: 模型检查点路径（使用模型时必需）
        config_path: 配置文件路径（使用模型时必需）
        train_csv: 训练集 CSV 文件名
        val_csv: 验证集 CSV 文件名
        test_csv: 测试集 CSV 文件名
        max_length: 最大 token 长度（None 表示不限制）
        device: 设备
        use_model: 是否使用训练好的模型
        process_test: 是否处理测试集
    """
    model = None
    if use_model:
        if not checkpoint_path or not config_path:
            raise ValueError("--checkpoint and --config are required when --use_model is set")
        # 加载配置与 checkpoint
        config = load_yaml_config(config_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model_from_config(config, checkpoint, device)
        model.eval()
        print("✓ Loaded CrystalVQVAE model weights successfully")
    else:
        print("Using minimal encoding (no model)")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    def encode_dataset(dataset, split_name="train"):
        """处理一个数据集（train/val/test）"""
        results = []
        records = dataset.id_prop_data
        source_start = 0
        if record_range:
            source_start, source_end = parse_record_range(record_range, len(records))
            records = records[source_start:source_end + 1]
            print(
                f"[{split_name}] processing source rows [{source_start}, {source_end}] "
                f"({len(records)} records)"
            )
        # 直接从 dataset.id_prop_data 读取，避免重复调用 dataset[idx]
        for source_index, cry in enumerate(
                tqdm(records, desc=f"Tokenizing {split_name} CIFs"), start=source_start
        ):
            try:
                cif_str = cry.get('cif', '')
                if not cif_str:
                    continue

                # 获取 cif_id
                cif_id = cry.get('material_id', cry.get('id', f'crystal_{source_index}'))

                # 解析结构
                structure = CifParser(StringIO(cif_str)).get_structures()[0]

                # 可选：从 CSV 表头读取物性并离散化
                property_value = None
                if include_property and property_col:
                    property_value = cry.get(property_col, None)
                    if property_value is None:
                        # 兼容常见候选列名
                        for alt in [property_col.replace(" ", "_"), property_col.replace("_", " "),
                                    property_col.lower(), property_col.upper()]:
                            if alt in cry:
                                property_value = cry.get(alt, None)
                                break

                # 编码
                if use_model and model is not None:
                    global_code_ids, global_attrs, local_code_ids, local_attrs = encode_structure_with_vq(
                        model,
                        structure,
                        device=device,
                    )
                    tokens = build_crystal_tokens(
                        structure,
                        global_code_ids=global_code_ids,
                        global_attrs=global_attrs,
                        local_code_ids=local_code_ids,
                        local_attrs=local_attrs,
                        property_value=property_value,
                        property_name=property_name or property_col,
                        property_bin_size=property_bin_size,
                        property_bin_origin=property_bin_origin,
                        property_max_bins=property_max_bins,
                        property_scheme=property_scheme,
                        include_space_group=include_space_group,
                        include_crystal_system=include_crystal_system,
                        include_coord=include_coord,
                    )
                else:
                    # 使用最小示例
                    tokens = encode_structure_minimal(structure)
                    if include_property and property_col:
                        property_tokens = build_property_tokens(
                            property_value,
                            property_name=property_name or property_col,
                            bin_size=property_bin_size,
                            bin_origin=property_bin_origin,
                            max_bins=property_max_bins,
                            property_scheme=property_scheme,
                        )
                        tokens = insert_property_block(tokens, property_tokens)

                # 处理长度
                if max_length is not None:
                    if len(tokens) > max_length:
                        raise ValueError(
                            f"Canonical HDS length {len(tokens)} exceeds max_length={max_length}; "
                            "truncation is forbidden"
                        )
                    else:
                        tokens = tokens + ["<PAD>"] * (max_length - len(tokens))

                results.append({"id": cif_id, "tokens": tokens, "source_index": source_index})
            except Exception as e:
                print(f"[Skip {split_name}] idx={source_index}, encode error: {e}")
                import traceback
                traceback.print_exc()
                if property_scheme == "formation_energy_10":
                    raise
                continue
        return results

    # The legacy no-argument path below is kept intact. Any explicit split or
    # range request uses this branch so independent GPU workers never rewrite
    # one another's files.
    if splits is not None or record_range:
        split_to_csv = {"train": train_csv, "val": val_csv, "test": test_csv}
        selected_splits = list(splits) if splits is not None else ["train", "val", "test"]
        if not process_test and "test" in selected_splits:
            selected_splits.remove("test")
        invalid_splits = set(selected_splits).difference(split_to_csv)
        if invalid_splits:
            raise ValueError(f"unsupported split names: {sorted(invalid_splits)}")

        for split_name in selected_splits:
            csv_file = split_to_csv[split_name]
            print(f"Processing {split_name} split: {csv_file}")
            dataset = CIFData(
                root_dir=root_dir,
                proper="formation_energy",
                cond_guide=False,
                csv_file=csv_file,
                shuffle=False,
            )
            results = encode_dataset(dataset, split_name=split_name)
            split_output = output_path / f"{split_name}_tokens.pkl.gz"
            save_pickle(results, str(split_output))
            print(f"Saved {split_name} tokens to {split_output} ({len(results)} samples)")
        return

    # 处理训练集
    print(f"\n处理训练集: {train_csv}")
    train_dataset = CIFData(
        root_dir=root_dir,
        proper='formation_energy',  # 占位符，实际不使用
        cond_guide=False,
        csv_file=train_csv,
        shuffle=False,  # 不需要打乱，因为只是编码
    )
    train_results = encode_dataset(train_dataset, split_name="train")
    train_output = output_path / "train_tokens.pkl.gz"
    save_pickle(train_results, str(train_output))
    print(f"✓ Saved train tokens to {train_output} ({len(train_results)} samples)")

    # 处理验证集
    print(f"\n处理验证集: {val_csv}")
    val_dataset = CIFData(
        root_dir=root_dir,
        proper='formation_energy',
        cond_guide=False,
        csv_file=val_csv,
        shuffle=False,
    )
    val_results = encode_dataset(val_dataset, split_name="val")
    val_output = output_path / "val_tokens.pkl.gz"
    save_pickle(val_results, str(val_output))
    print(f"✓ Saved val tokens to {val_output} ({len(val_results)} samples)")

    # 处理测试集（可选）
    if process_test:
        print(f"\n处理测试集: {test_csv}")
        test_dataset = CIFData(
            root_dir=root_dir,
            proper='formation_energy',
            cond_guide=False,
            csv_file=test_csv,
            shuffle=False,
        )
        test_results = encode_dataset(test_dataset, split_name="test")
        test_output = output_path / "test_tokens.pkl.gz"
        save_pickle(test_results, str(test_output))
        print(f"✓ Saved test tokens to {test_output} ({len(test_results)} samples)")

    print(f"\n{'=' * 60}")
    print("所有数据集处理完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIF → VQ token (single or batch)")
    parser.add_argument(
        "--cif",
        type=str,
        required=False,
        default="../model/mp-18767-LiMnO2.cif",
        help="Path to CIF file (single example)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory containing train.csv, val.csv, test.csv (CSV mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save tokenized results (CSV mode)",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train.csv",
        help="Training set CSV filename (default: train.csv)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="val.csv",
        help="Validation set CSV filename (default: val.csv)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="Test set CSV filename (default: test.csv)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml file (required if using model)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained CrystalVQVAE checkpoint .pth (required if using model)",
    )
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Use trained model for encoding (default: False, use minimal encoding)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Pad/truncate token sequences to this length",
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        help="Skip processing test set (CSV mode)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Only tokenize these splits. Defaults to legacy train/val/test processing.",
    )
    parser.add_argument(
        "--record_range",
        type=str,
        default="",
        help=(
            "Inclusive source-row range for a shard, e.g. '0,999' or '0:999'. "
            "Applied to every selected split."
        ),
    )
    parser.add_argument(
        "--merge_shards",
        nargs="+",
        default=None,
        help="Ordered token shard files to merge instead of encoding CIFs.",
    )
    parser.add_argument(
        "--merge_output",
        type=str,
        default="",
        help="Output .pkl/.pkl.gz path used with --merge_shards.",
    )
    # OC20 LMDB support removed; OC20-specific CLI args deleted
    parser.add_argument(
        "--include_space_group",
        action="store_true",
        help="Include <SPACE_GROUP_*> tokens. Disabled by default to match the simplified tokenizer.",
    )
    parser.add_argument(
        "--include_crystal_system",
        action="store_true",
        help="Include <CRYSTAL_SYSTEM_*> tokens. Disabled by default to match the simplified tokenizer.",
    )
    parser.add_argument(
        "--include_coord",
        action="store_true",
        help="Include <COORD_*> tokens. Disabled by default to match the simplified tokenizer.",
    )
    parser.add_argument(
        "--oc20_property",
        type=str,
        default="energy",
        help="OC20 property field name (default: 'energy')",
    )
    parser.add_argument(
        "--include_property",
        action="store_true",
        help="Include a physical property token block in DSS-seq.",
    )
    parser.add_argument(
        "--property_col",
        type=str,
        default=None,
        help="CSV/LMDB field name for the target property (e.g. band_gap).",
    )
    parser.add_argument(
        "--property_name",
        type=str,
        default=None,
        help="Displayed property name token suffix (defaults to property_col).",
    )
    parser.add_argument(
        "--property_bin_size",
        type=float,
        default=0.5,
        help="Bin size for discretizing continuous properties.",
    )
    parser.add_argument(
        "--property_bin_origin",
        type=float,
        default=0.0,
        help="Bin origin for discretizing continuous properties.",
    )
    parser.add_argument(
        "--property_max_bins",
        type=int,
        default=128,
        help="Maximum number of property bins reserved in the tokenizer vocabulary.",
    )
    parser.add_argument(
        "--property_scheme",
        choices=["generic", "formation_energy_10"],
        default="generic",
        help=(
            "Independent property branch. formation_energy_10 maps [-4.5,0.5) "
            "eV/atom strictly to one-based bins 1..10."
        ),
    )
    args = parser.parse_args()

    if args.merge_shards:
        if not args.merge_output:
            raise ValueError("--merge_output is required with --merge_shards")
        merge_token_shards(args.merge_shards, args.merge_output)
        raise SystemExit(0)

    # OC20 LMDB support removed. To tokenize CIF/CSV use --root_dir mode.
    # CSV 模式：从 CSV 文件读取（与 train_cond_diff.py 一致）
    if args.root_dir:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_model = args.use_model and args.checkpoint and args.config
        if use_model and (not args.checkpoint or not args.config):
            raise ValueError("--checkpoint and --config are required when --use_model is set")

        if not args.output_dir:
            raise ValueError("--output_dir is required when using --root_dir")

        process_csv_to_tokens(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint if use_model else None,
            config_path=args.config if use_model else None,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            max_length=args.max_length,
            device=device,
            use_model=use_model,
            process_test=not args.no_test,
            splits=args.splits,
            record_range=args.record_range,
            include_property=args.include_property,
            property_col=args.property_col,
            property_name=args.property_name,
            property_bin_size=args.property_bin_size,
            property_bin_origin=args.property_bin_origin,
            property_max_bins=args.property_max_bins,
            property_scheme=args.property_scheme,
            include_space_group=args.include_space_group,
            include_crystal_system=args.include_crystal_system,
            include_coord=args.include_coord,
        )
        print(f"[Done] Saved tokenized results to {args.output_dir}")
    else:
        # 单 CIF 演示
        cif_path = Path(args.cif)
        if not cif_path.is_file():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

        structure = load_structure(str(cif_path))

        if args.use_model and args.checkpoint and args.config:
            # 使用模型编码
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = load_yaml_config(args.config)
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model = build_model_from_config(config, checkpoint, device)

            global_code_ids, global_attrs, local_code_ids, local_attrs = encode_structure_with_vq(
                model,
                structure,
                device=device,
            )

            tokens = build_crystal_tokens(
                structure,
                global_code_ids,
                global_attrs,
                local_code_ids,
                local_attrs,
                include_space_group=args.include_space_group,
                include_crystal_system=args.include_crystal_system,
                include_coord=args.include_coord,
            )
        else:
            # 使用最小示例（不加载模型）
            tokens = encode_structure_minimal(structure)

        print("\n".join(tokens))
# # 示例 1: 处理 mp_20 数据集
# python scripts/cif2hds.py --root_dir dataset/mp_20 --output_dir dataset/mp_20/tokens --checkpoint ckpts/mp_20/attn_rvq/best_model.pth --config config/data/train_cryvqvae_prep.yaml --use_model --max_length 512
# python scripts/cif2hds.py --root_dir dataset/carbon_24 --output_dir dataset/carbon_24/tokens --checkpoint ckpts/carbon_24/attn_rvq/best_model.pth --config config/data/train_cryvqvae_carbon_24.yaml --use_model --max_length 512
# python scripts/cif2hds.py --root_dir dataset/perov_5 --output_dir dataset/perov_5/tokens --checkpoint ckpts/perov_5/attn_rvq/best_model.pth --config config/data/train_cryvqvae_perov_5.yaml --use_model --max_length 512
# # 示例 2: 处理 carbon_24 数据集
# python utils/cif2hds.py \
#     --root_dir dataset/carbon_24 \
#     --output_dir results/tokens/carbon_24 \
#     --checkpoint ckpts/carbon_24/attn_rvq/best_model.pth \
#     --config config/data/train_cryvqvae_carbon_24.yaml \
#     --use_model \
#     --max_length 512

# # 示例 3: 处理 perov_5 数据集（跳过测试集）
# python utils/cif2hds.py \
#     --root_dir dataset/perov_5 \
#     --output_dir results/tokens/perov_5 \
#     --checkpoint ckpts/perov_5/attn_rvq/best_model.pth \
#     --config config/data/train_cryvqvae_perov_5.yaml \
#     --use_model \
#     --max_length 512 \
#     --no_test

# # 示例 4: 处理 mpts_52 数据集（自定义 CSV 文件名）
# python utils/cif2hds.py \
#     --root_dir dataset/mpts_52 \
#     --output_dir results/tokens/mpts_52 \
#     --checkpoint ckpts/mpts_52/attn_rvq/best_model.pth \
#     --config config/data/train_cryvqvae_mpts_52.yaml \
#     --train_csv train.csv \
#     --val_csv val.csv \
#     --test_csv test.csv \
#     --use_model \
#     --max_length 512
# 示例 5: 处理 oc_20 数据集
# python scripts/cif2hds.py \
#     --oc20_train_src ../OC20DB/train/train \
#     --oc20_test_src ../OC20DB/5939/5939 \
#     --output_dir results/cryseq/oc_20 \
#     --checkpoint ckpts/oc_20/attn_rvq/best_model.pth \
#     --config config/data/train_cryvqvae_oc20.yaml \
#     --use_model \
#     --val_ratio_oc20 0.1 \
#     --max_length 512


"""
形成能属性段
CUDA_VISIBLE_DEVICES=1 python scripts/cif2hds.py \
  --root_dir dataset/mp_20 \
  --output_dir dataset/mp_20/DSS-seqs_fe \
  --config config/attn_rvq/train_cryvqvae_mp_20.yaml \
  --checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth \
  --use_model \
  --include_property \
  --property_col formation_energy_per_atom \
  --property_name FORM_ENERGY \
  --property_bin_origin -5.0 \
  --property_bin_size 0.5 \
  --property_max_bins 13 \
  --max_length 256

"""
