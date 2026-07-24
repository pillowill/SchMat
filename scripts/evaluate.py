"""
评估 CondDiffusion 模型的脚本

功能：
1. 在测试集上评估模型损失
2. 可选：生成样本并评估生成质量（结构匹配、组成匹配等）
"""
import sys
from pathlib import Path
import torch.nn.functional as F
# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import os

import random
from collections import defaultdict

import yaml
import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm
import pandas as pd

from data.data import CIFData, collate_pool, get_separate_loaders, get_train_val_test_loader
from model.cond_diffusion import CondDiffusion
from model.attn_rvq import CrystalVQVAE

# pymatgen 可选导入（用于结构评估）
try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.cif import CifParser
    from io import StringIO
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not installed. Structure evaluation features will be disabled.")

# 全局唯一的 StructureMatcher 实例，所有评估函数共用
GLOBAL_MATCHER = StructureMatcher(ltol = 0.3, stol = 0.5, angle_tol = 10) if PYMATGEN_AVAILABLE else None
# GLOBAL_MATCHER = StructureMatcher() if PYMATGEN_AVAILABLE else None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_pretrained_vqvae(checkpoint_path: str, config_path: str, device: torch.device):
    """加载预训练的 CrystalVQVAE（仅用于提供 VQ indices 条件）

    Parameters
    ----------
    checkpoint_path : str
        预训练 VQVAE 的 .pth 权重路径
    config_path : str
        训练该 VQVAE 时使用的 yaml 配置文件路径（如 config/data/train_cryvqvae_mpts_52.yaml）
    device : torch.device
        加载设备
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    cfg = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = ckpt.get("config", {})

    # Infer embedding-table size from checkpoint to avoid num_elements drift.
    inferred_num_elements = cfg.get("num_elements", 100)
    emb_key = "encoder.node_embedding.weight"
    if emb_key in state_dict and hasattr(state_dict[emb_key], "shape"):
        inferred_num_elements = int(state_dict[emb_key].shape[0])

    model = CrystalVQVAE(
        hidden_dim=cfg.get("hidden_dim", 256),
        num_quantizers=cfg.get("num_quantizers", 4),  # 向后兼容
        num_quantizers_global=cfg.get("num_quantizers_global", None),  # 全局量化器数量
        num_quantizers_local=cfg.get("num_quantizers_local", None),   # 局部量化器数量
        num_layers=cfg.get("num_layers", 4),
        codebook_size=cfg.get("codebook_size", 256),
        num_space_groups=cfg.get("num_space_groups", 230),
        max_cn=cfg.get("max_cn", 20),
        num_elements=inferred_num_elements,
        commitment_weight=cfg.get("commitment_weight", 0.0),
        loss_weights=cfg.get("loss_weights", None),
        sample_codebook_temp=cfg.get("sample_codebook_temp", 0.1),
        vq_type=cfg.get("vq_type", "residual"),
        aggregator_dropout=cfg.get("aggregator_dropout", 0.1),
        aggregator_attn_dropout=cfg.get("aggregator_attn_dropout", 0.0),
        aggregator_use_positional=cfg.get("aggregator_use_positional", True),
        aggregator_positional_scale=cfg.get("aggregator_positional_scale", 1.0),
        residual_use_outer_aggregator=cfg.get("residual_use_outer_aggregator", True),
        residual_attn_use_outer_aggregator=cfg.get("residual_attn_use_outer_aggregator", False),
        stacked_vq_use_outer_aggregator=cfg.get("stacked_vq_use_outer_aggregator", True),
        enable_residual_codebook_params=cfg.get("enable_residual_codebook_params", False),
        residual_codebook_params=cfg.get("residual_codebook_params", None),
    ).to(device)

    # strict=False cannot ignore same-key shape mismatch; pre-filter by shape.
    model_state = model.state_dict()
    filtered_state = {}
    shape_mismatch = []
    for k, v in state_dict.items():
        if k in model_state:
            if tuple(v.shape) == tuple(model_state[k].shape):
                filtered_state[k] = v
            else:
                if len(shape_mismatch) < 8:
                    shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if shape_mismatch:
        print(f"  Warning: {len(shape_mismatch)} shape-mismatched keys in VQVAE checkpoint (ignored)")
        for k, src_shape, dst_shape in shape_mismatch:
            print(f"    - {k}: ckpt{src_shape} vs model{dst_shape}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in VQVAE checkpoint (ignored)")
    if missing:
        print(f"  Warning: {len(missing)} missing keys in VQVAE checkpoint")

    model.eval().requires_grad_(False)
    return model


def prepare_batch(batch_dict, device, pretrained_vqvae=None):
    """
    将 collate_pool 的输出转换为 CondDiffusion 所需的 Batch。
    1) 将 pyg_data 列表打包成 torch_geometric Batch
    2) 如果提供预训练 VQVAE，则计算 global/local VQ indices，并写入 Batch 属性
    """
    pyg_list = batch_dict.get("pyg_data", None)
    batch = Batch.from_data_list(pyg_list).to(device)

    # 计算 VQ indices 作为条件
    if pretrained_vqvae is not None:
        with torch.no_grad():
            g, h = pretrained_vqvae.encode(
                batch.atom_types,
                batch.frac_coords,
                batch.lattices,
                batch.natoms,
                batch.batch,
            )
            _, _, indices_g, indices_l, _ = pretrained_vqvae.quantize(g, h)
        batch.global_vq_indices = indices_g  # (B,)
        batch.local_vq_indices = indices_l   # (total_atoms,)
    else:
        batch.global_vq_indices = None
        batch.local_vq_indices = None

    return batch


def process_one_rms(pred_st, gt_st, is_pred_valid, matcher):
    """
    计算单个生成结构与参考结构的 RMSD（参考 Mat2Seq postprocess.py）
    """
    if not is_pred_valid:
        return None
    try:
        rms_dist = matcher.get_rms_dist(pred_st, gt_st)
        rms_dist = None if rms_dist is None else rms_dist[0]
        return rms_dist
    except Exception:
        return None


def get_match_rate_and_rms(gen_structs_list, true_structs_list, matcher):
    """
    按照 Mat2Seq postprocess.py 的方式计算 match_rate 和 rms_dist

    Args:
        gen_structs_list: List[List[Structure]]，每个测试结构对应一个生成结构列表
        true_structs_list: List[Structure]，参考结构列表
        matcher: StructureMatcher 实例

    Returns:
        dict: {'match_rate': float, 'rms_dist': float}
    """
    assert len(gen_structs_list) == len(true_structs_list), \
        f"gen_structs_list ({len(gen_structs_list)}) and true_structs_list ({len(true_structs_list)}) must have same length"

    rms_dists = []
    for i in tqdm(range(len(gen_structs_list)), desc="Computing match_rate and rms_dist"):
        tmp_rms_dists = []
        for j in range(len(gen_structs_list[i])):
            try:
                struct_valid = is_valid_structure(gen_structs_list[i][j])
                rmsd = process_one_rms(gen_structs_list[i][j], true_structs_list[i], struct_valid, matcher)
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            except Exception:
                pass
        if len(tmp_rms_dists) == 0:
            rms_dists.append(None)
        else:
            rms_dists.append(np.min(tmp_rms_dists))

    rms_dists = np.array(rms_dists)
    match_rate = sum(rms_dists != None) / len(gen_structs_list)
    mean_rms_dist = rms_dists[rms_dists != None].mean() if sum(rms_dists != None) > 0 else None
    return {"match_rate": match_rate, "rms_dist": mean_rms_dist}


@torch.no_grad()
def evaluate_generation(
    model,
    loader,
    device,
    pretrained_vqvae=None,
    n_samples_per_structure: int = 10,
    cond_scale: float = 2.0,
    max_eval_samples: int = None,
    save_path: str | None = None,
):
    """
    评估生成质量（需要 pymatgen）

    返回指标：
    - valid_rate: 有效生成样本的比例
    - unique_rate: 唯一结构的比例
    - match_rate: 匹配率（按照 Mat2Seq 方式计算）
    - rms_dist: 平均 RMSD（按照 Mat2Seq 方式计算）
    """
    if not PYMATGEN_AVAILABLE:
        print("Warning: pymatgen not available. Skipping generation evaluation.")
        return None

    model.eval()
    if pretrained_vqvae is not None:
        pretrained_vqvae.eval()
        # 评估阶段关键提示：如果扩散训练中联合微调了 VQ-VAE 聚合器，
        # 这里必须加载同一份（已微调的）VQ-VAE checkpoint，否则条件向量分布会不一致。
        try:
            vq_type = getattr(pretrained_vqvae, "vq_type", None)
            num_quantizers = getattr(pretrained_vqvae, "num_quantizers", None)
            print(f"[Eval] Using pretrained VQVAE: vq_type={vq_type}, num_quantizers={num_quantizers}")
        except Exception:
            pass

    all_metrics = defaultdict(list)
    per_structure_rows = []  # 用于保存到本地的结果（每个测试结构一行）

    # 使用全局唯一的 StructureMatcher（用于计算 match_rate 和 rms_dist）
    matcher = GLOBAL_MATCHER
    # 根据 DataLoader 的 dataset 构建 cif_id -> 参考结构 的映射
    id_to_struct = {}
    dataset = getattr(loader, "dataset", None)

    # 处理 CIF CSV 数据集
    if PYMATGEN_AVAILABLE and dataset is not None and hasattr(dataset, "id_prop_data"):
        for cry in getattr(dataset, "id_prop_data", []):
            cif_str = cry.get("cif", None)
            if cif_str is None:
                continue
            cif_io = StringIO(cif_str)
            try:
                parser = CifParser(cif_io)
                struct = parser.get_structures()[0]
            except Exception:
                continue
            cif_id = cry.get("material_id", cry.get("id", None))
            if cif_id is not None:
                id_to_struct[cif_id] = struct

    # OC20 LMDB support removed: no special handling for Subset-wrapped OC20 datasets

    # 以"结构"为单位显示进度条，而不是以 batch 为单位
    total_structs = len(loader.dataset) if hasattr(loader, "dataset") else None
    if hasattr(loader, "sampler") and isinstance(loader.sampler, torch.utils.data.sampler.SubsetRandomSampler):
        total_structs = len(loader.sampler.indices)
    if max_eval_samples is not None and total_structs is not None:
        total_structs = min(total_structs, max_eval_samples)
    pbar = tqdm(total=total_structs, desc="Generating samples") if total_structs is not None else None

    eval_count = 0

    for batch_dict, batch_cif_ids in loader:
        if max_eval_samples is not None and eval_count >= max_eval_samples:
            break

        batch = prepare_batch(batch_dict, device, pretrained_vqvae)

        # 获取批次中的每个结构
        batch_size = batch.batch.max().item() + 1
        natoms_list = batch.natoms.tolist()

        for i in range(batch_size):
            if max_eval_samples is not None and eval_count >= max_eval_samples:
                break

            # 获取当前结构的 VQ indices
            start_idx = sum(natoms_list[:i])
            end_idx = start_idx + natoms_list[i]

            # 为该结构构造单图 Batch；并行采样由 CondDiffusion.sample(n_samples=...) 内部完成
            pyg_list = batch_dict.get("pyg_data", None)
            if pyg_list is None or i >= len(pyg_list):
                continue
            single_batch = Batch.from_data_list([pyg_list[i]]).to(device)

            # 准备条件输入（离散 VQ 索引，由 CondDiffusion 内部查询 codebook）
            # 注意：保持正确的形状以支持 ResidualVQ 的多个量化器
            # indices_g 可能是 (B, num_quantizers) 或 (B,)，需要保持维度
            indices_g_one = batch.global_vq_indices[i]
            # 如果是 1D 且长度 > 1，可能是 (num_quantizers,)，需要添加 batch 维度 -> (1, num_quantizers)
            if indices_g_one.dim() == 1 and indices_g_one.shape[0] > 1:
                indices_g_one = indices_g_one.unsqueeze(0)  # (num_quantizers,) -> (1, num_quantizers)
            elif indices_g_one.dim() == 0:
                indices_g_one = indices_g_one.unsqueeze(0)  # 标量 -> (1,)
            
            indices_l_one = batch.local_vq_indices[start_idx:end_idx]
            # indices_l 可能是 (n_atoms, num_quantizers) 或 (n_atoms,)，保持原样

            # ---------- 额外的形状健壮性检查（避免误展平导致聚合错误） ----------
            if pretrained_vqvae is not None:
                vq_type = getattr(pretrained_vqvae, "vq_type", None)

                # 分别获取全局和局部的量化器数量（向后兼容旧模型）
                q_g = getattr(pretrained_vqvae, "num_quantizers_global", None)
                q_l = getattr(pretrained_vqvae, "num_quantizers_local", None)
                q_shared = getattr(pretrained_vqvae, "num_quantizers", None)
                if q_g is None:
                    q_g = q_shared
                if q_l is None:
                    q_l = q_shared

                if vq_type == "residual":
                    # -------- Global indices 检查 --------
                    if isinstance(q_g, int) and q_g > 1:
                        if indices_g_one.dim() == 2:
                            assert indices_g_one.shape[0] == 1 and indices_g_one.shape[1] == q_g, \
                                f"Unexpected indices_g_one shape: {tuple(indices_g_one.shape)}; expected (1, {q_g})"
                        elif indices_g_one.dim() == 1:
                            # (q_g,) 或 (1,)
                            assert indices_g_one.shape[0] in (q_g, 1), \
                                f"Unexpected indices_g_one shape: {tuple(indices_g_one.shape)}; expected ({q_g},) or (1,)"
                        else:
                            raise ValueError(
                                f"Unexpected indices_g_one dim: {indices_g_one.dim()} with shape {tuple(indices_g_one.shape)}"
                            )

                    # -------- Local indices 检查 --------
                    if isinstance(q_l, int) and q_l > 1:
                        if indices_l_one.dim() == 2:
                            assert indices_l_one.shape[1] == q_l, \
                                f"Unexpected indices_l_one shape: {tuple(indices_l_one.shape)}; expected (n_atoms, {q_l})"
                        elif indices_l_one.dim() == 1:
                            # 允许 (n_atoms,) 这种“单量化器/旧格式”
                            pass
                        else:
                            raise ValueError(
                                f"Unexpected indices_l_one dim: {indices_l_one.dim()} with shape {tuple(indices_l_one.shape)}"
                            )

            gen_structures = []

            from pymatgen.core import Structure
            from pymatgen.io.ase import AseAtomsAdaptor

            cond_inputs = [indices_g_one, indices_l_one]
            final_atoms = model.sample(
                batch=single_batch,
                cond_inputs=cond_inputs,
                n_samples=n_samples_per_structure,
                cond_scale=cond_scale,
                step_lr=1e-5,
                return_trajectory=False,
                stream=False,
            )

            if final_atoms is None:
                continue

            # 转换为 Structure 对象列表（生成结构）
            atoms_list = final_atoms if isinstance(final_atoms, list) else [final_atoms]
            for atoms in atoms_list:
                if isinstance(atoms, Structure):
                    gen_structures.append(atoms)
                elif hasattr(atoms, 'get_chemical_symbols'):
                    gen_structures.append(AseAtomsAdaptor.get_structure(atoms))

            if not gen_structures:
                continue

            # 参考结构（ground truth）：根据当前样本的 cif_id 从映射中获取
            # 对于 OC20 数据，也可以从 batch_dict 中的 pyg_data 重建
            ref_st = None
            if batch_cif_ids is not None and i < len(batch_cif_ids):
                cif_id = batch_cif_ids[i]
                ref_st = id_to_struct.get(cif_id, None)

            # 如果从映射中没找到（OC20 模式），尝试从 pyg_data 重建
            if ref_st is None and PYMATGEN_AVAILABLE:
                try:
                    pyg_list = batch_dict.get("pyg_data", None)
                    if pyg_list is not None and i < len(pyg_list):
                        pyg_data = pyg_list[i]
                        # 从 pyg_data 重建 Structure
                        from pymatgen.core import Structure, Lattice
                        from pymatgen.io.ase import AseAtomsAdaptor
                        from ase import Atoms

                        # 获取原子信息
                        atom_types = pyg_data.atom_types.cpu().numpy()
                        frac_coords = pyg_data.frac_coords.cpu().numpy()
                        lattice = pyg_data.lattices[0].cpu().numpy()  # (3, 3)

                        # 构建 Structure
                        ref_st = Structure(
                            lattice=Lattice(lattice),
                            species=atom_types,
                            coords=frac_coords,
                            coords_are_cartesian=False,
                        )
                except Exception as e:
                    # 如果重建失败，ref_st 保持为 None
                    pass

            # -----------------------------
            # 评估指标计算
            # 0. 有效样本比例 valid_samples
            # 1. 唯一性 unique
            # 2. 结构匹配（是否至少有一个匹配）
            # 3. 结构匹配率（匹配的个数 / 生成的有效结构个数）
            # 4. 按照 Mat2Seq 方式计算的匹配率和 RMSD
            # -----------------------------
            valid_structures = filter_valid_structures(gen_structures)

            # 有效样本比例
            all_metrics["valid_samples"].append(
                len(valid_structures) / len(gen_structures) if len(gen_structures) > 0 else 0.0
            )

            # 初始化 match_rate 和 rms_dist（当前结构）
            match_flag = 0.0
            rms_dist_value = None

            if valid_structures:
                # 唯一性
                num_unique = count_unique_structures(valid_structures)
                all_metrics["unique"].append(
                    num_unique / len(valid_structures)
                )

                # 只有在存在参考结构时才计算匹配相关指标
                if ref_st is not None:
                    # 结构匹配（是否至少有一个匹配）
                    num_match = test_structure_matching(valid_structures, ref_st)
                    all_metrics["structure_matching"].append(
                        float(num_match > 0)
                    )
                    all_metrics["structure_matching_ratio"].append(
                        num_match / len(valid_structures) if len(valid_structures) > 0 else 0.0
                    )

                    # 按照 Mat2Seq 方式计算 match_rate 和 rms_dist
                    # 对当前测试结构，遍历所有生成的有效结构，计算 RMSD，取最小值
                    tmp_rms_dists = []
                    for gen_st in valid_structures:
                        try:
                            struct_valid = is_valid_structure(gen_st)
                            rmsd = process_one_rms(gen_st, ref_st, struct_valid, matcher)
                            if rmsd is not None:
                                tmp_rms_dists.append(rmsd)
                        except Exception:
                            pass

                    if len(tmp_rms_dists) > 0:
                        rms_dist_value = float(np.min(tmp_rms_dists))
                        match_flag = 1.0

                    # 累积到全局指标
                    all_metrics["match_rate"].append(match_flag)
                    if rms_dist_value is not None:
                        all_metrics["rms_dist"].append(rms_dist_value)

                    # 计算累积的平均值
                    avg_match_rate = np.mean(all_metrics["match_rate"]) if len(all_metrics["match_rate"]) > 0 else 0.0
                    avg_rms_dist = np.mean(all_metrics["rms_dist"]) if len(all_metrics["rms_dist"]) > 0 else None

                    # 打印当前结构和累积的平均值
                    current_rms_str = f"{rms_dist_value:.4f}" if rms_dist_value is not None else "N/A"
                    avg_rms_str = f"{avg_rms_dist:.4f}" if avg_rms_dist is not None else "N/A"
                    print(f"\nStructure {eval_count + 1} (ID: {batch_cif_ids[i]}):")
                    print(f"  Current - match_rate: {match_flag:.4f}, rms_dist: {current_rms_str}")
                    print(f"  Average - match_rate: {avg_match_rate:.4f}, rms_dist: {avg_rms_str}")

                    # 收集当前结构的详细结果（用于保存）
                    row = {
                        "material_id": batch_cif_ids[i],
                        "natoms": natoms_list[i],
                        "valid_samples": len(valid_structures) / len(gen_structures) if len(gen_structures) > 0 else 0.0,
                        "unique": num_unique / len(valid_structures) if len(valid_structures) > 0 else 0.0,
                        # "structure_matching": float(num_match > 0),
                        # "structure_matching_ratio": num_match / len(valid_structures) if len(valid_structures) > 0 else 0.0,
                        # "composition_matching": num_comp_match / len(valid_structures) if len(valid_structures) > 0 else 0.0,
                        # "crystal_system_matching": num_crystal_match / len(valid_structures) if len(valid_structures) > 0 else 0.0,
                        # "lattice_system_matching": num_lattice_match / len(valid_structures) if len(valid_structures) > 0 else 0.0,
                        "match_rate": match_flag,
                        "rms_dist": rms_dist_value,
                        "ref_structure": ref_st.to(fmt="cif"),
                    }
                    # 将所有生成结构转为 cif 字符串，按列存储 gen_structure_0, gen_structure_1, ...
                    for idx_gen, st in enumerate(gen_structures):
                        row[f"gen_structure_{idx_gen}"] = st.to(fmt="cif")
                    per_structure_rows.append(row)

            eval_count += 1
            if pbar is not None:
                # 计算当前累积的 match_rate 和 rms_dist 平均值
                current_match_rate = np.mean(all_metrics["match_rate"]) if len(all_metrics["match_rate"]) > 0 else 0.0
                current_rms_dist = np.mean(all_metrics["rms_dist"]) if len(all_metrics["rms_dist"]) > 0 else None

                # 更新进度条显示
                postfix_dict = {
                    "match_rate": f"{current_match_rate:.4f}",
                }
                if current_rms_dist is not None:
                    postfix_dict["rms_dist"] = f"{current_rms_dist:.4f}"
                else:
                    postfix_dict["rms_dist"] = "N/A"

                pbar.set_postfix(postfix_dict)
                pbar.update(1)



    if pbar is not None:
        pbar.close()

    # 保存所有结构及指标到本地（类似 Chemeleon 的 results.csv）
    if save_path is not None and len(per_structure_rows) > 0:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        path_results = save_dir / "results.csv"
        df_results = pd.DataFrame(per_structure_rows)
        df_results.to_csv(path_results, index=False)
        print(f"\n✓ Saved generated structures and metrics to {path_results}")

    # 计算平均指标
    avg_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in all_metrics.items()}
    return avg_metrics


def is_valid_structure(struct):
    """
    检查结构是否有效（参考 Mat2Seq postprocess.py）
    包括组成有效性和结构有效性
    """
    if not PYMATGEN_AVAILABLE:
        return True

    from collections import Counter

    # 组成有效性检查（参考 Mat2Seq 的 smact_validity）
    try:
        atom_types = [str(specie) for specie in struct.species]
        elem_counter = Counter(atom_types)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        if len(composition) == 0:
            return False
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        comps = tuple(counts.astype("int").tolist())

        # 简化版组成有效性：检查元素是否合理
        comp_valid = True  # 可以进一步实现 smact_validity
    except Exception:
        comp_valid = False

    # 结构有效性检查（参考 Mat2Seq 的 structure_validity）
    try:
        dist_mat = struct.distance_matrix
        # Pad diagonal with a large number
        dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (0.5 + 10.))
        struct_valid = (dist_mat.min() >= 0.5) and (struct.volume >= 0.1)
    except Exception:
        struct_valid = False

    return comp_valid and struct_valid


def filter_valid_structures(structures):
    """过滤有效结构（基于物理约束）"""
    if not PYMATGEN_AVAILABLE:
        return structures

    valid = []
    for st in structures:
        if is_valid_structure(st):
            valid.append(st)
    return valid


def count_unique_structures(structures):
    """计算唯一结构数量"""
    if not PYMATGEN_AVAILABLE:
        return len(structures)

    # 先过滤掉明显异常的结构（例如 0 个原子、退化晶格等），避免 Niggli 化简时报错
    clean_structs = []
    for st in structures:
        try:
            if st is None:
                continue
            # 过滤 0 原子结构
            n_sites = getattr(st, "num_sites", len(st))
            if n_sites == 0:
                continue
            # 过滤体积过小或晶格矩阵非法的结构
            if st.volume <= 0:
                continue
            _ = st.lattice.matrix  # 触发潜在的非法晶格错误
            # 额外检查：尝试做 Niggli 化简，如果失败则跳过该结构
            try:
                _ = st.get_reduced_structure(reduction_algo="niggli")
            except Exception:
                # Niggli 化简失败，跳过该结构
                continue
            clean_structs.append(st)
        except Exception:
            continue

    if not clean_structs:
        return 0

    # 使用 try-except 包裹 group_structures，防止某些结构在内部 Niggli 化简时出错
    try:
        grouped = GLOBAL_MATCHER.group_structures(clean_structs)
        return len(grouped)
    except (ValueError, RuntimeError) as e:
        # 如果 group_structures 失败，回退到逐个检查结构（跳过有问题的结构）
        # 这通常发生在某些结构在 Niggli 化简时变成空结构
        print(f"Warning: group_structures failed with error: {e}")
        print(f"  Falling back to per-structure matching (may be slower)")
        
        # 逐个尝试匹配，跳过有问题的结构
        unique_groups = []
        for st in clean_structs:
            try:
                # 检查是否与已有组匹配
                matched = False
                for group in unique_groups:
                    try:
                        if GLOBAL_MATCHER.fit(group[0], st):
                            group.append(st)
                            matched = True
                            break
                    except Exception:
                        # 匹配过程中出错，跳过这个组
                        continue
                
                if not matched:
                    # 创建新组
                    unique_groups.append([st])
            except Exception:
                # 处理该结构时出错，跳过
                continue
        
        return len(unique_groups)


def test_structure_matching(st_list, ref_st):
    """结构匹配：生成结构中与参考结构匹配的个数"""
    if not PYMATGEN_AVAILABLE:
        return 0
    num_match = 0
    for st in st_list:
        if GLOBAL_MATCHER.fit(ref_st, st):
            num_match += 1
    return num_match


def test_composition_matching(st_list, ref_st):
    """组成匹配：生成结构中与参考结构化学计量比一致的个数"""
    if not PYMATGEN_AVAILABLE:
        return 0
    num_match = 0
    for st in st_list:
        if ref_st.composition == st.composition:
            num_match += 1
    return num_match


def test_crystal_system_matching(st_list, ref_st, symprec=0.1, angle_tolerance=10):
    """晶系匹配：生成结构中晶系与参考结构一致的个数"""
    if not PYMATGEN_AVAILABLE:
        return 0
    num_match = 0
    ref_sga = SpacegroupAnalyzer(
        ref_st, symprec=symprec, angle_tolerance=angle_tolerance
    )
    ref_system = ref_sga.get_crystal_system()
    for st in st_list:
        sga = SpacegroupAnalyzer(
            st, symprec=symprec, angle_tolerance=angle_tolerance
        )
        if sga.get_crystal_system() == ref_system:
            num_match += 1
    return num_match


def test_lattice_system_matching(st_list, ref_st, symprec=0.1, angle_tolerance=10):
    """点阵类型匹配：生成结构中点阵类型与参考结构一致的个数"""
    if not PYMATGEN_AVAILABLE:
        return 0
    num_match = 0
    ref_sga = SpacegroupAnalyzer(
        ref_st, symprec=symprec, angle_tolerance=angle_tolerance
    )
    ref_lattice = ref_sga.get_lattice_type()
    for st in st_list:
        sga = SpacegroupAnalyzer(
            st, symprec=symprec, angle_tolerance=angle_tolerance
        )
        if sga.get_lattice_type() == ref_lattice:
            num_match += 1
    return num_match


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conditional Diffusion Model")
    parser.add_argument("--config", type=str, default="config/cond_diffusion/train_cond_diffusion_carbon_24.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Override data path from config")
    # OC20 support removed: --oc20_src argument deleted
    parser.add_argument("--pretrained_vqvae_path", type=str, default=None,
                       help="Override pretrained VQVAE checkpoint path from config")
    parser.add_argument("--pretrained_vqvae_config", type=str, default=None,
                       help="Override pretrained VQVAE config yaml path from config")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--eval_generation", action="store_true",
                       help="Evaluate generation quality (requires pymatgen)")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of samples per structure for generation evaluation")
    parser.add_argument("--cond_scale", type=float, default=2.0,
                       help="Conditional scale for sampling")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Maximum number of test samples to evaluate (None = all)")
    parser.add_argument("--save_path", type=str, default="results/mp_20/fixed_atom_types",
                       help="Directory to save generated structures and metrics (CSV with CIF strings)")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 路径解析函数：支持相对路径和绝对路径
    def resolve_path(path):
        if not path:
            return None
        if os.path.isabs(path):
            return path
        # 处理 ../ 开头的路径
        if path.startswith("../"):
            cwd_path = os.path.abspath(os.path.join(os.getcwd(), path))
            if os.path.exists(cwd_path):
                return cwd_path
            return os.path.abspath(os.path.join(project_root.parent, path[3:]))
        # 先尝试当前目录，再尝试项目根目录
        cwd_path = os.path.abspath(path)
        return cwd_path if os.path.exists(cwd_path) else os.path.abspath(os.path.join(project_root, path))

    # 解析并验证路径
    def check_path(path, name):
        if not path:
            return None
        resolved = resolve_path(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"{name} not found: {resolved}\n  Original: {path}\n  Project root: {project_root}")
        return resolved
    
    # 解析所有路径
    config["data_path"] = check_path(
        args.data_path or config.get("data_path", "dataset/mp_20"), 
        "Data path"
    )
    config["pretrained_vqvae_path"] = check_path(
        args.pretrained_vqvae_path or config.get("pretrained_vqvae_path"), 
        "Pretrained VQVAE checkpoint"
    )
    if args.pretrained_vqvae_config is not None:
        config["pretrained_vqvae_config"] = args.pretrained_vqvae_config
    checkpoint_path = check_path(args.checkpoint, "Model checkpoint")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(config.get("random_seed", 42))
    
    print(f"Using device: {device}")
    print(f"Data path: {config['data_path']}")
    if config.get("pretrained_vqvae_path"):
        print(f"Loading crystal-vqvae checkpoint: {config['pretrained_vqvae_path']}")
    print(f"Loading model checkpoint: {checkpoint_path}")
    
    # 先加载扩散 checkpoint，后续可用其中保存的训练配置纠正当前 yaml 的模型超参数。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 优先使用 checkpoint 内保存的关键结构参数，避免评估时 yaml 与训练配置不一致。
    ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_critical_keys = [
        "time_dim", "hidden_dim", "cond_dim", "num_layers", "max_atoms",
        "act_fn", "dis_emb", "num_freqs", "edge_style", "cutoff", "max_neighbors",
        "ln", "ip", "smooth", "pred_atom_types", "cond_guide", "diffuse_atom_types",
        "timesteps", "beta_schedule", "d3pm_hybrid_coeff",
    ]
    for k in model_critical_keys:
        if k in ckpt_cfg:
            if k in config and config[k] != ckpt_cfg[k]:
                print(f"[Config override] {k}: yaml={config[k]} -> ckpt={ckpt_cfg[k]}")
            config[k] = ckpt_cfg[k]

    # 加载预训练 VQVAE
    pretrained_vqvae = None
    if config.get("pretrained_vqvae_path"):
        vqvae_config_path = config.get("pretrained_vqvae_config", "")
        pretrained_vqvae = load_pretrained_vqvae(
            config["pretrained_vqvae_path"],
            vqvae_config_path,
            device,
        )
        print("✓ Pretrained VQVAE loaded")
    
    # 加载模型
    model = CondDiffusion(
        config,
        pretrained_vqvae_model=pretrained_vqvae
    ).to(device)
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    filtered_state = {}
    shape_mismatch = []
    for k, v in state_dict.items():
        if k in model_state:
            if tuple(v.shape) == tuple(model_state[k].shape):
                filtered_state[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if shape_mismatch:
        print(f"[Load] shape-mismatched keys ignored: {len(shape_mismatch)}")
        for k, src_shape, dst_shape in shape_mismatch[:12]:
            print(f"  - {k}: ckpt{src_shape} vs model{dst_shape}")
    if missing:
        print(f"[Load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load] unexpected keys: {len(unexpected)}")
    
    model.eval()
    print("✓ Model loaded")

    # 评估时确保使用条件引导（否则 cond_inputs 会被忽略）
    if args.eval_generation and not config.get("cond_guide", False):
        print("Warning: config.cond_guide is False but --eval_generation was set. "
              "Generation will run WITHOUT VQ conditions (cond_inputs ignored).")
    
    # 数据加载：CIF CSV 模式（OC20 support removed; always use CIF CSV mode）
        data_root = config.get("data_path", "dataset/mp_20")
        use_single_csv = config.get("use_single_csv", False)
        train_csv = config.get("train_csv", "train.csv")
        val_csv = config.get("val_csv", "val.csv")
        test_csv = config.get("test_csv", "test.csv")
        return_test = True

        if use_single_csv:
            print('dataset random split mode')
            # 单一 CSV：例如 mpts 上只有 data.csv
            csv_file = config.get("csv_file", "data.csv")
            train_ratio = float(config.get("train_ratio", 0.8))
            val_ratio = float(config.get("val_ratio", 0.1))
            # 剩余部分作为 test_ratio（这里只用不到 test_loader，可以留作将来扩展）
            test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
            print(f"Loading CIF-based data from single CSV {csv_file} in {data_root}")
            full_dataset = CIFData(
                root_dir=data_root,
                proper=config.get("property", "formation_energy"),
                random_seed=config["random_seed"],
                csv_file=csv_file,
                shuffle=True,
            )
            train_loader, val_loader, test_loader = get_train_val_test_loader(
                full_dataset,
                collate_fn=collate_pool,
                batch_size=config["batch_size"],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                return_test=return_test,
                num_workers=config.get("workers", 4),
                pin_memory=False,
            )

            total_size = len(full_dataset)
            train_size = int(train_ratio * total_size)
            valid_size = int(val_ratio * total_size)
            test_size = total_size - train_size - valid_size

            print(
                f"Dataset size (single CSV): {total_size}\n"
                f"Train samples: {train_size}, Val samples: {valid_size}, Test samples: {test_size}\n"
                f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
            )
            test_samples = len(test_loader.sampler.indices)
        else:
            loader_kwargs = dict(
                root_dir=data_root,
                collate_fn=collate_pool,
                train_csv=train_csv,
                val_csv=val_csv,
                test_csv=test_csv,
                proper=config.get("property", "formation_energy"),
                train_batch_size=config.get("train_batch_size", config.get("batch_size", 256)),
                val_batch_size=config.get("val_batch_size", config.get("batch_size", 256)),
                test_batch_size=config.get("test_batch_size", config.get("batch_size", 256)),
                num_workers=config.get("workers", 0),
                pin_memory=config.get("pin_memory", True),
                random_seed=config.get("random_seed", 42),
                cond_guide=config.get("cond_guide", True),
                return_test=True,
            )

            train_loader, val_loader, test_loader = get_separate_loaders(**loader_kwargs)
            if test_loader is None:
                print("Warning: test_loader is None. 请确保配置中设置了 test_csv 并启用 return_test。")
                return
            test_samples = len(test_loader.dataset)

    print(f"✓ Test dataset loaded: {test_samples} samples")

    
    # 评估生成质量（可选）
    gen_metrics = None
    if args.eval_generation:
        print("\n" + "="*60)
        print("Evaluating generation quality...")
        print("="*60)
        gen_metrics = evaluate_generation(
            model, test_loader, device, pretrained_vqvae,
            n_samples_per_structure=args.n_samples,
            cond_scale=args.cond_scale,
            max_eval_samples=args.max_eval_samples,
            save_path=args.save_path,
        )
        if gen_metrics:
            print("\nGeneration Metrics:")
            for key, value in gen_metrics.items():
                print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
    # gen-1:
    # CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --checkpoint ckpts/mpts_52/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --data_path dataset/mpts_52 --pretrained_vqvae_path ckpts/mpts_52/attn_rvq-MinaRVQ/best_model.pth --device cuda --eval_generation --n_samples 20 --cond_scale 2.0 --save_path results/mpts_52/cond_diffusion
    #  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --data_path dataset/mp_20 --pretrained_vqvae_path ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth --device cuda --eval_generation --n_samples 20 --cond_scale 2.0 --save_path results/mp_20/cond_diffusion
    #  CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --checkpoint ckpts/perov_5/cond_diffusion/best_diffusion.pth --config config/cond_diff
    # CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --checkpoint ckpts/mp_20/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --data_path dataset/mp_20 --pretrained_vqvae_path ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth --device cuda --eval_generation --n_samples 20 --cond_scale 2.0 --save_path results/mp_20/cond_diffusion
    # CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py --checkpoint ckpts/perov_5/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_perov_5.yaml --data_path dataset/perov_5 --pretrained_vqvae_path ckpts/perov_5/attn_rvq-MinaRVQ/best_model.pth --device cuda --eval_generation --n_samples 1 --cond_scale 2.0 --save_path results/perov_5/cond_diffusion-1
    # CUDA_VISIBLE_DEVICES=5 python scripts/evaluate.py --checkpoint ckpts/carbon_24/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_carbon_24.yaml --data_path dataset/carbon_24 --pretrained_vqvae_path ckpts/carbon_24/attn_rvq-MinaRVQ/best_model.pth --device cuda --eval_generation --n_samples 1 --cond_scale 2.0 --save_path results/carbon_24/cond_diffusion-1
    # CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py --checkpoint ckpts/mpts_52/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --data_path dataset/mpts_52 --pretrained_vqvae_path ckpts/mpts_52/attn_rvq-MinaRVQ/best_model.pth --device cuda --eval_generation --n_samples 20 --cond_scale 2.0 --save_path results/mpts_52/cond_diffusion
    # python scripts/cif2hds.py --root_dir dataset/mp_20 --output_dir dataset/mp_20/DSS-seqs_bandgap --checkpoint ckpts/mp_20/attn_rvq-MinaRVQ/best_model.pth --config config/attn_rvq/train_cryvqvae_mp_20.yaml --use_model --max_length 256 --include_property --property_col band_gap --property_name band_gap --property_bin_size 0.5 --property_bin_origin 0.0 --property_max_bins 64


    # 消融实验，VQ
    # CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --checkpoint ckpts/mp_20/ablation/VQ/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --data_path dataset/mp_20 --pretrained_vqvae_path ckpts/mp_20/attn_rvq-VQ/best_model.pth --device cuda --eval_generation --n_samples 1 --cond_scale 2.0 --save_path results/mp_20/ablation/VQ/cond_diffusion-1
    # CUDA_VISIBLE_DEVICES=7 python scripts/evaluate.py --checkpoint ckpts/mpts_52/ablation/VQ/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --data_path dataset/mpts_52 --pretrained_vqvae_path ckpts/mpts_52/attn_rvq-VQ/best_model.pth --device cuda --eval_generation --n_samples 1 --cond_scale 2.0 --save_path results/mpts_52/ablation/VQ/cond_diffusion-1

    # 消融实验，RVQ
    # CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py --checkpoint ckpts/mpts_52/ablation/RVQ/cond_diffusion/best_diffusion.pth --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --data_path dataset/mpts_52 --pretrained_vqvae_path ckpts/mpts_52/attn_rvq-RVQ/best_model.pth --device cuda --eval_generation --n_samples 1 --cond_scale 2.0 --save_path results/mpts_52/ablation/RVQ/cond_diffusion-1
