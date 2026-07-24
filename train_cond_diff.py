"""
训练 CondDiffusin 模型的脚本（支持多GPU训练）

核心数据流：
    CIFData(cond_guide=True, collate_pool) -> batch_dict["pyg_data"] (PyG Data 列表)
    -> Batch.from_data_list -> (atom_types, frac_coords, lattices, natoms, batch)
    -> 条件来源二选一：
       A) 预训练 CrystalVQVAE 编码 + 量化 -> (global_vq_indices, local_vq_indices)
       B) 预训练 crystal_token_gpt 根据输入成分生成离散 code token -> 解析得到 (global_vq_indices, local_vq_indices)
    -> CondDiffusion.forward(batch)  (内部再映射为 VQ 连续向量并做 cond-drop)
"""
import argparse
import os
import random
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from tqdm import tqdm

from data.data import CIFData, collate_pool, get_separate_loaders, get_train_val_test_loader
from model.cond_diffusion import CondDiffusion
from model.attn_rvq import CrystalVQVAE

# pymatgen Element 用于原子序数 -> 元素符号
try:
    from pymatgen.core.periodic_table import Element
    PYMATGEN_ELEMENT_AVAILABLE = True
except Exception:
    PYMATGEN_ELEMENT_AVAILABLE = False

# wandb 可选导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def as_bool(value, default: bool = False) -> bool:
    """Parse bool-like config values robustly (supports yaml string forms)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


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

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = ckpt.get("config", {})

    # Infer embedding-table size from checkpoint to avoid shape mismatch when config changed.
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

    # strict=False does not ignore same-key shape mismatch, so pre-filter by shape.
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
    
    # 默认冻结（方案 A 会在外部再选择性打开聚合器梯度）
    model.eval().requires_grad_(False)
    return model


def load_skipped_ids_txt(path: Path) -> set[str]:
    """Read skipped ids txt (one id per line). Missing file -> empty set."""
    if not path.is_file():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            ids.add(s)
    return ids


def filter_cif_dataset_by_skipped_ids(dataset, skipped_ids: set[str], split_name: str):
    """Filter CIFData.id_prop_data in-place by skipped ids."""
    if not skipped_ids:
        return 0
    if not hasattr(dataset, "id_prop_data"):
        return 0

    original = len(dataset.id_prop_data)
    filtered = []
    removed = 0
    for rec in dataset.id_prop_data:
        rid = rec.get("material_id", rec.get("id", None))
        rid = str(rid) if rid is not None else None
        if rid is not None and rid in skipped_ids:
            removed += 1
            continue
        filtered.append(rec)
    dataset.id_prop_data = filtered
    # Clear cache because CIFData.__getitem__ is lru_cached by index.
    if hasattr(dataset.__getitem__, "cache_clear"):
        dataset.__getitem__.cache_clear()
    print(f"[Info] {split_name}: filtered {removed}/{original} samples by skipped_ids before training.")
    return removed


def prepare_batch(
    batch_dict,
    batch_ids,
    device,
    pretrained_vqvae=None,
    gpt_indices: dict | None = None,
    *,
    skip_missing_gpt_indices: bool = False,
    use_gpt_precomputed_vq_vectors: bool = False,
    property_cond_enabled: bool = False,
    property_bin_size: float = 0.5,
    property_bin_origin: float = 0.0,
    property_max_bins: int = 128,
):
    """
    将 collate_pool 的输出转换为 CondDiffusion.forward 所需的 Batch。
    1) 将 pyg_data 列表打包成 torch_geometric Batch
    2) 如果提供预训练 VQVAE，则计算 global/local VQ indices，并写入 Batch 属性
    """
    pyg_list = batch_dict.get("pyg_data", None)
    if pyg_list is None or len(pyg_list) == 0:
        raise ValueError("cond_guide=True 时必须提供 pyg_data（在 CIFData(cond_guide=True) 下生成）")

    batch = Batch.from_data_list(pyg_list).to(device)

    # 计算 VQ 条件（连续向量）作为条件：
    #   - 若 gpt_indices 提供且包含 z_g/z_l，且 use_gpt_precomputed_vq_vectors=True：直接写入 batch.global_vq_vectors/local_vq_vectors
    #   - 否则写入 indices，然后在 CondDiffusion 内部 indices -> continuous vectors
    if gpt_indices is not None:
        # 根据 batch_ids（一般是 material_id）逐个样本查表
        natoms_list = batch.natoms.detach().cpu().long().tolist()
        offsets = np.cumsum([0] + natoms_list)
        indices_g_rows = []
        indices_l_rows = []
        vq_vector_g_rows = []
        vq_vector_l_rows = []
        property_bins = []

        for i, bid in enumerate(batch_ids):
            rec = gpt_indices.get(str(bid))
            if rec is None:
                if skip_missing_gpt_indices:
                    # None 表示该 batch 无法构造 GPT 条件，训练循环将跳过该 batch
                    return None
                raise KeyError(f"gpt_indices not found for id={bid}")
            ig = torch.as_tensor(rec["indices_g"], dtype=torch.long, device=device)  # (1, G) or (G,)
            il = torch.as_tensor(rec["indices_l"], dtype=torch.long, device=device)  # (S, L) or (S,)
            if ig.dim() == 1:
                ig = ig.unsqueeze(0)
            if il.dim() == 1:
                il = il.unsqueeze(1)

            n_atoms_i = natoms_list[i]
            # 对局部 indices_l 按原子数对齐：截断或用 0 padding
            if il.size(0) >= n_atoms_i:
                il_i = il[:n_atoms_i]
            else:
                pad = torch.zeros((n_atoms_i - il.size(0), il.size(1)), dtype=torch.long, device=device)
                il_i = torch.cat([il, pad], dim=0)

            indices_g_rows.append(ig.squeeze(0))  # (G,)
            indices_l_rows.append(il_i)           # (n_atoms_i, L)

            if property_cond_enabled:
                pbin = rec.get("property_bin", None)
                if pbin is None:
                    pbin = -1
                try:
                    pbin = int(pbin)
                except Exception:
                    pbin = -1
                if pbin >= property_max_bins:
                    pbin = property_max_bins - 1
                property_bins.append(pbin)

            if use_gpt_precomputed_vq_vectors and ("z_g" in rec) and ("z_l" in rec):
                zg = torch.as_tensor(rec["z_g"], dtype=torch.float32, device=device)  # (1, D) or (D,)
                zl = torch.as_tensor(rec["z_l"], dtype=torch.float32, device=device)  # (S, D) or (S, )
                if zg.dim() == 1:
                    zg = zg.unsqueeze(0)
                # 对齐局部向量长度：截断或用 0 padding
                if zl.dim() == 1:
                    zl = zl.unsqueeze(0)
                if zl.size(0) >= n_atoms_i:
                    zl_i = zl[:n_atoms_i]
                else:
                    pad = torch.zeros((n_atoms_i - zl.size(0), zl.size(1)), dtype=torch.float32, device=device)
                    zl_i = torch.cat([zl, pad], dim=0)

                vq_vector_g_rows.append(zg.squeeze(0))  # (D,)
                vq_vector_l_rows.append(zl_i)           # (n_atoms_i, D)

        batch.global_vq_indices = torch.stack(indices_g_rows, dim=0)  # (B, G)
        batch.local_vq_indices = torch.cat(indices_l_rows, dim=0)    # (total_atoms, L)

        if property_cond_enabled:
            if len(property_bins) != len(indices_g_rows):
                property_bins = [-1] * len(indices_g_rows)
            batch.property_bins = torch.tensor(property_bins, dtype=torch.long, device=device)

        if use_gpt_precomputed_vq_vectors and len(vq_vector_g_rows) == len(indices_g_rows):
            batch.global_vq_vectors = torch.stack(vq_vector_g_rows, dim=0)  # (B, D)
            batch.local_vq_vectors = torch.cat(vq_vector_l_rows, dim=0)    # (total_atoms, D)
    elif pretrained_vqvae is not None:
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
        if property_cond_enabled:
            raw_prop = batch_dict.get("property", None)
            if raw_prop is not None:
                vals = raw_prop.detach().cpu().float().view(-1)
                bins = torch.floor((vals - float(property_bin_origin)) / float(property_bin_size)).long()
                bins = torch.clamp(bins, min=0, max=max(property_max_bins - 1, 0)).to(device)
                batch.property_bins = bins
            else:
                batch.property_bins = torch.full((batch.num_graphs,), -1, dtype=torch.long, device=device)
    else:
        batch.global_vq_indices = None
        batch.local_vq_indices = None
        if property_cond_enabled:
            batch.property_bins = torch.full((batch.num_graphs,), -1, dtype=torch.long, device=device)

    return batch


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    pretrained_vqvae=None,
    grad_clip=None,
    gpt_indices: dict | None = None,
    *,
    skip_missing_gpt_indices: bool = False,
    use_gpt_precomputed_vq_vectors: bool = False,
    property_cond_enabled: bool = False,
    property_bin_size: float = 0.5,
    property_bin_origin: float = 0.0,
    property_max_bins: int = 128,
    debug_property_bins_once: bool = False,
):
    """
    单个 epoch 的训练循环。
    监控 CondDiffusion.forward 返回的所有 loss 组件和关键中间量（以标量形式）。

    返回
    ----
    avg_loss : float
        平均总损失
    metrics : dict
        包含各子损失和噪声重建误差的平均值
    """
    model.train()
    if pretrained_vqvae is not None:
        pretrained_vqvae.eval()

    total_loss = 0.0
    # 累积各个 loss 分量
    total_vb_loss = 0.0
    total_ce_loss = 0.0
    total_lattice_mse = 0.0
    total_coords_mse = 0.0
    total_property_loss = 0.0
    property_loss_count = 0

    num_batches = 0
    pbar = tqdm(loader, desc="Train")

    skipped_batches = 0
    for batch_dict, batch_ids in pbar:
        batch = prepare_batch(
            batch_dict,
            batch_ids,
            device,
            pretrained_vqvae,
            gpt_indices=gpt_indices,
            skip_missing_gpt_indices=skip_missing_gpt_indices,
            use_gpt_precomputed_vq_vectors=use_gpt_precomputed_vq_vectors,
            property_cond_enabled=property_cond_enabled,
            property_bin_size=property_bin_size,
            property_bin_origin=property_bin_origin,
            property_max_bins=property_max_bins,
        )
        if batch is None:
            skipped_batches += 1
            continue

        if debug_property_bins_once and property_cond_enabled and num_batches == 0 and hasattr(batch, "property_bins"):
            raw_prop = batch_dict.get("property", None)
            if raw_prop is not None:
                raw_prop = raw_prop.detach().cpu().float().view(-1)
                print(
                    "[PropertyDebug][Train] "
                    f"raw_min={raw_prop.min().item():.4f}, raw_max={raw_prop.max().item():.4f}, "
                    f"bin_min={int(batch.property_bins.min().item())}, bin_max={int(batch.property_bins.max().item())}, "
                    f"unique_bins={torch.unique(batch.property_bins).detach().cpu().tolist()}"
                )

        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["loss"]

        # 反向传播
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # 统计损失
        vb_loss = outputs.get("vb_loss_atom_types", None)
        ce_loss = outputs.get("ce_loss_atom_types", None)
        property_loss = outputs.get("property_loss", None)

        # 使用 forward 返回的 true/pred noise 计算 MSE，用于监控
        lattice_mse = None
        coords_mse = None
        if "pred_noise_lattice" in outputs and "true_noise_lattice" in outputs:
            lattice_mse = F.mse_loss(
                outputs["pred_noise_lattice"],
                outputs["true_noise_lattice"],
            )
        if "pred_noise_coords" in outputs and "true_noise_coords" in outputs:
            coords_mse = F.mse_loss(
                outputs["pred_noise_coords"],
                outputs["true_noise_coords"],
            )

        total_loss += loss.item()
        if vb_loss is not None:
            total_vb_loss += vb_loss.item()
        if ce_loss is not None:
            total_ce_loss += ce_loss.item()
        if lattice_mse is not None:
            total_lattice_mse += lattice_mse.item()
        if coords_mse is not None:
            total_coords_mse += coords_mse.item()
        if property_loss is not None:
            total_property_loss += float(property_loss.item())
            property_loss_count += 1
        num_batches += 1

        # 在进度条上显示当前 batch 的各项 loss
        postfix = {"loss": f"{loss.item():.4f}"}
        if vb_loss is not None:
            postfix["vb"] = f"{vb_loss.item():.4f}"
        if ce_loss is not None:
            postfix["ce"] = f"{ce_loss.item():.4f}"
        if lattice_mse is not None:
            postfix["lat_mse"] = f"{lattice_mse.item():.4f}"
        if coords_mse is not None:
            postfix["coord_mse"] = f"{coords_mse.item():.4f}"
        if property_loss is not None:
            postfix["prop_loss"] = f"{property_loss.item():.4f}"
        pbar.set_postfix(**postfix)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = {
        "vb_loss_atom_types": total_vb_loss / max(num_batches, 1),
        "ce_loss_atom_types": total_ce_loss / max(num_batches, 1),
        "lattice_mse": total_lattice_mse / max(num_batches, 1),
        "coords_mse": total_coords_mse / max(num_batches, 1),
        "property_loss": total_property_loss / max(property_loss_count, 1),
        "skipped_batches": float(skipped_batches),
    }
    return avg_loss, metrics


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    pretrained_vqvae=None,
    gpt_indices: dict | None = None,
    *,
    skip_missing_gpt_indices: bool = False,
    use_gpt_precomputed_vq_vectors: bool = False,
    property_cond_enabled: bool = False,
    property_bin_size: float = 0.5,
    property_bin_origin: float = 0.0,
    property_max_bins: int = 128,
    debug_property_bins_once: bool = False,
):
    """
    验证循环。

    返回
    ----
    avg_loss : float
        平均总损失
    metrics : dict
        各子损失和噪声重建误差的平均值（与 train_epoch 对齐）
    """
    model.eval()
    if pretrained_vqvae is not None:
        pretrained_vqvae.eval()

    total_loss = 0.0
    total_vb_loss = 0.0
    total_ce_loss = 0.0
    total_lattice_mse = 0.0
    total_coords_mse = 0.0
    total_property_loss = 0.0
    property_loss_count = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Val")

    skipped_batches = 0
    for batch_dict, batch_ids in pbar:
        batch = prepare_batch(
            batch_dict,
            batch_ids,
            device,
            pretrained_vqvae,
            gpt_indices=gpt_indices,
            skip_missing_gpt_indices=skip_missing_gpt_indices,
            use_gpt_precomputed_vq_vectors=use_gpt_precomputed_vq_vectors,
            property_cond_enabled=property_cond_enabled,
            property_bin_size=property_bin_size,
            property_bin_origin=property_bin_origin,
            property_max_bins=property_max_bins,
        )
        if batch is None:
            skipped_batches += 1
            continue

        if debug_property_bins_once and property_cond_enabled and num_batches == 0 and hasattr(batch, "property_bins"):
            raw_prop = batch_dict.get("property", None)
            if raw_prop is not None:
                raw_prop = raw_prop.detach().cpu().float().view(-1)
                print(
                    "[PropertyDebug][Val] "
                    f"raw_min={raw_prop.min().item():.4f}, raw_max={raw_prop.max().item():.4f}, "
                    f"bin_min={int(batch.property_bins.min().item())}, bin_max={int(batch.property_bins.max().item())}, "
                    f"unique_bins={torch.unique(batch.property_bins).detach().cpu().tolist()}"
                )
        outputs = model(batch)
        loss = outputs["loss"]

        vb_loss = outputs.get("vb_loss_atom_types", None)
        ce_loss = outputs.get("ce_loss_atom_types", None)
        property_loss = outputs.get("property_loss", None)
        lattice_mse = None
        coords_mse = None
        if "pred_noise_lattice" in outputs and "true_noise_lattice" in outputs:
            lattice_mse = F.mse_loss(
                outputs["pred_noise_lattice"],
                outputs["true_noise_lattice"],
            )
        if "pred_noise_coords" in outputs and "true_noise_coords" in outputs:
            coords_mse = F.mse_loss(
                outputs["pred_noise_coords"],
                outputs["true_noise_coords"],
            )

        total_loss += loss.item()
        if vb_loss is not None:
            total_vb_loss += vb_loss.item()
        if ce_loss is not None:
            total_ce_loss += ce_loss.item()
        if lattice_mse is not None:
            total_lattice_mse += lattice_mse.item()
        if coords_mse is not None:
            total_coords_mse += coords_mse.item()
        if property_loss is not None:
            total_property_loss += float(property_loss.item())
            property_loss_count += 1
        num_batches += 1

        postfix = {"loss": f"{loss.item():.4f}"}
        if vb_loss is not None:
            postfix["vb"] = f"{vb_loss.item():.4f}"
        if ce_loss is not None:
            postfix["ce"] = f"{ce_loss.item():.4f}"
        if lattice_mse is not None:
            postfix["lat_mse"] = f"{lattice_mse.item():.4f}"
        if coords_mse is not None:
            postfix["coord_mse"] = f"{coords_mse.item():.4f}"
        if property_loss is not None:
            postfix["prop_loss"] = f"{property_loss.item():.4f}"
        pbar.set_postfix(**postfix)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = {
        "vb_loss_atom_types": total_vb_loss / max(num_batches, 1),
        "ce_loss_atom_types": total_ce_loss / max(num_batches, 1),
        "lattice_mse": total_lattice_mse / max(num_batches, 1),
        "coords_mse": total_coords_mse / max(num_batches, 1),
        "property_loss": total_property_loss / max(property_loss_count, 1),
        "skipped_batches": float(skipped_batches),
    }
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion (single GPU)")
    parser.add_argument("--config", type=str, default="config/train_cond_diff.yaml")
    parser.add_argument("--data_path", type=str, default=None, help="override data path")
    # OC20 support removed: --oc20_src argument deleted
    parser.add_argument("--pretrained_vqvae_path", type=str, default=None, help="path to pretrained CrystalVQVAE checkpoint (.pth)")
    parser.add_argument("--pretrained_vqvae_config", type=str, default=None, help="path to VQVAE training config yaml (e.g. config/data/train_cryvqvae_mpts_52.yaml)")
    parser.add_argument(
        "--gpt_indices_path",
        type=str,
        default=None,
        help="Path to pkl(.gz) produced by scripts/gpt_tokens_to_vq_indices.py; "
             "if set, use these offline GPT-generated VQ indices as conditions.",
    )
    parser.add_argument(
        "--gpt_indices_path_train",
        type=str,
        default=None,
        help="(Optional) GPT indices for training split. If not set, fallback to --gpt_indices_path.",
    )
    parser.add_argument(
        "--gpt_indices_path_val",
        type=str,
        default=None,
        help="(Optional) GPT indices for validation split. If not set, fallback to --gpt_indices_path.",
    )
    parser.add_argument(
        "--gpt_indices_path_test",
        type=str,
        default=None,
        help="(Optional) GPT indices for test split. This script currently trains with train+val only, "
             "but the option is kept for future test evaluation/sampling hooks.",
    )
    parser.add_argument(
        "--skip_missing_gpt_indices",
        action="store_true",
        help="If set and --gpt_indices_path is provided, skip batches that contain ids not found in the GPT indices table.",
    )
    parser.add_argument("--save_dir", type=str, default="ckpts/mp_20/cond_diffusion/")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.data_path is not None:
        config["data_path"] = args.data_path
    if args.pretrained_vqvae_path is not None:
        config["pretrained_vqvae_path"] = args.pretrained_vqvae_path
    if args.pretrained_vqvae_config is not None:
        config["pretrained_vqvae_config"] = args.pretrained_vqvae_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("random_seed", 42))

    use_gpt_precomputed_vq_vectors = bool(config.get("use_gpt_precomputed_vq_vectors", False))
    filter_skipped_ids_before_training = bool(config.get("filter_skipped_ids_before_training", True))

    # CIF CSV 模式（OC20 support removed; always use CIF CSV mode）
    data_root = config.get("data_path", "dataset/mp_20")
    use_single_csv = config.get("use_single_csv", False)
    train_csv = config.get("train_csv", "train.csv")
    val_csv = config.get("val_csv", "val.csv")
    test_csv = config.get("test_csv", "test.csv")
    return_test = config.get("return_test", False)

    if use_single_csv:
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
        train_loader, val_loader = get_train_val_test_loader(
            full_dataset,
            collate_fn=collate_pool,
            batch_size=config["batch_size"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            return_test=False,
            num_workers=config.get("workers", 4),
            pin_memory=True,
        )

        print(
            f"Dataset size (single CSV): {len(full_dataset)}\n"
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )
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
            num_workers=config.get("workers", 4),
            pin_memory=config.get("pin_memory", True),
            random_seed=config.get("random_seed", 42),
            cond_guide=config.get("cond_guide", True),
            return_test=return_test,
        )

        if return_test:
            train_loader, val_loader, test_loader = get_separate_loaders(**loader_kwargs)
        else:
            train_loader, val_loader = get_separate_loaders(**loader_kwargs)
            test_loader = None

        # Optional: pre-filter invalid ids from DSS2indices/*_skipped_ids.txt before training,
        # instead of skipping batches during train loop.
        if filter_skipped_ids_before_training:
            dss2_dir = config.get("gpt_indices_dir", str(Path(data_root) / "DSS2indices"))
            dss2_dir = Path(dss2_dir)
            train_skipped = load_skipped_ids_txt(dss2_dir / "train_skipped_ids.txt")
            val_skipped = load_skipped_ids_txt(dss2_dir / "val_skipped_ids.txt")
            test_skipped = load_skipped_ids_txt(dss2_dir / "test_skipped_ids.txt")

            if train_skipped:
                filter_cif_dataset_by_skipped_ids(train_loader.dataset, train_skipped, "train")
            if val_skipped:
                filter_cif_dataset_by_skipped_ids(val_loader.dataset, val_skipped, "val")
            if test_loader is not None and test_skipped:
                filter_cif_dataset_by_skipped_ids(test_loader.dataset, test_skipped, "test")

    # pretrained vqvae
    pretrained_vqvae = None
    if config.get("pretrained_vqvae_path"):
        vqvae_config_path = config.get("pretrained_vqvae_config", "")
        pretrained_vqvae = load_pretrained_vqvae(
            config["pretrained_vqvae_path"],
            vqvae_config_path,
            device,
        )

    def load_gpt_indices(indices_path: str) -> dict:
        """Load GPT VQ indices record list -> dict(id->rec) for fast lookup."""
        import gzip
        import pickle

        indices_path_p = Path(indices_path)
        assert indices_path_p.is_file(), f"gpt_indices_path not found: {indices_path_p}"
        opener = gzip.open if str(indices_path_p).endswith(".gz") else open
        with opener(indices_path_p, "rb") as f:
            records = pickle.load(f)
        gpt_dict = {}
        for rec in records:
            rid = rec.get("id", None)
            if rid is None:
                continue
            rid = str(rid)
            if not rid:
                continue
            gpt_dict[rid] = rec
        return gpt_dict

    # offline GPT indices (optional, split-aware)
    # 优先级：命令行参数 > 配置文件 > 旧参数（兼容）
    base_gpt_path = args.gpt_indices_path or config.get("gpt_indices_path")
    gpt_path_train = args.gpt_indices_path_train or config.get("gpt_indices_path_train")
    gpt_path_val = args.gpt_indices_path_val or config.get("gpt_indices_path_val")
    gpt_path_test = args.gpt_indices_path_test or config.get("gpt_indices_path_test")

    gpt_indices_train = load_gpt_indices(gpt_path_train) if gpt_path_train else (load_gpt_indices(base_gpt_path) if base_gpt_path else None)
    gpt_indices_val = load_gpt_indices(gpt_path_val) if gpt_path_val else (load_gpt_indices(base_gpt_path) if base_gpt_path else None)
    gpt_indices_test = load_gpt_indices(gpt_path_test) if gpt_path_test else (load_gpt_indices(base_gpt_path) if base_gpt_path else None)

    if gpt_indices_train is not None:
        print(f"[Info] Loaded GPT VQ indices (train) for {len(gpt_indices_train)} samples.")
    if gpt_indices_val is not None:
        print(f"[Info] Loaded GPT VQ indices (val)   for {len(gpt_indices_val)} samples.")
    if gpt_indices_test is not None:
        print(f"[Info] Loaded GPT VQ indices (test)  for {len(gpt_indices_test)} samples.")

    # model
    model = CondDiffusion(config, pretrained_vqvae_model=pretrained_vqvae).to(device)

    # ===================== VQ-VAE 联合微调（可选） =====================
    # 支持两类聚合：
    # 1) 外部聚合器：global_aggregator / local_aggregator
    # 2) 内部聚合器：residual_attn 的 vq_global.gates / vq_local.gates
    # 默认内部聚合微调关闭。
    finetune_vqvae_outer_aggregator = as_bool(
        config.get("finetune_vqvae_outer_aggregator", config.get("finetune_vqvae_aggregator", False)),
        default=False,
    )
    finetune_vqvae_inner_aggregator = as_bool(config.get("finetune_vqvae_inner_aggregator", False), default=False)

    base_lr = float(config.get("learning_rate", 1e-4))
    vqvae_outer_agg_lr = float(
        config.get("vqvae_outer_agg_lr", config.get("vqvae_agg_lr", base_lr * 0.1))
    )
    vqvae_inner_agg_lr = float(config.get("vqvae_inner_agg_lr", base_lr * 0.05))

    vqvae_outer_agg_params = []
    vqvae_inner_agg_params = []

    if pretrained_vqvae is not None and (finetune_vqvae_outer_aggregator or finetune_vqvae_inner_aggregator):
        # 先冻结全部 VQ-VAE 参数，再按需打开
        pretrained_vqvae.requires_grad_(False)

        if finetune_vqvae_outer_aggregator:
            if hasattr(pretrained_vqvae, "global_aggregator") and pretrained_vqvae.global_aggregator is not None:
                pretrained_vqvae.global_aggregator.requires_grad_(True)
                vqvae_outer_agg_params += list(pretrained_vqvae.global_aggregator.parameters())
            if hasattr(pretrained_vqvae, "local_aggregator") and pretrained_vqvae.local_aggregator is not None:
                pretrained_vqvae.local_aggregator.requires_grad_(True)
                vqvae_outer_agg_params += list(pretrained_vqvae.local_aggregator.parameters())

        if finetune_vqvae_inner_aggregator:
            for vq_attr in ("vq_global", "vq_local"):
                vq_mod = getattr(pretrained_vqvae, vq_attr, None)
                if vq_mod is None:
                    continue
                gates = getattr(vq_mod, "gates", None)
                if gates is None:
                    continue
                gates.requires_grad_(True)
                vqvae_inner_agg_params += list(gates.parameters())

        if finetune_vqvae_outer_aggregator and len(vqvae_outer_agg_params) == 0:
            print("Warning: finetune_vqvae_outer_aggregator=True but no outer aggregators found.")
        if finetune_vqvae_inner_aggregator and len(vqvae_inner_agg_params) == 0:
            print("Warning: finetune_vqvae_inner_aggregator=True but no internal gates found.")

        if len(vqvae_outer_agg_params) > 0:
            print(f"✓ Finetuning VQ-VAE outer aggregators with lr={vqvae_outer_agg_lr:.2e}")
        if len(vqvae_inner_agg_params) > 0:
            print(f"✓ Finetuning VQ-VAE inner aggregators (gates) with lr={vqvae_inner_agg_lr:.2e}")

    # optimizer（将 VQ-VAE 聚合器作为独立参数组，便于设置独立 lr）
    weight_decay = float(config.get("weight_decay", 1e-4))

    # 只优化 requires_grad=True 的参数，避免将冻结参数传入 optimizer
    # base params：除 VQ-VAE 聚合器以外的所有可训练参数
    vqvae_outer_ids = set(id(p) for p in vqvae_outer_agg_params)
    vqvae_inner_ids = set(id(p) for p in vqvae_inner_agg_params)
    excluded_ids = vqvae_outer_ids | vqvae_inner_ids
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in excluded_ids]

    param_groups = [
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
    ]
    if len(vqvae_outer_agg_params) > 0:
        param_groups.append({"params": vqvae_outer_agg_params, "lr": vqvae_outer_agg_lr, "weight_decay": weight_decay})
    if len(vqvae_inner_agg_params) > 0:
        param_groups.append({"params": vqvae_inner_agg_params, "lr": vqvae_inner_agg_lr, "weight_decay": weight_decay})

    num_base = sum(p.numel() for p in base_params)
    num_outer = sum(p.numel() for p in vqvae_outer_agg_params)
    num_inner = sum(p.numel() for p in vqvae_inner_agg_params)
    print(f"[Opt] base params: {num_base:,} (lr={base_lr:.2e})")
    if num_outer > 0:
        print(f"[Opt] outer aggregator params: {num_outer:,} (lr={vqvae_outer_agg_lr:.2e})")
    if num_inner > 0:
        print(f"[Opt] inner aggregator params: {num_inner:,} (lr={vqvae_inner_agg_lr:.2e})")

    optimizer = optim.AdamW(param_groups)
    
    # scheduler
    scheduler_type = config.get("scheduler", "cosine").lower()
    if scheduler_type == "cosine":
        T_max = config.get("cosine_T_max")
        if T_max is None:
            T_max = config.get("num_epochs", 50)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.get("min_lr", 1e-6),
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("plateau_mode", "min"),
            factor=config.get("plateau_factor", 0.5),
            patience=config.get("plateau_patience", 10),
            threshold=config.get("plateau_threshold", 1e-4),
            cooldown=config.get("plateau_cooldown", 0),
            min_lr=config.get("min_lr", 1e-6),
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("step_gamma", 0.1),
        )
    elif scheduler_type == "constant":
        scheduler = None  # 不使用调度器
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # early stopping
    use_early_stopping = config.get("early_stopping", False)
    if use_early_stopping:
        early_stopping_patience = config.get("early_stopping_patience", 20)
        early_stopping_min_delta = config.get("early_stopping_min_delta", 0.0)
        early_stopping_mode = config.get("early_stopping_mode", "min")
        best_val_loss = float("inf") if early_stopping_mode == "min" else float("-inf")
        patience_counter = 0
    else:
        early_stopping_patience = None
        best_val_loss = float("inf")
        patience_counter = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "cond_diffusion"),
            name=wandb_config.get("run_name", None),
            config=config,
            dir=str(save_dir),
        )
        print(f"✓ Wandb initialized: project={wandb_config.get('project', 'cond_diffusion')}, run={wandb_config.get('run_name', wandb.run.name)}")

    for epoch in range(config.get("num_epochs", 50)):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.get('num_epochs', 50)}")
        print(f"{'='*60}")
        
        # 训练一个 epoch，返回总损失和各子损失 / 误差
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pretrained_vqvae=pretrained_vqvae,
            grad_clip=config.get("grad_clip", None),
            gpt_indices=gpt_indices_train,
            skip_missing_gpt_indices=args.skip_missing_gpt_indices,
            use_gpt_precomputed_vq_vectors=use_gpt_precomputed_vq_vectors,
            property_cond_enabled=bool(config.get("property_cond_enabled", False)),
            property_bin_size=float(config.get("property_bin_size", 0.5)),
            property_bin_origin=float(config.get("property_bin_origin", 0.0)),
            property_max_bins=int(config.get("property_vocab_size", config.get("property_max_bins", 128))),
            debug_property_bins_once=bool(config.get("debug_property_bins_once", False)),
        )

        if val_loader is not None:
            val_loss, val_metrics = validate(
                model,
                val_loader,
                device,
                pretrained_vqvae=pretrained_vqvae,
                gpt_indices=gpt_indices_val,
                skip_missing_gpt_indices=args.skip_missing_gpt_indices,
                use_gpt_precomputed_vq_vectors=use_gpt_precomputed_vq_vectors,
                property_cond_enabled=bool(config.get("property_cond_enabled", False)),
                property_bin_size=float(config.get("property_bin_size", 0.5)),
                property_bin_origin=float(config.get("property_bin_origin", 0.0)),
                property_max_bins=int(config.get("property_vocab_size", config.get("property_max_bins", 128))),
                debug_property_bins_once=bool(config.get("debug_property_bins_once", False)),
            )
        else:
            val_loss, val_metrics = train_loss, train_metrics

        # 更新学习率调度器
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current LR: {current_lr:.2e}")

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # 打印子损失和噪声重建误差（epoch 级别）
        print("  Train metrics:", ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()]))
        print("  Val   metrics:", ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

        # 记录到 wandb
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            # 将所有子损失 / 误差记录到 wandb，便于监控
            for k, v in train_metrics.items():
                log_dict[f"train_{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val_{k}"] = v
            if scheduler is not None:
                log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
            if use_early_stopping:
                log_dict["early_stopping_patience"] = patience_counter
            wandb.log(log_dict)

        # Early stopping 和保存最佳模型
        is_best = False
        if use_early_stopping:
            if early_stopping_mode == "min":
                improved = val_loss < (best_val_loss - early_stopping_min_delta)
            else:
                improved = val_loss > (best_val_loss + early_stopping_min_delta)
            
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered! Best validation loss: {best_val_loss:.4f}")
                    break
        else:
            # 不使用 early stopping，只保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True

        # 保存最佳模型
        if is_best:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "config": config,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            torch.save(ckpt, save_dir / "best_diffusion.pth")
            print(f"✓ Saved best model to {save_dir / 'best_diffusion.pth'}")
        
        # 定期保存检查点（save_every <=0 时不保存，避免除零）
        save_every = config.get("save_every", 10)
        if save_every and save_every > 0 and (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "config": config,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            torch.save(ckpt, save_dir / f"checkpoint_epoch_{epoch + 1}.pth")
            print(f"Saved checkpoint to {save_dir / f'checkpoint_epoch_{epoch + 1}.pth'}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    
    # 结束 wandb 运行
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # 无属性训练
    # CUDA_VISIBLE_DEVICES=0 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --save_dir ckpts/mp_20/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=2 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_perov_5.yaml --save_dir ckpts/perov_5/ablation/RVQ/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=0 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_carbon_24.yaml --save_dir ckpts/carbon_24/ablation/RVQ/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=0 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --save_dir ckpts/mpts_52/ablation/RVQ/cond_diffusion --wandb

    # 带属性训练
    # CUDA_VISIBLE_DEVICES=1 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --save_dir ckpts/mp_20/cond_diffusion_bandgap --wandb
    # CUDA_VISIBLE_DEVICES=2 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --save_dir ckpts/mp_20/cond_diffusion_fe --wandb

    # 消融实验，对attn-RVQ消融，也就是单层的attn-RVQ，也就是VQ
    # CUDA_VISIBLE_DEVICES=4 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --save_dir ckpts/mp_20/ablation/VQ/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=6 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --save_dir ckpts/mpts_52/ablation/VQ/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=4 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_perov_5.yaml --save_dir ckpts/perov_5/ablation/VQ/cond_diffusion --wandb

    # 消融实验，对attn进行消融，也就是RVQ
    # CUDA_VISIBLE_DEVICES=7 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mp_20.yaml --save_dir ckpts/mp_20/ablation/RVQ/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=3 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_mpts_52.yaml --save_dir ckpts/mpts_52/ablation/RVQ/cond_diffusion --wandb
    # CUDA_VISIBLE_DEVICES=3 python train_cond_diff.py --config config/cond_diffusion/train_cond_diffusion_perov_5.yaml --save_dir ckpts
    main()
