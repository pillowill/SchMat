"""
训练 CrystalVQVAE 模型的脚本（支持多GPU训练）

架构：
    Structure → CSPNet → g, h → VQ → z_g, z_l → Decoders → ĝ, ĥ + aux tasks
"""
import argparse
import os
import platform
import random
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
from data.data import CIFData, collate_pool, get_train_val_test_loader

from model.crystal_vq_vae import CrystalVQVAE

warnings.filterwarnings("ignore")



def set_seed(seed):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_batch(batch_dict, device):
    """
    将 collate_pool 输出的 batch_dict 转换为 CrystalVQVAE 需要的张量形式。

    适配当前 data.py 中的 CIFData + collate_pool：
    - batch_dict['pyg_data']: List[torch_geometric.data.Data]
        每个 Data 至少包含：
        - atom_types : (n_i,)
        - frac_coords : (n_i, 3)
        - lattices : (1, 3, 3)
        - natoms : (1,)
    - batch_dict['global_vq_targets'] : (B,)  由 CIFData.calculate_global_vq_target 给出
    - batch_dict['local_vq_targets']  : List[List[dict]]  由 CIFData.calculate_local_vq_target 给出
    """
    pyg_list = batch_dict['pyg_data']              # List[Data]
    batch_size = len(pyg_list)

    atom_types_list = []
    frac_coords_list = []
    lattice_list = []
    num_atoms_list = []
    node2batch = []

    for i, data in enumerate(pyg_list):
        # 原子类型（原子序数）
        atom_types = data.atom_types.to(device).long()          # (n_i,)
        # 分数坐标
        frac_coords = data.frac_coords.to(device).float()       # (n_i, 3)
        # 晶格矩阵：data.lattices 形状为 (1, 3, 3)
        lattice = data.lattices.to(device).float()[0]           # (3, 3)
        # 原子数
        if hasattr(data, "natoms"):
            n_atoms = int(data.natoms.item())
        else:
            n_atoms = atom_types.shape[0]

        atom_types_list.append(atom_types)
        frac_coords_list.append(frac_coords)
        lattice_list.append(lattice)
        num_atoms_list.append(n_atoms)
        node2batch.extend([i] * n_atoms)

    atom_types_flat = torch.cat(atom_types_list, dim=0)                    # (total_atoms,)
    frac_coords_flat = torch.cat(frac_coords_list, dim=0)                  # (total_atoms, 3)
    lattice_matrices = torch.stack(lattice_list, dim=0)                    # (B, 3, 3)
    num_atoms = torch.tensor(num_atoms_list, dtype=torch.long, device=device)  # (B,)
    node2batch = torch.tensor(node2batch, dtype=torch.long, device=device)     # (total_atoms,)

    # 目标：来自 CIFData.calculate_global_vq_target / calculate_local_vq_target
    global_targets = batch_dict['global_vq_targets'].to(device)            # (B,)
    local_targets = batch_dict['local_vq_targets']                          # List[List[dict]]

    return {
        'atom_types': atom_types_flat,
        'frac_coords': frac_coords_flat,
        'lattice': lattice_matrices,
        'num_atoms': num_atoms,
        'node2batch': node2batch,
        'global_targets': global_targets,
        'local_targets': local_targets,
    }


def compute_codebook_usage(indices, codebook_size, device, num_quantizers=1):
    """
    计算 codebook 使用率（支持 ResidualVQ 的多个量化器）
    
    Parameters
    ----------
    indices : Tensor
        codebook 索引 (batch_size, num_quantizers) 或 (total_atoms, num_quantizers)
        如果是旧格式 (batch_size,) 或 (total_atoms,)，会自动处理
    codebook_size : int
        codebook 大小
    device : torch.device
        设备
    num_quantizers : int
        量化器数量（默认1，用于向后兼容）
    
    Returns
    -------
    dict
        - usage_rate: float, 使用率 (0-1)
        - num_used: int, 使用的 codebook 向量数量
        - usage_hist: Tensor, 每个 codebook 向量的使用次数
        - per_quantizer_usage: list, 每个量化器的使用率（如果 num_quantizers > 1）
    """
    if len(indices) == 0:
        return {
            'usage_rate': 0.0,
            'num_used': 0,
            'usage_hist': torch.zeros(codebook_size, device=device),
            'per_quantizer_usage': []
        }
    
    # 处理索引形状
    if indices.dim() == 1:
        # 旧格式：扩展为 (batch_size, 1) 或 (total_atoms, 1)
        indices = indices.unsqueeze(-1)
        num_quantizers = 1
    
    # 展平所有量化器的索引
    all_indices = indices.flatten()  # (batch_size * num_quantizers,)
    
    # 计算每个 codebook 向量的使用次数（所有量化器合并统计）
    usage_hist = torch.bincount(all_indices, minlength=codebook_size)
    
    # 计算使用率
    num_used = (usage_hist > 0).sum().item()
    usage_rate = num_used / codebook_size
    
    # 计算每个量化器的使用率
    per_quantizer_usage = []
    if num_quantizers > 1:
        for q_idx in range(num_quantizers):
            q_indices = indices[:, q_idx].flatten()
            q_usage_hist = torch.bincount(q_indices, minlength=codebook_size)
            q_num_used = (q_usage_hist > 0).sum().item()
            q_usage_rate = q_num_used / codebook_size
            per_quantizer_usage.append({
                'quantizer': q_idx,
                'usage_rate': q_usage_rate,
                'num_used': q_num_used
            })
    
    return {
        'usage_rate': usage_rate,
        'num_used': num_used,
        'usage_hist': usage_hist,
        'per_quantizer_usage': per_quantizer_usage
    }


def print_codebook_stats(model, prefix: str = "", extra_stats: dict | None = None):
    """
    打印 Global / Local codebook 的均值和方差（支持 ResidualVQ）

    Parameters
    ----------
    model : nn.Module 或 DDP 包装的模型
        需要包含 vq_global 和 vq_local 成员
    prefix : str
        打印前缀，方便对齐输出
    """
    try:
        # 兼容 DDP：取出内部的实际模型
        module = model.module if hasattr(model, "module") else model

        # 必须包含 vq_global / vq_local
        if not hasattr(module, "vq_global"):
            print(f"{prefix}Warning: model has no vq_global attribute")
            return
        if not hasattr(module, "vq_local"):
            print(f"{prefix}Warning: model has no vq_local attribute")
            return

        with torch.no_grad():
            # 尝试获取 codebook（支持 ResidualVQ）
            # ResidualVQ 使用 quantizers 列表存储多个量化器
            # ResidualVQ：使用 layers 列表访问每个量化器的 codebook
            # VectorQuantize：直接使用 _codebook.embed
            def get_codebook_tensor(vq_module):
                """从 VQ 模块提取 codebook 张量"""
                # 尝试 ResidualVQ 方式（有 layers 列表）
                if hasattr(vq_module, "layers"):
                    # ResidualVQ：合并所有量化器的 codebook
                    codebooks = []
                    for layer in vq_module.layers:
                        if hasattr(layer, "_codebook") and hasattr(layer._codebook, "embed"):
                            codebooks.append(layer._codebook.embed)
                    if len(codebooks) > 0:
                        return torch.cat(codebooks, dim=0), len(codebooks)

                # 尝试 VectorQuantize 方式
                if hasattr(vq_module, "_codebook") and hasattr(vq_module._codebook, "embed"):
                    return vq_module._codebook.embed, 1

                # 尝试直接 codebook 属性
                if hasattr(vq_module, "codebook"):
                    if isinstance(vq_module.codebook, torch.Tensor):
                        return vq_module.codebook, 1
                    elif hasattr(vq_module.codebook, "embed"):
                        return vq_module.codebook.embed, 1

                return None, 0
            global_cb, num_quantizers_g = get_codebook_tensor(module.vq_global)
            local_cb, num_quantizers_l = get_codebook_tensor(module.vq_local)

            if global_cb is None:
                print(f"{prefix}Warning: vq_global.codebook is None")
                return
            if local_cb is None:
                print(f"{prefix}Warning: vq_local.codebook is None")
                return

            # 确保 codebook 在正确的设备上
            if global_cb.device.type == 'cuda':
                global_cb = global_cb.cpu()
            if local_cb.device.type == 'cuda':
                local_cb = local_cb.cpu()

            # 展平成标量统计：整体均值 / 方差
            g_mean = global_cb.mean().item()
            g_std = global_cb.std().item()
            l_mean = local_cb.mean().item()
            l_std = local_cb.std().item()

        print(f"{prefix}Global codebook: mean = {g_mean:.4f}, std = {g_std:.4f} (quantizers: {num_quantizers_g})")
        print(f"{prefix}Local  codebook: mean = {l_mean:.4f}, std = {l_std:.4f} (quantizers: {num_quantizers_l})")
        # 额外统计（例如 encoder embedding / code indices 的均值方差），由训练循环传入
        if extra_stats:
            for k, v in extra_stats.items():
                if v is None:
                    continue
                try:
                    print(f"{prefix}{k}: {float(v):.6f}")
                except Exception:
                    print(f"{prefix}{k}: {v}")
    except Exception as e:
        print(f"{prefix}Error computing codebook statistics: {e}")
        import traceback
        traceback.print_exc()


def train_epoch(model, loader, optimizer, device, rank, grad_clip=None, 
                codebook_size=None, log_interval=30):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {
        'recon_global': 0.0,
        'recon_local': 0.0,
        'space_group': 0.0,
        'cn': 0.0,
        # 'neighbor': 0.0,
        'commitment': 0.0,
    }
    
    # 统计：encoder embedding / code indices 的均值与方差（epoch 级别，避免频繁打印）
    emb_sum = 0.0
    emb_sumsq = 0.0
    emb_count = 0
    idx_sum = 0.0
    idx_sumsq = 0.0
    idx_count = 0
    alpha_metric_sums = {}
    alpha_metric_counts = {}
    step = 0
    
    # 只在主进程显示进度条
    pbar = tqdm(loader, desc="Training") if rank == 0 else loader
    
    for batch_idx, batch in enumerate(pbar):
        batch_dict, _ = batch
        
        # 准备数据
        inputs = prepare_batch(batch_dict, device)
        
        # 前向传播
        optimizer.zero_grad()
        losses = model(
            inputs['atom_types'],
            inputs['frac_coords'],
            inputs['lattice'],
            inputs['num_atoms'],
            inputs['node2batch'],
            inputs['global_targets'],
            inputs['local_targets'],
        )
        
        # 累积统计：encoder embedding / code indices
        if 'emb_global' in losses and 'emb_local' in losses:
            g = losses['emb_global'].detach()
            h = losses['emb_local'].detach()
            emb_flat = torch.cat([g.reshape(-1), h.reshape(-1)], dim=0).float()
            emb_sum += emb_flat.sum().item()
            emb_sumsq += (emb_flat * emb_flat).sum().item()
            emb_count += int(emb_flat.numel())

        if 'indices_g' in losses and 'indices_l' in losses:
            ig = losses['indices_g'].detach().reshape(-1).float()
            il = losses['indices_l'].detach().reshape(-1).float()
            idx_flat = torch.cat([ig, il], dim=0)
            idx_sum += idx_flat.sum().item()
            idx_sumsq += (idx_flat * idx_flat).sum().item()
            idx_count += int(idx_flat.numel())

        # 累积 residual_attn 的每层 alpha 统计
        for k, v in losses.items():
            if not str(k).startswith("alpha_"):
                continue
            if isinstance(v, torch.Tensor):
                val = float(v.detach().float().mean().item())
            else:
                val = float(v)
            alpha_metric_sums[k] = alpha_metric_sums.get(k, 0.0) + val
            alpha_metric_counts[k] = alpha_metric_counts.get(k, 0) + 1
        
        # 反向传播
        loss = losses['total_loss']
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        step += 1
        
        # 统计
        total_loss += loss.item()
        for key in loss_components.keys():
            if f'{key}_loss' in losses:
                loss_components[key] += losses[f'{key}_loss'].item()
        
        # 不再进行 step 级别打印；epoch 级统计在函数返回后由 print_codebook_stats 统一输出
        
        # 更新进度条（只在主进程）
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon_g': f'{losses.get("recon_global_loss", 0):.4f}',
                'recon_l': f'{losses.get("recon_local_loss", 0):.4f}',
            })
    
    # 计算平均
    num_batches = len(loader)
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    # epoch 级统计（均值/标准差）
    stats = {}
    if emb_count > 0:
        emb_mean = emb_sum / emb_count
        emb_var = max(0.0, (emb_sumsq / emb_count) - emb_mean * emb_mean)
        stats["encoder_embedding_mean"] = emb_mean
        stats["encoder_embedding_std"] = emb_var ** 0.5
    if idx_count > 0:
        idx_mean = idx_sum / idx_count
        idx_var = max(0.0, (idx_sumsq / idx_count) - idx_mean * idx_mean)
        stats["code_index_mean"] = idx_mean
        stats["code_index_std"] = idx_var ** 0.5
    for k, s in alpha_metric_sums.items():
        c = max(1, alpha_metric_counts.get(k, 1))
        stats[k] = s / c

    return total_loss, loss_components, stats


@torch.no_grad()
def validate(model, loader, device, rank):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    loss_components = {
        'recon_global': 0.0,
        'recon_local': 0.0,
        'space_group': 0.0,
        'cn': 0.0,
        # 'neighbor': 0.0,
        'commitment': 0.0,
    }
    
    # 只在主进程显示进度条
    pbar = tqdm(loader, desc="Validation") if rank == 0 else loader
    
    for batch in pbar:
        batch_dict, _ = batch
        
        # 准备数据
        inputs = prepare_batch(batch_dict, device)
        
        # 前向传播
        losses = model(
            inputs['atom_types'],
            inputs['frac_coords'],
            inputs['lattice'],
            inputs['num_atoms'],
            inputs['node2batch'],
            inputs['global_targets'],
            inputs['local_targets'],
        )
        
        # 统计
        total_loss += losses['total_loss'].item()
        for key in loss_components.keys():
            if f'{key}_loss' in losses:
                loss_components[key] += losses[f'{key}_loss'].item()
        
        # 更新进度条（只在主进程）
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{losses["total_loss"].item():.4f}',
            })
    
    # 计算平均
    num_batches = len(loader)
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return total_loss, loss_components


def apply_aux_loss_warmup(model, epoch: int, config: dict):
    """
    Optional linear warmup for auxiliary losses (space_group / cn).
    This keeps reconstruction dominant in early epochs and avoids noisy plateaus.
    """
    warmup_epochs = int(config.get("aux_warmup_epochs", 0))
    start_factor = float(config.get("aux_warmup_start_factor", 1.0))

    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "loss_weights") or warmup_epochs <= 0:
        return 1.0

    # Cache base weights once.
    if not hasattr(module, "_base_loss_weights"):
        module._base_loss_weights = dict(module.loss_weights)

    if epoch >= warmup_epochs:
        factor = 1.0
    else:
        progress = epoch / max(1, warmup_epochs)
        factor = start_factor + (1.0 - start_factor) * progress

    for k in ("space_group", "cn"):
        if k in module._base_loss_weights:
            module.loss_weights[k] = module._base_loss_weights[k] * factor

    return factor


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train CrystalVQVAE (Multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_cryvqvae_mpts_52.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data path in config (CIF CSV mode)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 覆盖配置（仅对 CIF CSV 模式有效）
    if args.data_path is not None:
        config["data_path"] = args.data_path

    
    # 强制单进程/单GPU（或CPU）训练：不使用 DDP
    print("Starting single GPU training (DDP disabled)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(config["random_seed"])

    # ===================== 数据加载 =====================
    # -------- 传统 CIF CSV 模式 --------
    # 支持两种方式：
    #   1) use_single_csv=True：使用单一 data.csv，并在内部按比例划分 train/val
    #   2) use_single_csv=False（默认）：使用独立的 train.csv / val.csv
    data_root = config["data_path"]
    use_single_csv = config.get("use_single_csv", False)

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
        # 传统：显式提供 train.csv / val.csv
        train_csv = config.get("train_csv", "train.csv")
        val_csv = config.get("val_csv", "val.csv")

        print(f"Loading already split CIF data from {data_root}")
        train_dataset = CIFData(
            root_dir=data_root,
            proper=config.get("property", "formation_energy"),
            random_seed=config["random_seed"],
            csv_file=train_csv,
            shuffle=True,
        )
        val_dataset = CIFData(
            root_dir=data_root,
            proper=config.get("property", "formation_energy"),
            random_seed=config["random_seed"],
            csv_file=val_csv,
            shuffle=False,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_pool,
            num_workers=config.get("workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_pool,
            num_workers=config.get("workers", 4),
            pin_memory=True,
        )

        print(
            f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}\n"
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

    # 创建模型（使用 residual_attn）
    model = CrystalVQVAE(
        hidden_dim=config['hidden_dim'],
        codebook_size=config['codebook_size'],
        num_layers=config.get("num_layers", 4),
        num_space_groups=config.get('num_space_groups', 230),
        max_cn=config.get('max_cn', 20),
        num_elements=config.get('num_elements', 104),
        commitment_weight=config.get('commitment_weight', 0.0),
        loss_weights=config.get('loss_weights', None),
        num_quantizers=config.get('num_quantizers', 4),  # residual_attn的量化器数量（向后兼容）
        num_quantizers_global=config.get('num_quantizers_global', None),  # 全局量化器数量
        num_quantizers_local=config.get('num_quantizers_local', None),   # 局部量化器数量
        sample_codebook_temp=config.get('sample_codebook_temp', 0.1),
        vq_type=config.get("vq_type", "residual_attn"),
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
    print(config.get('num_elements', 104))
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  Global quantizers: {model.num_quantizers_global}, Local quantizers: {model.num_quantizers_local}")

    # 创建优化器（仅优化 requires_grad=True 的参数，便于冻结预训练 encoder）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    frozen_param_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable_param_count}, Frozen params: {frozen_param_count}")

    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check freeze_pretrained_csp and model config.")

    optimizer = optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config.get('min_lr', 1e-6)
    )

    # 训练循环
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 WandB（需要在 save_dir 定义之后）
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            WANDB_AVAILABLE = True
        except ImportError:
            WANDB_AVAILABLE = False
            print("Warning: wandb not installed. Install with: pip install wandb")
            use_wandb = False
    
    if use_wandb:
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "crystal-vqvae"),
            name=wandb_config.get("run_name", None),
            config=config,
            dir=str(save_dir),
        )
        print(f"✓ Wandb initialized: project={wandb_config.get('project', 'crystal-vqvae')}, run={wandb_config.get('run_name', wandb.run.name)}")

    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")

        # warmup_factor = apply_aux_loss_warmup(model, epoch, config)
        # if config.get("aux_warmup_epochs", 0) > 0:
        #     print(f"Aux loss warmup factor: {warmup_factor:.4f}")

        # 训练
        train_loss, train_components, train_stats = train_epoch(
            model, train_loader, optimizer, device, 0,
            grad_clip=config.get('grad_clip', None),
            codebook_size=config.get('codebook_size', None),
            log_interval=30
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Recon Global: {train_components['recon_global']:.4f}")
        print(f"  Recon Local: {train_components['recon_local']:.4f}")
        print(f"  Space Group: {train_components['space_group']:.4f}")
        print(f"  CN: {train_components['cn']:.4f}")
        # print(f"  Neighbor: {train_components['neighbor']:.4f}")
        print(f"  Commitment: {train_components['commitment']:.4f}")

        # 验证
        val_loss, val_components = validate(model, val_loader, device, 0)

        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"  Recon Global: {val_components['recon_global']:.4f}")
        print(f"  Recon Local: {val_components['recon_local']:.4f}")
        print(f"  Space Group: {val_components['space_group']:.4f}")
        print(f"  CN: {val_components['cn']:.4f}")
        # print(f"  Neighbor: {val_components['neighbor']:.4f}")
        print(f"  Commitment: {val_components['commitment']:.4f}")

        # 每个 epoch 只打印一次统计信息（与已有 codebook 统计保持一致）
        print(f"\nCodebook / Embedding / Index Statistics:")
        print_codebook_stats(model, prefix="  ", extra_stats=train_stats)

        # WandB 日志
        if use_wandb:
            wandb_train_stats = {
                f"train_stat_{k}": float(v)
                for k, v in (train_stats or {}).items()
                if isinstance(v, (int, float))
            }
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_components.items()},
                **{f'val_{k}': v for k, v in val_components.items()},
                **wandb_train_stats,
            })

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }
            save_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"\nSaved best model to {save_path}")

        # 定期保存检查点（save_every<=0 时不保存）
        save_every = config.get('save_every', 10)
        if save_every and save_every > 0 and (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }
            save_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # 结束 wandb 运行
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # 3070B
    # CUDA_VISIBLE_DEVICES=0 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_mp_20.yaml --save_dir ./ckpts/mp_20/attn_rvq-Ablation --wandb
    # CUDA_VISIBLE_DEVICES=2 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_mp_20.yaml --save_dir ./ckpts/mp_20/attn_rvq-RVQ --wandb
    # CUDA_VISIBLE_DEVICES=1 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_perov_5.yaml --save_dir "ckpts/perov_5/attn_rvq-MinaRVQ" --wandb
    # CUDA_VISIBLE_DEVICES=2 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_carbon_24.yaml --save_dir "ckpts/carbon_24/attn_rvq-MinaRVQ" --wandb
    # A100
    # CUDA_VISIBLE_DEVICES=3 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_mpts_52.yaml --save_dir "ckpts/mpts_52/attn_rvq-MinaRVQ" --wandb
    # CUDA_VISIBLE_DEVICES=1 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_prep.yaml --save_dir "ckpts/prep/attn_rvq-MinaRVQ" --wandb
    # 消融
    # CUDA_VISIBLE_DEVICES=0 python train_attn_rvq.py --config config/attn_rvq/train_cryvqvae_carbon_24.yaml --save_dir "ckpts/carbon_24/attn_rvq-RVQ" --wandb
    main()
