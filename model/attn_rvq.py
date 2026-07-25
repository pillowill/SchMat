"""Attn-RVQ crystal representation model.
全局使用CSPNet

局部使用CSPNet 局部特征
"""
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from pymatgen.core import Structure, Lattice
from pathlib import Path

from model.cspnet import CSPNet
from model.attention_residual_vq import AttentionResidualVQ


class SharedSemanticBackbone(nn.Module):
    """Lightweight shared MLP backbone for decoder heads."""

    def __init__(self, in_dim: int, decoder_dims=None, dropout: float = 0.1):
        super().__init__()
        dims = list(decoder_dims) if decoder_dims is not None else [256, 256]
        dims.insert(0, in_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalSemanticDecoder(nn.Module):
    def __init__(self, dim, max_cn=20, num_elements=118, hidden_dim=None, decoder_dims=None, dropout=0.1):
        super().__init__()
        self.max_cn = max_cn
        self.num_elements = num_elements
        self.dim = dim

        self.backbone = SharedSemanticBackbone(dim, decoder_dims=decoder_dims, dropout=dropout)
        feat_dim = self.backbone.out_dim

        self.reconstruction_head = nn.Linear(feat_dim, dim)
        self.cn_head = nn.Sequential(
            nn.Linear(feat_dim, max(32, feat_dim // 2)),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(max(32, feat_dim // 2), max_cn),
        )

    def forward(self, z_l):
        features = self.backbone(z_l)
        return {
            "h_recon": self.reconstruction_head(features),
            "cn_logits": self.cn_head(features),
        }


class GlobalSemanticDecoder(nn.Module):
    def __init__(self, dim, num_space_groups=230, hidden_dim=None, decoder_dims=None, dropout=0.1):
        super().__init__()
        self.num_space_groups = num_space_groups
        self.dim = dim

        self.backbone = SharedSemanticBackbone(dim, decoder_dims=decoder_dims, dropout=dropout)
        feat_dim = self.backbone.out_dim

        self.reconstruction_head = nn.Linear(feat_dim, dim)
        self.space_group_head = nn.Sequential(
            nn.Linear(feat_dim, max(32, feat_dim // 2)),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(max(32, feat_dim // 2), num_space_groups),
        )

    def forward(self, z_g):
        features = self.backbone(z_g)
        return {
            "g_recon": self.reconstruction_head(features),
            "space_group_logits": self.space_group_head(features),
        }


class CrystalVQVAE(nn.Module):
    """
    晶体结构的 VQ-VAE 模型
    
    架构：
        Structure → CSPNet → g (global) → VQ → z_g → Decoder_global → space_group
                           ↘ h (local)  → VQ → z_l → Decoder_local  → cn, neighbor_elements
    
    损失：
        - VQ commitment loss (global + local)
        - Space group classification loss
        - Coordination number classification loss  
        - Neighbor elements distribution loss
    """
    def __init__(
        self, 
        hidden_dim=256,
        num_layers=4,
        codebook_size=512,
        num_space_groups=230,
        max_cn=20,
        num_elements=100,
        # 与 VQCrystal 一致：可以通过 EMA / k-means 学习码本，而不依赖 commitment loss
        commitment_weight=0.0,
        loss_weights=None,
        num_quantizers=4,  # ResidualVQ 的量化器数量（vq_type='residual' 时有效，向后兼容）
        num_quantizers_global=None,  # 全局量化器数量（如果为None，则使用num_quantizers）
        num_quantizers_local=None,   # 局部量化器数量（如果为None，则使用num_quantizers）
        sample_codebook_temp: float = 0.1,
        vq_type: str = "residual",  # 'residual' | 'residual_attn' | 'stacked_vq_attn' | 'vector'
        aggregator_dropout: float = 0.1,  # CodebookAttentionAggregator 的 MLP dropout
        aggregator_attn_dropout: float = 0.0,
        aggregator_use_positional: bool = True,
        aggregator_positional_scale: float = 1.0,
        residual_use_outer_aggregator: bool = True,
        residual_attn_use_outer_aggregator: bool = False,
        stacked_vq_use_outer_aggregator: bool = True,
        enable_residual_codebook_params: bool = False,
        residual_codebook_params: dict | None = None,
    ):
        """
        Parameters
        ----------
        hidden_dim : int
            隐藏层维度
        codebook_size : int
            codebook 大小
        num_space_groups : int
            空间群数量（默认230）
        max_cn : int
            最大配位数（默认20）
        num_elements : int
            元素种类数（默认118）
        commitment_weight : float
            VQ commitment loss 权重
        loss_weights : dict, optional
            各损失项权重，默认 {'global': 1.0, 'cn': 1.0, 'neighbor': 1.0, 'commitment': 0.25}
        num_quantizers : int
            ResidualVQ 的量化器数量（默认4，向后兼容，当num_quantizers_global/local为None时使用）
        num_quantizers_global : int, optional
            全局量化器数量（如果为None，则使用num_quantizers）
        num_quantizers_local : int, optional
            局部量化器数量（如果为None，则使用num_quantizers）
        """
        super().__init__()

        vq_type = str(vq_type).lower().strip()
        if vq_type not in ("residual", "residual_attn", "stacked_vq_attn", "vector"):
            raise ValueError(
                f"Unsupported vq_type: {vq_type}. Expected 'residual', 'residual_attn', 'stacked_vq_attn', or 'vector'."
            )
        self.vq_type = vq_type
        self.residual_use_outer_aggregator = bool(residual_use_outer_aggregator)
        self.residual_attn_use_outer_aggregator = bool(residual_attn_use_outer_aggregator)
        self.stacked_vq_use_outer_aggregator = bool(stacked_vq_use_outer_aggregator)
        self.enable_residual_codebook_params = bool(enable_residual_codebook_params)
        # The alpha-weighted ResidualVQ variants do not use an outer aggregator.
        # Define these compatibility attributes for every VQ type nonetheless:
        # older utility code may inspect them, while the HDS metric below must not
        # depend on an encoder-side attention module.
        self.global_aggregator = None
        self.local_aggregator = None
        if residual_codebook_params is None:
            self.residual_codebook_params = {}
        elif isinstance(residual_codebook_params, dict):
            self.residual_codebook_params = dict(residual_codebook_params)
        else:
            raise ValueError("residual_codebook_params must be a dict or None")

        # 设置全局和局部的量化器数量
        if num_quantizers_global is None:
            num_quantizers_global = num_quantizers
        if num_quantizers_local is None:
            num_quantizers_local = num_quantizers
        
        self.num_quantizers_global = num_quantizers_global
        self.num_quantizers_local = num_quantizers_local
        # 向后兼容：保留 num_quantizers 属性（使用全局的作为默认）
        self.num_quantizers = num_quantizers_global

        
        # -------- Encoder --------
        # VQVAE encoder 不需要时间嵌入和条件嵌入，明确设置为 0 以避免创建不必要的 FilmLayer
        self.encoder = CSPNet(
            hidden_dim=hidden_dim,
            time_dim=0,      # VQVAE encoder 不需要时间嵌入
            cond_dim=0,      # VQVAE encoder 不需要条件嵌入
            num_layers=num_layers,
            max_atoms=int(num_elements),
            cond_guided=False  # VQVAE encoder 不需要条件引导
        )


        
        # -------- VQ modules --------
        # 兼容两种 checkpoint：
        # - vq_type='residual': ResidualVQ（多量化器），state_dict key 类似 `vq_global.layers.0._codebook.*`
        # - vq_type='residual_attn': 每一级 residual 更新使用 attention 门控加权
        # - vq_type='stacked_vq_attn': 非残差的多层VQ，输出通过外部聚合器聚合
        # - vq_type='vector'  : VectorQuantize（单一码本），state_dict key 类似 `vq_global._codebook.*`
        if self.vq_type in ("residual", "residual_attn", "stacked_vq_attn"):
            if self.vq_type == "residual_attn":
                vq_cls = AttentionResidualVQ
            elif self.vq_type == "vector":
                vq_cls = VectorQuantize
            else:
                vq_cls = ResidualVQ

            # Optional codebook kwargs are only meaningful for residual-style quantizers.
            residual_vq_kwargs = {}
            if self.enable_residual_codebook_params and self.vq_type in ("residual", "residual_attn"):
                residual_vq_kwargs = dict(self.residual_codebook_params)

            self.vq_global = vq_cls(
                dim=hidden_dim,
                num_quantizers=num_quantizers_global,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                # stochastic_sample_codes=True,
                # sample_codebook_temp=sample_codebook_temp,
                # #
                # shared_codebook=True,  # whether to share the codebooks for all quantizers or not
                # kmeans_init=True,  # set to True
                # kmeans_iters=10,
                **residual_vq_kwargs,
            )
            self.vq_local = vq_cls(
                dim=hidden_dim,
                num_quantizers=num_quantizers_local,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                # stochastic_sample_codes=True,
                # sample_codebook_temp=sample_codebook_temp,
                # #
                # shared_codebook=True,  # whether to share the codebooks for all quantizers or not
                # kmeans_init=True,  # set to True
                # kmeans_iters=10,
                **residual_vq_kwargs,
            )
        else:
            # VectorQuantize：单一码本
            self.num_quantizers_global = 1
            self.num_quantizers_local = 1
            self.num_quantizers = 1

            # 注意：VectorQuantize 的接口/参数在不同版本可能略有差异，
            # 但其 state_dict 的核心字段通常是 `_codebook.*`（与你的 MP-20 checkpoint 一致）。
            self.vq_global = VectorQuantize(
                dim=hidden_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
            )
            self.vq_local = VectorQuantize(
                dim=hidden_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
            )

        # -------- Decoders --------

        self.global_decoder = GlobalSemanticDecoder(
            dim=hidden_dim,
            num_space_groups=num_space_groups
        )
        self.local_decoder = LocalSemanticDecoder(
            dim=hidden_dim,
            max_cn=max_cn,
            num_elements=num_elements
        )
        
        # -------- Loss weights --------
        self.loss_weights = loss_weights
        
        # -------- 其他 --------
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.max_cn = max_cn
        self.num_elements = num_elements
        self.epoch = 0
        self.step = 0



    def encode(self, atom_types, frac_coords, lattice, num_atoms, node2batch):
        """
        编码：Structure → CSPNet → (g, h)
        
        Parameters
        ----------
        atom_types : Tensor
            原子类型（原子序数） (total_atoms,) 
        frac_coords : Tensor
            分数坐标 (total_atoms, 3)
        lattice : Tensor
            晶格矩阵 (batch_size, 3, 3)
        num_atoms : Tensor
            每个结构的原子数 (batch_size,)
        node2batch : Tensor
            节点到batch的映射 (total_atoms,)
        
        Returns
        -------
        g : Tensor
            全局特征 (batch_size, hidden_dim)
        h : Tensor
            局部特征 (total_atoms, hidden_dim)
        """
        # CSPNet 编码
        h, g = self.encoder(atom_types, frac_coords, lattice, num_atoms, node2batch)
        # h: (total_atoms, hidden_dim) - 局部特征
        # g: (batch_size, hidden_dim) - 全局特征
        return g, h

    def quantize(self, g, h):
        """
        量化：(g, h) → ResidualVQ → (z_g, z_l, indices, commitment_loss)
        
        Parameters
        ----------
        g : Tensor
            全局特征 (batch_size, hidden_dim)
        h : Tensor
            局部特征 (total_atoms, hidden_dim)
        
        Returns
        -------
        z_g : Tensor
            聚合后的全局特征 (batch_size, hidden_dim)
        z_l : Tensor
            聚合后的局部特征 (total_atoms, hidden_dim)
        indices_g : Tensor
            全局codebook索引 (batch_size, num_quantizers_global)
        indices_l : Tensor
            局部codebook索引 (total_atoms, num_quantizers_local)
        commitment_loss : Tensor
            commitment loss (global + local)
        """
        if self.vq_type in ("residual", "residual_attn", "stacked_vq_attn"):
            # 全局 ResidualVQ
            # ResidualVQ 输入需要 (batch_size, seq_len, hidden_dim)
            z_g_quantized, indices_g, commit_loss_g = self.vq_global(g.unsqueeze(1))
            indices_g = indices_g.squeeze(1)  # (batch_size, num_quantizers_global)
            commit_loss_g = commit_loss_g.mean()

            # 默认用量化输出；按模式可选开启外层 codebook attention 聚合
            z_g = z_g_quantized.squeeze(1)

            use_outer_agg = (
                (self.vq_type == "residual" and self.residual_use_outer_aggregator)
                or (self.vq_type == "residual_attn" and self.residual_attn_use_outer_aggregator)
                or (self.vq_type == "stacked_vq_attn" and self.stacked_vq_use_outer_aggregator)
            )
            if use_outer_agg:
                codebook_vectors_g = self.vq_global.get_codes_from_indices(indices_g)  # (batch_size, num_quantizers_global, hidden_dim)
                if codebook_vectors_g.shape[0] == self.num_quantizers_global:
                    # (Q, B, D) -> (B, Q, D)
                    codebook_vectors_g = codebook_vectors_g.transpose(0, 1)
                # 使用注意力聚合器聚合
                z_g = self.global_aggregator(codebook_vectors_g)  # (batch_size, hidden_dim)

            # 局部 ResidualVQ
            z_l_quantized, indices_l, commit_loss_l = self.vq_local(h.unsqueeze(1))
            indices_l = indices_l.squeeze(1)  # (total_atoms, num_quantizers_local)
            commit_loss_l = commit_loss_l.mean()

            z_l = z_l_quantized.squeeze(1)
            if use_outer_agg:
                codebook_vectors_l = self.vq_local.get_codes_from_indices(indices_l)  # (total_atoms, num_quantizers_local, hidden_dim)
                if codebook_vectors_l.shape[0] == self.num_quantizers_local:
                    # (Q, B, D) -> (B, Q, D)
                    codebook_vectors_l = codebook_vectors_l.transpose(0, 1)
                # 使用注意力聚合器聚合
                z_l = self.local_aggregator(codebook_vectors_l)  # (total_atoms, hidden_dim)

            # commitment loss 总是标量
            commitment_loss = commit_loss_g + commit_loss_l
        else:
            # VectorQuantize：通常输入 (N, D)，输出 (N, D), (N,), loss
            z_g, indices_g, commit_loss_g = self.vq_global(g)
            z_l, indices_l, commit_loss_l = self.vq_local(h)
            # 统一规约为标量
            commitment_loss = commit_loss_g.mean() + commit_loss_l.mean()
        
        return z_g, z_l, indices_g, indices_l, commitment_loss
    
    def _default_use_outer_aggregation(self) -> bool:
        if self.vq_type == "residual_attn":
            return bool(self.residual_attn_use_outer_aggregator)
        if self.vq_type == "residual":
            return bool(self.residual_use_outer_aggregator)
        if self.vq_type == "stacked_vq_attn":
            return bool(self.stacked_vq_use_outer_aggregator)
        return False

    def get_att_agg_output_from_indices(self, indices, is_global=True, use_aggregation=None):
        """
        从 codebook indices 获取对应的连续向量（支持 ResidualVQ 的多个 codebook）
        
        Parameters
        ----------
        indices : Tensor
            codebook 索引 (batch_size, num_quantizers_global/local) 或 (total_atoms, num_quantizers_global/local)
            如果是旧格式 (batch_size,) 或 (total_atoms,)，会自动扩展为 (..., num_quantizers_global/local)
        is_global : bool
            是否为全局 codebook（True）或局部 codebook（False）
        use_aggregation : bool | None
            是否使用注意力机制聚合多个 codebook 向量。
            为 None 时按当前 vq_type 与配置自动选择。
        
        Returns
        -------
        Tensor
            对应的连续向量 (batch_size, hidden_dim) 或 (total_atoms, hidden_dim)
        """
        if is_global:
            vq_module = self.vq_global
            aggregator = self.global_aggregator
        else:
            vq_module = self.vq_local
            aggregator = self.local_aggregator

        if use_aggregation is None:
            use_aggregation = self._default_use_outer_aggregation()

        # VectorQuantize：直接从单一码本取 embedding（不需要聚合器）
        if self.vq_type == "vector":
            if indices.dim() > 1:
                indices = indices.squeeze(-1)
            indices = indices.long()
            # 直接调用 get_codes_from_indices 或 fallback
            if hasattr(vq_module, "get_codes_from_indices"):
                return vq_module.get_codes_from_indices(indices)
            # fallback：访问内部 codebook
            for cb_attr in ("_codebook", "codebook"):
                if hasattr(vq_module, cb_attr):
                    cb = getattr(vq_module, cb_attr)
                    if hasattr(cb, "embed"):
                        embed = cb.embed
                        return embed[indices]
            raise ValueError("VectorQuantize codebook access failed: could not locate embedding tensor.")
        
        # ResidualVQ：先取出每个量化器对应的 codebook 向量，再按需聚合
        if hasattr(vq_module, "get_codes_from_indices"):
            num_quantizers = self.num_quantizers_global if is_global else self.num_quantizers_local
            codebook_vectors = vq_module.get_codes_from_indices(indices)  # 可能 (Q, B, D) 或 (B, Q, D)
            if codebook_vectors.shape[0] == num_quantizers:
                # (Q, B, D) -> (B, Q, D)
                codebook_vectors = codebook_vectors.transpose(0, 1)
            # 确保输出为 (B, Q, D)
            if not use_aggregation:
                # 简单求和（ResidualVQ 的残差量化特性）
                return vq_module.get_output_from_indices(indices)  # (B, D)
            if aggregator is None:
                raise ValueError(
                    "global_aggregator/local_aggregator is None, but use_aggregation=True was requested."
                )
            return aggregator(codebook_vectors)  # (B, D)
        else:
            raise ValueError("ResidualVQ does not have get_codes_from_indices method.")

    @torch.no_grad()
    def get_hds_metric_embedding_from_indices(self, indices, is_global: bool = True) -> torch.Tensor:
        """Return the deterministic, index-only VQ embedding used for HDS metrics.

        HDS stores quantizer indices but not the encoder residual seen at each
        alpha-weighted ``residual_attn`` stage. Consequently, its exact
        encoder-conditioned alpha values cannot be reconstructed after
        generation. For a fair generated-vs-reference comparison, this method
        maps both HDSs into the same trained codebook space without fabricating
        those unavailable residuals. Residual-style VQs therefore use the sum
        of their selected stage codebook vectors; a vector VQ uses its selected
        codebook vector directly.

        This is intentionally separate from ``get_att_agg_output_from_indices``:
        that method serves model paths which may request an outer aggregator,
        whereas the HDS cosine metric must be fully determined by discrete codes.
        """
        vq_module = self.vq_global if is_global else self.vq_local

        if self.vq_type == "vector":
            if indices.dim() > 1:
                indices = indices.squeeze(-1)
            indices = indices.long()
            if hasattr(vq_module, "get_codes_from_indices"):
                return vq_module.get_codes_from_indices(indices)
            for codebook_attr in ("_codebook", "codebook"):
                if hasattr(vq_module, codebook_attr):
                    codebook = getattr(vq_module, codebook_attr)
                    if hasattr(codebook, "embed"):
                        return codebook.embed[indices]
            raise ValueError("VectorQuantize codebook access failed: could not locate embedding tensor.")

        if not hasattr(vq_module, "get_output_from_indices"):
            raise ValueError("Residual-style VQ does not expose index-to-output conversion.")
        return vq_module.get_output_from_indices(indices)

    def get_codebook_vectors_from_indices(self, indices, is_global: bool = True) -> torch.Tensor:
        """
        从 codebook indices 获取所有量化器对应的 code 向量，不进行聚合。
        返回形状为 (batch_size, num_quantizers_global/local, hidden_dim)。

        该接口用于在扩散模型中自定义聚合策略（例如 CodebookAttentionAggregator）。
        当前仅支持 ResidualVQ（vq_type='residual'）。
        """
        if self.vq_type != "residual":
            raise NotImplementedError(
                "get_codebook_vectors_from_indices 目前只支持 ResidualVQ（vq_type='residual'）。"
            )

        # 根据 is_global 选择对应的 num_quantizers
        num_quantizers = self.num_quantizers_global if is_global else self.num_quantizers_local

        # 处理索引形状：如果是旧格式，扩展为多量化器格式
        if indices.dim() == 1:
            # 需要区分两种情况：
            # 1. (num_quantizers,) - 单个样本的多个量化器，需要添加 batch 维度 -> (1, num_quantizers)
            # 2. (batch_size,) - 多个样本的单个量化器（旧格式），需要添加 quantizer 维度 -> (batch_size, num_quantizers)
            # 判断标准：如果长度等于 num_quantizers，很可能是情况1
            if indices.shape[0] == num_quantizers:
                # 单个样本的多个量化器：添加 batch 维度
                indices = indices.unsqueeze(0)  # (num_quantizers,) -> (1, num_quantizers)
            else:
                # 多个样本的单个量化器（旧格式）：添加 quantizer 维度
                indices = indices.unsqueeze(-1).expand(-1, num_quantizers)
        elif indices.dim() == 2:
            if indices.shape[1] == self.codebook_size:
                if indices.dtype not in (torch.long, torch.int64, torch.int32):
                    indices = indices.argmax(dim=1)
                else:
                    indices = indices.argmax(dim=1)
                indices = indices.unsqueeze(-1).expand(-1, num_quantizers)
            elif indices.shape[1] != num_quantizers:
                raise ValueError(
                    f"Unexpected indices shape: {indices.shape}. "
                    f"Expected (B,), (B, {num_quantizers}), or (B, {self.codebook_size})"
                )

        if is_global:
            vq_module = self.vq_global
        else:
            vq_module = self.vq_local

        # 使用 get_codes_from_indices 获取未聚合的 codebook 向量
        if hasattr(vq_module, "get_codes_from_indices"):
            codebook_vectors = vq_module.get_codes_from_indices(indices)  # 可能 (Q, B, D) 或 (B, Q, D)
            if codebook_vectors.shape[0] == num_quantizers:
                # (Q, B, D) -> (B, Q, D)
                codebook_vectors = codebook_vectors.transpose(0, 1)
            # 确保输出为 (B, Q, D)
            return codebook_vectors
        else:
            raise ValueError("ResidualVQ does not have get_codes_from_indices method.")

    def decode(self, z_g, z_l):
        """
        解码：(z_g, z_l) → Decoders → reconstructions + predictions
        
        Parameters
        ----------
        z_g : Tensor
            量化后的全局特征 (batch_size, hidden_dim)
        z_l : Tensor
            量化后的局部特征 (total_atoms, hidden_dim)
        
        Returns
        -------
        dict
            - g_recon: (batch_size, dim) 重建的全局语义嵌入
            - h_recon: (total_atoms, dim) 重建的局部语义嵌入
            - space_group_logits: (batch_size, num_space_groups) 空间群预测
            - cn_logits: (total_atoms, max_cn) 配位数预测
            - neighbor_logits: (total_atoms, num_elements) 邻居元素预测
        """
        # 全局解码
        global_outputs = self.global_decoder(z_g)
        
        # 局部解码
        local_outputs = self.local_decoder(z_l)

        return {

            # 重建的语义嵌入（主任务）
            'g_recon': global_outputs['g_recon'],
            'h_recon': local_outputs['h_recon'],
            # 辅助任务预测
            'space_group_logits': global_outputs['space_group_logits'],
            'cn_logits': local_outputs['cn_logits'],
            # 'neighbor_logits': local_outputs['neighbor_logits'],
        }

    def forward(self, atom_types, frac_coords, lattice, num_atoms, node2batch, 
                global_targets, local_targets):
        """
        前向传播 + 损失计算
        
        Parameters
        ----------
        atom_types : Tensor
            原子类型（原子序数） (total_atoms,)
        frac_coords : Tensor
            分数坐标 (total_atoms, 3)
        lattice : Tensor
            晶格矩阵 (batch_size, 3, 3)
        num_atoms : Tensor
            每个结构的原子数 (batch_size,)
        node2batch : Tensor
            节点到batch的映射 (total_atoms,)
        global_targets : Tensor
            全局目标（空间群编号） (batch_size,)
        local_targets : list of list of dict
            局部目标，每个结构包含每个原子的 {'cn': int, 'neighbor_elements': dict}
        
        Returns
        -------
        dict
            包含各损失项的字典
        """
        # 1. 编码：Structure → CSPNet → (g, h)
        g, h = self.encode(atom_types, frac_coords, lattice, num_atoms, node2batch)
        
        # 2. 量化：(g, h) → VQ → (z_g, z_l)
        z_g, z_l, indices_g, indices_l, commitment_loss = self.quantize(g, h)
        
        # 3. 解码：(z_g, z_l) → Decoders → (ĝ, ĥ, predictions)
        outputs = self.decode(z_g, z_l)
        
        # 4. 计算损失
        losses = {}
        
        # 4.1 主任务：语义重建损失
        # 全局语义重建（MSE loss）
        recon_global_loss = F.mse_loss(outputs['g_recon'], g.detach())
        losses['recon_global_loss'] = recon_global_loss
        
        # 局部语义重建（MSE loss）
        recon_local_loss = F.mse_loss(outputs['h_recon'], h.detach())
        losses['recon_local_loss'] = recon_local_loss
        
        # 4.2 辅助任务：空间群分类（CE loss）
        space_group_loss = F.cross_entropy(
            outputs['space_group_logits'], 
            global_targets
        )
        losses['space_group_loss'] = space_group_loss
        
        # 4.3 辅助任务：配位数分类（CE loss）
        # 从 local_targets 提取 cn
        cn_targets = []
        cn_mask = []
        for batch_idx, structure_targets in enumerate(local_targets):
            for atom_target in structure_targets:
                cn = atom_target['cn']
                if cn is None or cn < 0 or cn >= self.max_cn:
                    cn_targets.append(0)  # dummy
                    cn_mask.append(0)
                else:
                    cn_targets.append(atom_target['cn'])
                    cn_mask.append(1)
        cn_targets = torch.tensor(cn_targets, dtype=torch.long, device=outputs['cn_logits'].device)
        cn_mask = torch.tensor(cn_mask, dtype=torch.float, device=outputs['cn_logits'].device).bool()
        
        # 确保有有效的配位数目标
        if cn_mask.sum() > 0:
            cn_loss = F.cross_entropy(outputs['cn_logits'][cn_mask], cn_targets[cn_mask])
        else:
            # 如果没有有效目标，使用 dummy loss 确保梯度流
            cn_loss = torch.tensor(0.0, device=outputs['cn_logits'].device, requires_grad=True)
        losses['cn_loss'] = cn_loss
        
        # 4.5 VQ commitment loss
        losses['commitment_loss'] = commitment_loss
        
        # 4.6 返回 codebook 索引（用于监控 collapse）
        # ResidualVQ 返回的 indices 形状为 (batch_size, num_quantizers_global) 或 (total_atoms, num_quantizers_local)
        losses['indices_g'] = indices_g  # (batch_size, num_quantizers_global)
        losses['indices_l'] = indices_l  # (total_atoms, num_quantizers_local)
        # 记录 encoder 输出 embedding，用于统计分布
        losses['emb_global'] = g        # (batch_size, hidden_dim)
        losses['emb_local'] = h        # (total_atoms, hidden_dim)

        # 记录 residual_attn 每一层 alpha 的统计，供训练日志打印
        if self.vq_type == "residual_attn":
            g_alpha_mean = getattr(self.vq_global, "last_alpha_means", None)
            g_alpha_std = getattr(self.vq_global, "last_alpha_stds", None)
            l_alpha_mean = getattr(self.vq_local, "last_alpha_means", None)
            l_alpha_std = getattr(self.vq_local, "last_alpha_stds", None)

            if g_alpha_mean is not None:
                for i, v in enumerate(g_alpha_mean):
                    losses[f"alpha_global_stage{i}_mean"] = float(v)
            if g_alpha_std is not None:
                for i, v in enumerate(g_alpha_std):
                    losses[f"alpha_global_stage{i}_std"] = float(v)
            if l_alpha_mean is not None:
                for i, v in enumerate(l_alpha_mean):
                    losses[f"alpha_local_stage{i}_mean"] = float(v)
            if l_alpha_std is not None:
                for i, v in enumerate(l_alpha_std):
                    losses[f"alpha_local_stage{i}_std"] = float(v)
        
        # 5. 总损失（加权求和）
        total_loss = (
            self.loss_weights['recon_global'] * recon_global_loss +
            self.loss_weights['recon_local'] * recon_local_loss +
            self.loss_weights['space_group'] * space_group_loss +
            self.loss_weights['cn'] * cn_loss +
            # self.loss_weights['neighbor'] * neighbor_loss +
            self.loss_weights['commitment'] * commitment_loss
        )
        losses['total_loss'] = total_loss
        
        return losses

    @torch.no_grad()
    def generate(self, z_g_indices, z_l_indices, num_atoms):
        """
        从codebook indices生成语义重建和结构预测
        
        Parameters
        ----------
        z_g_indices : Tensor
            全局codebook索引 (batch_size, num_quantizers_global) 或 (batch_size,)
        z_l_indices : Tensor
            局部codebook索引 (total_atoms, num_quantizers_local) 或 (total_atoms,)
        num_atoms : Tensor
            每个结构的原子数 (batch_size,)
        
        Returns
        -------
        dict
            包含重建的语义嵌入和预测结果
        """
        # 从codebook获取特征（使用注意力聚合）
        z_g = self.get_att_agg_output_from_indices(z_g_indices, is_global=True)  # (batch_size, hidden_dim)
        z_l = self.get_att_agg_output_from_indices(z_l_indices, is_global=False)  # (total_atoms, hidden_dim)
        
        # 解码
        outputs = self.decode(z_g, z_l)
        
        # 整理输出
        predictions = {
            # 重建的语义嵌入
            'g_recon': outputs['g_recon'],  # (batch_size, hidden_dim)
            'h_recon': outputs['h_recon'],  # (total_atoms, hidden_dim)
            # 辅助任务预测
            'space_group': outputs['space_group_logits'].argmax(dim=-1),  # (batch_size,)
            'cn': outputs['cn_logits'].argmax(dim=-1),  # (total_atoms,)
            # 'neighbor_elements': F.softmax(outputs['neighbor_logits'], dim=-1),  # (total_atoms, num_elements)
        }
        
        return predictions
