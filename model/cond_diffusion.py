"""
该脚本是条件扩散模型对应于固定atom-types的版本，该版本不需要对atom-types进行扩散
"""
from typing import Union, Any, Dict, List, Optional
import os
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from model.cspnet import CSPNet, SinusoidalTimeEmbeddings
from utils.diff_utils import (
    BetaScheduler,
    SigmaScheduler,
    d_log_p_wrapped_normal,
    D3PM,
    prob_mask_like,
)
from model.attn_rvq import CrystalVQVAE
from model.schema import TrajectoryStep, TrajectoryContainer



class CondDiffusion(nn.Module):
    def __init__(self, _config: Dict[str, Any], pretrained_vqvae_model: Optional[CrystalVQVAE] = None, **kwargs):
        super().__init__()
        self.config = _config
        # time embedding
        self.time_embed = SinusoidalTimeEmbeddings(_config["time_dim"])
        # conditional guidance
        self.cond_guide = _config.get("cond_guide", False)
        self.pretrained_vqvae_model = pretrained_vqvae_model
        if self.cond_guide:
            if self.pretrained_vqvae_model is None and "path_ckpt_vqvae" in kwargs:
                # 从检查点加载预训练模型
                checkpoint_path = kwargs["path_ckpt_vqvae"]
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model_config = checkpoint.get('config', {})
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                inferred_num_elements = model_config.get('num_elements', 118)
                emb_key = 'encoder.node_embedding.weight'
                if emb_key in state_dict and hasattr(state_dict[emb_key], 'shape'):
                    inferred_num_elements = int(state_dict[emb_key].shape[0])

                self.pretrained_vqvae_model = CrystalVQVAE(
                    hidden_dim=model_config.get('hidden_dim', 256),
                    num_layers=model_config.get('num_layers', 4),
                    codebook_size=model_config.get('codebook_size', 512),
                    num_space_groups=model_config.get('num_space_groups', 230),
                    max_cn=model_config.get('max_cn', 20),
                    num_elements=inferred_num_elements,
                    commitment_weight=model_config.get('commitment_weight', 0.25),
                    loss_weights=model_config.get('loss_weights', None),
                    num_quantizers=model_config.get('num_quantizers', 4),  # 向后兼容
                    num_quantizers_global=model_config.get('num_quantizers_global', None),  # 全局量化器数量
                    num_quantizers_local=model_config.get('num_quantizers_local', None),   # 局部量化器数量
                    sample_codebook_temp=model_config.get('sample_codebook_temp', 0.1),
                    vq_type=model_config.get('vq_type', 'residual'),
                    aggregator_dropout=model_config.get('aggregator_dropout', 0.1),
                    aggregator_attn_dropout=model_config.get('aggregator_attn_dropout', 0.0),
                    aggregator_use_positional=model_config.get('aggregator_use_positional', True),
                    aggregator_positional_scale=model_config.get('aggregator_positional_scale', 1.0),
                    residual_use_outer_aggregator=model_config.get('residual_use_outer_aggregator', True),
                    residual_attn_use_outer_aggregator=model_config.get('residual_attn_use_outer_aggregator', False),
                    stacked_vq_use_outer_aggregator=model_config.get('stacked_vq_use_outer_aggregator', True),
                    enable_residual_codebook_params=model_config.get('enable_residual_codebook_params', False),
                    residual_codebook_params=model_config.get('residual_codebook_params', None),
                )
                # strict=False cannot ignore same-key shape mismatch; pre-filter by shape.
                model_state = self.pretrained_vqvae_model.state_dict()
                filtered_state = {}
                for k, v in state_dict.items():
                    if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                        filtered_state[k] = v
                self.pretrained_vqvae_model.load_state_dict(filtered_state, strict=False)
                self.pretrained_vqvae_model.eval()
            self.cond_drop_prob = _config.get("cond_drop_prob", 0.1)
            
            # 为 VQ 特征定义可学习的 null embedding (初始化为 0 起步平滑)
            self.cond_dim = _config.get("cond_dim", 256)
            self.null_vq_g = nn.Parameter(torch.zeros(self.cond_dim))
            self.null_vq_l = nn.Parameter(torch.zeros(self.cond_dim))

        self.property_cond_enabled = bool(_config.get("property_cond_enabled", False))
        self.property_vocab_size = int(_config.get("property_vocab_size", 0))
        self.property_fusion_mode = _config.get("property_fusion_mode", "linear")
        self.property_cond_drop_prob = float(_config.get("property_cond_drop_prob", self.cond_drop_prob if self.cond_guide else 0.1))
        self.property_pred_weight = float(_config.get("property_pred_weight", 0.0))
        if self.property_cond_enabled:
            if self.property_vocab_size <= 0:
                raise ValueError("property_cond_enabled=True requires property_vocab_size > 0")
            prop_dim = int(_config.get("property_emb_dim", _config.get("cond_dim", _config.get("hidden_dim", 256))))
            self.property_embedding = nn.Embedding(self.property_vocab_size + 1, prop_dim, padding_idx=0)
            self.property_proj = nn.Linear(prop_dim, _config.get("cond_dim", 256))
            if self.property_fusion_mode == "linear":
                self.property_fuse = nn.Linear(_config.get("cond_dim", 256) * 2, _config.get("cond_dim", 256))
            else:
                self.property_fuse = None
            self.property_head = nn.Linear(_config["hidden_dim"], self.property_vocab_size)
            
            # 为属性条件定义可学习的 null embedding
            self.null_prop = nn.Parameter(torch.zeros(_config.get("cond_dim", 256)))
        else:
            self.property_embedding = None
            self.property_proj = None
            self.property_fuse = None
            self.property_head = None
        # set scheduler
        self.num_timesteps = _config["timesteps"]
        self.beta_scheduler = BetaScheduler(
            timesteps=self.num_timesteps, scheduler_mode=_config.get("beta_schedule", "cosine")
        )
        self.sigma_scheduler = SigmaScheduler(timesteps=self.num_timesteps)
        # d3pm for atom types
        self.max_atoms = _config["max_atoms"]
        self.diffuse_atom_types = _config.get("diffuse_atom_types", True)  # 是否扩散原子类型
        if self.diffuse_atom_types:
            self.d3pm = D3PM(
                beta_scheduler=self.beta_scheduler,
                num_timesteps=_config["timesteps"],
                max_atoms=_config["max_atoms"],
                d3pm_hybrid_coeff=_config.get("d3pm_hybrid_coeff", 0.0),
            )
        else:
            self.d3pm = None  # 原子类型固定时不需要 D3PM
        # lattice noise mask (mask out 3 elements that can be derived from others)
        # all lattice matrix are re-defined from a function "from_paramters" in pymatgen
        # 使用 register_buffer 注册，PyTorch 会自动管理设备
        self.register_buffer(
            'mask_lattice_matrix',
            torch.tensor([[1, 0, 1], [1, 1, 1], [0, 0, 1]]).bool()
        )
        # decoder
        # 如果原子类型不扩散，则不需要预测原子类型（但仍需要接收原子类型作为输入）
        pred_atom_types_config = _config.get("pred_atom_types", True)
        # 当 diffuse_atom_types=False 时，强制 pred_atom_types=False（原子类型固定，不需要预测）
        if not self.diffuse_atom_types:
            pred_atom_types_config = False
        self.hds_condition_mode = str(_config.get("hds_condition_mode", "full")).strip().lower()
        self.decoder = CSPNet(
            hidden_dim=_config["hidden_dim"],
            cond_dim=_config.get("cond_dim", 256) if self.cond_guide else 0,
            time_dim=_config["time_dim"],
            num_layers=_config["num_layers"],
            max_atoms=_config["max_atoms"],
            act_fn=_config.get("act_fn", "silu"),
            dis_emb=_config.get("dis_emb", "sin"),
            num_freqs=_config.get("num_freqs", 10),
            edge_style=_config.get("edge_style", "fc"),
            cutoff=_config.get("cutoff", 6.0),
            max_neighbors=_config.get("max_neighbors", 20),
            ln=_config.get("ln", False),
            ip=_config.get("ip", True),
            smooth=_config.get("smooth", False),
            cond_guided=self.cond_guide,  # 启用条件引导
            pred_atom_types=pred_atom_types_config,  # 根据是否扩散原子类型决定是否预测
            use_multiscale_film=bool(_config.get("use_multiscale_film", False)),
            local_film_end_layer=_config.get("local_film_end_layer", None),
            global_film_start_layer=_config.get("global_film_start_layer", None),
            hds_condition_mode=self.hds_condition_mode,
        )
        print(f"[HDS condition] diffusion interface mode={self.hds_condition_mode}")
        # set cost for loss
        self.cost_atom_types = _config.get("cost_atom_types", 1.0)
        self.cost_lattice = _config.get("cost_lattice", 1.0)
        self.cost_coords = _config.get("cost_coords", 1.0)

    def _fuse_global_and_property(self, global_cond: torch.Tensor, property_cond: Optional[torch.Tensor]) -> torch.Tensor:
        if property_cond is None:
            return global_cond
        if self.property_fusion_mode == "sum":
            return global_cond + property_cond
        if self.property_fusion_mode == "mean":
            return 0.5 * (global_cond + property_cond)
        if self.property_fusion_mode == "linear" and self.property_fuse is not None:
            return self.property_fuse(torch.cat([global_cond, property_cond], dim=-1))
        return global_cond

    def _get_property_cond(self, property_bins: Optional[torch.Tensor], cond_drop_prob: float, device: torch.device) -> Optional[torch.Tensor]:
        if (not self.property_cond_enabled) or (self.property_embedding is None) or (property_bins is None):
            return None
        property_bins = property_bins.to(device).long().view(-1)
        # Shift by +1 so -1/invalid can map to padding index 0.
        shifted = torch.clamp(property_bins + 1, min=0, max=self.property_vocab_size)
        prop_emb = self.property_embedding(shifted)
        prop_cond = self.property_proj(prop_emb)
        keep_mask = prob_mask_like((prop_cond.shape[0],), 1.0 - cond_drop_prob, device)
        
        # 使用可学习的 null token
        prop_cond = torch.where(keep_mask[:, None], prop_cond, self.null_prop.unsqueeze(0).expand_as(prop_cond))
        return prop_cond

    def get_vq_vector(self, batch_vq_indices, cond_drop_prob, device):
        """
        从 VQ indices 获取连续向量并应用条件 drop

        Parameters
        ----------
        batch_vq_indices : tuple
            (indices_g, indices_l) 全局和局部 VQ indices
        cond_drop_prob : float
            条件 drop 概率（用于 classifier-free guidance）
        device : torch.device
            设备

        Returns
        -------
        tuple
            (vq_vector_g, vq_vector_l) 全局和局部 VQ 向量（已应用条件 drop）
        """
        batch_indices_g, batch_indices_l = batch_vq_indices

        if self.pretrained_vqvae_model is None:
            raise ValueError("pretrained_vqvae_model is None when trying to get VQ vectors.")

        # 按 VQ 模式自动选择是否使用外层聚合（residual_attn 可关闭外层聚合）
        use_outer_agg = None
        if hasattr(self.pretrained_vqvae_model, "_default_use_outer_aggregation"):
            use_outer_agg = self.pretrained_vqvae_model._default_use_outer_aggregation()

        batch_vq_vector_g = self.pretrained_vqvae_model.get_att_agg_output_from_indices(
            batch_indices_g, is_global=True, use_aggregation=use_outer_agg
        )
        batch_vq_vector_l = self.pretrained_vqvae_model.get_att_agg_output_from_indices(
            batch_indices_l, is_global=False, use_aggregation=use_outer_agg
        )
        
        batch_size_g = batch_vq_vector_g.shape[0]
        batch_size_l = batch_vq_vector_l.shape[0]

        # 条件 drop（用于 classifier-free guidance）
        cond_keep_mask_g = prob_mask_like(
            (batch_size_g,), 1.0 - cond_drop_prob, device
        )  # (batch_size,)
        cond_keep_mask_l = prob_mask_like(
            (batch_size_l,), 1.0 - cond_drop_prob, device
        )  # (total_atoms,)

        # 对于被 drop 的条件，使用可学习的 null condition
        batch_vq_vector_g = torch.where(
            cond_keep_mask_g[:, None],
            batch_vq_vector_g,
            self.null_vq_g.unsqueeze(0).expand_as(batch_vq_vector_g)
        )
        batch_vq_vector_l = torch.where(
            cond_keep_mask_l[:, None],
            batch_vq_vector_l,
            self.null_vq_l.unsqueeze(0).expand_as(batch_vq_vector_l)
        )
        return batch_vq_vector_g, batch_vq_vector_l


    def forward(self, batch: Batch) -> Dict[str, Any]:
        """Calculate P_loss
           在这里输入的其实应该是预训练的vqvae，而不是直接的codebook索引。
           考虑到在后期采样过程中需要接收来自LLM提供的codebook，这里应该如何设计呢？
           ➡ 应该是：在训练阶段，不用提供codebook索引，而在采样阶段需要输入codebook索引

        Args:
            batch (Batch): Batch
        Returns:
            Dict[str, Any]: results
        """
        device = next(self.parameters()).device
        batch_size = batch.num_graphs
        num_nodes = batch.num_nodes
        batched_t = self.beta_scheduler.uniform_sample_t(batch_size, device)  # [B]
        time_emb = self.time_embed(batched_t)  # [B, time_dim]

        alpha_cumprod = self.beta_scheduler.alphas_cumprod[batched_t]  # [B]

        c0 = torch.sqrt(alpha_cumprod)  # [B]
        c1 = torch.sqrt(1.0 - alpha_cumprod)  # [B]

        sigmas = self.sigma_scheduler.sigmas[batched_t]  # [B]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[batched_t]  # [B]

        # q_sample (x_t | x_0)
        # 1) d3pm for atom types (如果启用原子类型扩散)
        batch_idx = batch.batch
        a_0 = batch.atom_types.long()  # [B_n]，元素是原子序数

        # 原子类型固定：直接使用原始原子类型
        x_t_atom_types = batch.atom_types  # [B_n,]

        # 2) variance-preserving for lattice
        l_0 = batch.lattices
        mask_lattice_matrix = self.mask_lattice_matrix.to(device)
        noise_lattice = torch.randn_like(l_0) * mask_lattice_matrix  # [B, 3, 3]
        # x_t_lattice在DiffCSP中为input_lattice
        x_t_lattice = c0[:, None, None] * l_0 + c1[:, None, None] * noise_lattice

        # 3) variance-exploding for fractional coordinates
        frac_coords = batch.frac_coords
        noise_coords = torch.randn_like(frac_coords)  # [B_n, 3]
        sigmas_per_atom = sigmas[batch_idx][:, None]  # [B_n, 1]
        sigmas_norm_per_atom = sigmas_norm[batch_idx][:, None]  # [B_n, 1]

        # target_coords在DiffCSP中为tar_x
        target_coords = d_log_p_wrapped_normal(
            sigmas_per_atom * noise_coords, sigmas_per_atom
        ) / torch.sqrt(
            sigmas_norm_per_atom
        )  # [B_n, 1]
        # x_t_coords在DiffCSP中为input_frac_coords
        x_t_coords = (frac_coords + sigmas_per_atom * noise_coords) % 1.0

        # conditional inputs (VQ indices from pretrained VQVAE)
        cond_inputs = None
        if self.cond_guide and self.pretrained_vqvae_model is not None:
            # 从 batch 中获取 VQ indices
            if hasattr(batch, 'global_vq_indices') and hasattr(batch, 'local_vq_indices'):
                # 从 batch 中获取预计算的 VQ indices
                indices_g = batch.global_vq_indices
                indices_l = batch.local_vq_indices
            else:
                raise ValueError(
                    "cond_guide=True but batch is missing global_vq_indices/local_vq_indices. "
                    "Please provide DSS/VQ conditions in prepare_batch."
                )
            if indices_g is None or indices_l is None:
                raise ValueError("global_vq_indices/local_vq_indices cannot be None when cond_guide=True")
            cond_inputs = [indices_g, indices_l]
            cond_inputs = list(self.get_vq_vector(cond_inputs, self.cond_drop_prob, device=device))
            if self.property_cond_enabled:
                property_bins = getattr(batch, "property_bins", None)
                prop_cond = self._get_property_cond(property_bins, self.property_cond_drop_prob, device=device)
                if prop_cond is not None:
                    cond_inputs[0] = self._fuse_global_and_property(cond_inputs[0], prop_cond)
                cond_inputs.append(prop_cond)

        # predict noise lattice and fractional coordinates
        decoder_output = self.decoder(
            t=time_emb,
            frac_coords=x_t_coords,
            lattices=x_t_lattice,
            atom_types=x_t_atom_types,
            num_atoms=batch.natoms,
            node2graph=batch_idx,
            cond_inputs=cond_inputs,
        )
        pred_noise_lattice = decoder_output.lattice_out  # [B, 3, 3]
        pred_noise_coords = decoder_output.coords_out  # [B_n, 3]

        # loss for atom_types，此处无

        # loss for lattice
        loss_lattice = F.mse_loss(
            pred_noise_lattice.masked_select(mask_lattice_matrix),
            noise_lattice.masked_select(mask_lattice_matrix),
        )
        # loss for coords
        loss_coords = F.mse_loss(pred_noise_coords, target_coords)

        loss = (
                self.cost_lattice * loss_lattice
                + self.cost_coords * loss_coords
        )
        property_loss = None
        property_logits = None
        if self.property_cond_enabled and self.property_head is not None and hasattr(batch, "property_bins"):
            property_bins = getattr(batch, "property_bins").to(device).long().view(-1)
            valid_mask = property_bins >= 0
            if valid_mask.any():
                graph_features = scatter(decoder_output.node_features, batch_idx, dim=0, reduce='mean')
                property_logits = self.property_head(graph_features)
                property_loss = F.cross_entropy(property_logits[valid_mask], property_bins[valid_mask])
                if self.property_pred_weight > 0.0:
                    loss = loss + self.property_pred_weight * property_loss
        return {
            "loss": loss,
            "true_noise_lattice": noise_lattice.masked_select(mask_lattice_matrix),
            "pred_noise_lattice": pred_noise_lattice.masked_select(mask_lattice_matrix),
            "true_noise_coords": target_coords,
            "pred_noise_coords": pred_noise_coords,
            "property_loss": property_loss,
            "property_logits": property_logits,
        }

    def model_predictions(
        self,
        time_emb: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        batch_natoms: torch.Tensor,
        batch_idx: torch.Tensor,
        cond_scale: float,
        cond_inputs: Optional[List[torch.Tensor]] = None,
        null_cond_inputs: Optional[List[torch.Tensor]] = None,
    ):
        """
        模型预测（支持 classifier-free guidance）
        
        Parameters
        ----------
        t : torch.Tensor
            整数时间步 (batch_size,)
        atom_types : torch.Tensor
            原子类型 (total_atoms,)
        frac_coords : torch.Tensor
            分数坐标 (total_atoms, 3)
        lattices : torch.Tensor
            晶格矩阵 (batch_size, 3, 3)
        batch_natoms : torch.Tensor
            每个结构的原子数 (batch_size,)
        batch_idx : torch.Tensor
            节点到 batch 的映射 (total_atoms,)
        cond_scale : float
            条件缩放因子（用于 classifier-free guidance）
        cond_inputs : List[torch.Tensor], optional
            条件输入 [cond_vq_vector_g, cond_vq_vector_l]
        null_cond_inputs : List[torch.Tensor], optional
            null 条件输入（用于无条件采样）
        """
        if self.cond_guide:
            # conditional sampling
            # print("t", time_emb.shape)
            conditional_decoder_output = self.decoder(
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                t=time_emb,
                cond_inputs=cond_inputs,
            )
            pred_a_cond = conditional_decoder_output.atom_types_out
            pred_l_cond = conditional_decoder_output.lattice_out
            pred_x_cond = conditional_decoder_output.coords_out

            # unconditional sampling
            unconditional_decoder_output = self.decoder(
                t=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                cond_inputs=null_cond_inputs,  # null condition (zeros)
            )
            pred_a_null = unconditional_decoder_output.atom_types_out
            pred_l_null = unconditional_decoder_output.lattice_out
            pred_x_null = unconditional_decoder_output.coords_out

            # conditional free guidance
            # 只有当预测原子类型时才进行加权（diffuse_atom_types=True）
            if self.diffuse_atom_types and pred_a_cond is not None and pred_a_null is not None:
                pred_a = (1 - cond_scale) * pred_a_null + cond_scale * pred_a_cond
            else:
                pred_a = None  # 原子类型固定时不需要预测
            pred_l = (1 - cond_scale) * pred_l_null + cond_scale * pred_l_cond
            pred_x = (1 - cond_scale) * pred_x_null + cond_scale * pred_x_cond
        else:
            decoder_output = self.decoder(
                t=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                cond_inputs=None,
            )
            pred_a = decoder_output.atom_types_out  # 可能是 None（如果 diffuse_atom_types=False）
            pred_l = decoder_output.lattice_out
            pred_x = decoder_output.coords_out
        return pred_a, pred_l, pred_x

    @torch.no_grad()
    def _sample_generator(
        self,
        batch: Batch,
        cond_scale: float = 2.0,
        step_lr: float = 1e-5,
        cond_inputs: Optional[List[torch.Tensor]] = None,
    ):
        """Sample from the model

        Args:
            natoms (Union[int, List[int]]): number of atoms for each sample
                e.g., [10, 20, 30] or 10
            cond_scale (float, optional): scale for conditional sampling.
                if cond_scale > 0.0, use conditional sampling
                Defaults to 2.0.
            step_lr (float, optional): step size for Langevin dynamics.
                Defaults to 1e-5.
        """
        # data_list = [Data(x=torch.zeros(natom, 1), natoms=natom) for natom in natoms]
        # batch = Batch.from_data_list(data_list)
        device = next(self.parameters()).device
        batch = batch.to(device)

        # set properties
        batch_size = batch.num_graphs
        num_nodes = batch.num_nodes
        batch_idx = batch.batch
        batch_natoms = batch.natoms
        mask_lattice_matrix = self.mask_lattice_matrix.to(device)


        # 原子类型固定：直接使用 batch 中的 atom_types（已按 Batch 拼接到 num_nodes 维）
        # 注意：这里不能 repeat(batch_size)，否则长度会错误扩大。
        a_T = batch.atom_types
        l_T = torch.randn(batch_size, 3, 3, device=device) * mask_lattice_matrix
        x_T = torch.randn(num_nodes, 3, device=device)

        time_start = self.beta_scheduler.timesteps

        trajectory_container = TrajectoryContainer(total_steps=time_start)

        # Initialize the first step of the trajectory at t=T
        trajectory_container[time_start] = TrajectoryStep(
            num_atoms=batch_natoms,
            atom_types=a_T,
            frac_coords=x_T % 1.0,
            lattices=l_T,
            batch_idx=batch_idx,
        )

        if self.cond_guide:
            if cond_inputs is None or len(cond_inputs) < 2:
                raise ValueError(
                    "Sampling with cond_guide=True requires cond_inputs=[indices_g, indices_l] "
                    "(optionally plus property_bins)."
                )
            # 此处对传入的离散全局和局部codebook在vq中进行查询返回vq中连续的向量
            # 注意：cond_inputs 应该已经是扩展后的 indices（来自 _expand_for_parallel_sampling）
            # 如果 batch 被扩展为 n_samples 个图，cond_inputs[1] 应该有 n_samples * n_atoms 个元素
            prop_ids = None
            if isinstance(cond_inputs, list) and len(cond_inputs) >= 3:
                prop_ids = cond_inputs[2]
            cond_vq_vectors = list(self.get_vq_vector(
                cond_inputs[:2],
                cond_drop_prob=0.0,
                device=device,
            ))
            null_cond_vq_vectors = list(self.get_vq_vector(
                cond_inputs[:2],
                cond_drop_prob=1.0,
                device=device,
            ))
            if self.property_cond_enabled:
                prop_cond = self._get_property_cond(prop_ids, cond_drop_prob=0.0, device=device)
                null_prop_cond = self._get_property_cond(prop_ids, cond_drop_prob=1.0, device=device)
                cond_vq_vectors[0] = self._fuse_global_and_property(cond_vq_vectors[0], prop_cond)
                null_cond_vq_vectors[0] = self._fuse_global_and_property(null_cond_vq_vectors[0], null_prop_cond)
        else:
            cond_vq_vectors = None
            null_cond_vq_vectors = None
        # print("cond_vq_vectors:", cond_vq_vectors[0].shape, cond_vq_vectors[1].shape)
        # 这里不再使用 tqdm，避免在评估时产生大量多行进度条
        for t in tqdm(range(time_start, 0, -1)):
            batched_t = torch.full((batch_size,), t, dtype=torch.long, device=device)
            time_emb = self.time_embed(batched_t)
            # print("time_emb:", time_emb.shape)
            a_t = trajectory_container[t].atom_types
            x_t = trajectory_container[t].frac_coords
            l_t = trajectory_container[t].lattices

            ### Predictor ###
            pred_a, pred_l, pred_x = self.model_predictions(
                time_emb=time_emb,
                atom_types=a_t,
                frac_coords=x_t,
                lattices=l_t,
                batch_natoms=batch_natoms,
                batch_idx=batch_idx,
                cond_scale=cond_scale,
                cond_inputs=cond_vq_vectors,
                null_cond_inputs=null_cond_vq_vectors,
            )

            # 原子类型固定：保持不变
            a_t_minus_1 = a_t
            # update the state for lattice
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_l = rand_l * mask_lattice_matrix
            l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l
            l_t_minus_1 = l_t_minus_1 * mask_lattice_matrix
            # clip the lattice matrix when t = T (for the first step of prediction)
            # otherwise, the lattice matrix could be diverged when the first prediction is too large
            if t == time_start:
                l_t_minus_1 = l_t_minus_1.clip(-6, 6)
            # update the state for coords (0 -> 0.5 step)
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = torch.sqrt(
                (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2))
                / (sigma_x**2)
            )
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            ### Corrector ###
            _, _, pred_x = self.model_predictions(
                time_emb=time_emb,
                atom_types=a_t_minus_1,
                frac_coords=x_t_minus_05,
                lattices=l_t_minus_1,
                batch_natoms=batch_natoms,
                batch_idx=batch_idx,
                cond_scale=cond_scale,
                cond_inputs=cond_vq_vectors,
                null_cond_inputs=null_cond_vq_vectors,
            )

            # update the state for coords (0.5 -> 1 step)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            trajectory_step = TrajectoryStep(
                num_atoms=batch_natoms,
                atom_types=a_t_minus_1,
                frac_coords=x_t_minus_1 % 1.0,
                lattices=l_t_minus_1,
                batch_idx=batch_idx,
            )
            trajectory_container[t - 1] = trajectory_step
            yield trajectory_container.get_atoms(t=t - 1)

    def _expand_for_parallel_sampling(
        self,
        batch: Batch,
        cond_inputs: Optional[List[torch.Tensor]],
        n_samples: int,
    ) -> tuple[Batch, Optional[List[torch.Tensor]]]:
        """
        将单个结构的 Batch 扩展为 n_samples 份（并行采样）。

        约定（与 0.0.1 一致）：
        - cond_inputs[0] (indices_g) 长度应为 batch_size
        - cond_inputs[1] (indices_l) 长度应为 num_nodes

        若输入为单结构（batch.num_graphs==1）且 n_samples>1，则复制该图 n_samples 次；
        cond_inputs 若为单样本（indices_g 长度 1），则自动 repeat 到 n_samples；
        indices_l 若为单图长度（n_atoms），则自动 repeat 到 n_samples*n_atoms。
        """
        if n_samples is None or n_samples <= 1:
            return batch, cond_inputs

        if batch.num_graphs != 1:
            # 已经是多图 batch，就认为调用方已组织好并行采样所需的 batch/cond_inputs
            return batch, cond_inputs

        # 复制 Data n_samples 份，构造并行 Batch
        data_list = batch.to_data_list()
        if len(data_list) != 1:
            return batch, cond_inputs
        base_data = data_list[0]
        # torch_geometric 的 Batch/Data 没有 .device 属性；用模型 device 更稳
        device = next(self.parameters()).device
        expanded_batch = Batch.from_data_list([base_data.clone() for _ in range(n_samples)]).to(device)

        if cond_inputs is None:
            return expanded_batch, None

        if not (isinstance(cond_inputs, list) and len(cond_inputs) >= 2):
            raise ValueError("cond_inputs 必须是 [indices_g, indices_l] 或 [indices_g, indices_l, property_bins] 或 None")

        indices_g, indices_l = cond_inputs[0], cond_inputs[1]
        property_bins = cond_inputs[2] if len(cond_inputs) > 2 else None
        
        # indices_g: 检查是否是单个样本的全局索引
        # 可能的情况：
        # 1. (1,) - 单个样本，单个量化器
        # 2. (num_quantizers,) - 单个样本，多个量化器（展平后）
        # 3. (1, num_quantizers) - 单个样本，多个量化器
        # 
        # 判断标准：如果 indices_g 的第一维等于 1，说明是单个样本，需要扩展
        # 如果 indices_g 的第一维等于 n_samples，说明已经扩展过了，不需要再扩展
        # 如果 indices_g 的第一维不等于 1 且不等于 n_samples，可能是单个样本的多个量化器，需要扩展
        
        indices_g_needs_expansion = False
        expected_expanded_size = n_samples
        
        if indices_g.shape[0] == 1:
            # 第一维为 1，说明是单个样本
            indices_g_needs_expansion = True
        elif indices_g.shape[0] != expected_expanded_size:
            # 第一维不等于 1 且不等于 n_samples，可能是单个样本的多个量化器
            # 在这种情况下，我们需要扩展
            # 如果 indices_g 是 1D 的 (num_quantizers,)，需要先 reshape 为 (1, num_quantizers)，然后扩展
            # 如果 indices_g 是 2D 的 (1, num_quantizers)，直接扩展
            indices_g_needs_expansion = True
        
        if indices_g_needs_expansion:
            # 保持原始维度，只在第一维 repeat
            if indices_g.dim() == 1:
                # 1D: (num_quantizers,) -> (n_samples, num_quantizers)
                # 优先使用模型的 num_quantizers_global，如果无法获取则使用形状推断
                if self.pretrained_vqvae_model is not None:
                    num_quantizers_g = getattr(self.pretrained_vqvae_model, 'num_quantizers_global', None)
                    if num_quantizers_g is not None and indices_g.shape[0] == num_quantizers_g:
                        num_quantizers = num_quantizers_g
                    else:
                        num_quantizers = indices_g.shape[0]  # fallback: 使用形状推断
                else:
                    num_quantizers = indices_g.shape[0]  # fallback: 使用形状推断
                # reshape 为 (1, num_quantizers)，然后扩展为 (n_samples, num_quantizers)
                indices_g = indices_g.view(1, num_quantizers).repeat(n_samples, 1)
            else:
                # 2D: (1, num_quantizers) -> (n_samples, num_quantizers)
                indices_g = indices_g.repeat(n_samples, *([1] * (indices_g.dim() - 1)))
        
        # indices_l: 检查是否是单个结构的 indices
        # 关键：如果 batch 被扩展了（从 1 个图扩展到 n_samples 个图），
        # 那么 indices_l 也应该从单个结构的原子数扩展到 n_samples * 单个结构的原子数
        # 
        # 判断标准：如果 indices_l 的第一维等于 base_data.natoms，说明是单个结构，需要扩展
        # 如果 indices_l 的第一维等于 n_samples * base_data.natoms，说明已经扩展过了，不需要再扩展
        
        # 判断 indices_l 是否需要扩展
        # 关键：如果 indices_l 的第一维等于 base_data.natoms，说明是单个结构，需要扩展
        # 如果 indices_l 的第一维等于 n_samples * base_data.natoms，说明已经扩展过了，不需要再扩展
        # 如果 indices_l 的第一维不等于 base_data.natoms 且不等于 n_samples * base_data.natoms，
        # 可能是展平后的形状（n_atoms * num_quantizers），需要根据实际情况处理
        
        indices_l_needs_expansion = False
        expected_expanded_size = n_samples * base_data.natoms
        
        if indices_l.dim() == 1:
            # 1D: 检查第一维是否等于 base_data.natoms（单个结构的原子数）
            if indices_l.shape[0] == base_data.natoms:
                indices_l_needs_expansion = True
            # 如果第一维不等于 base_data.natoms，但也不等于 expected_expanded_size，
            # 可能是展平后的形状（n_atoms * num_quantizers），需要检查是否能被 base_data.natoms 整除
            elif indices_l.shape[0] != expected_expanded_size:
                # 尝试检查是否是展平后的形状：如果能被 base_data.natoms 整除，可能是 (n_atoms * num_quantizers,)
                # 优先使用模型的 num_quantizers_local，如果无法获取则使用形状推断
                if indices_l.shape[0] % base_data.natoms == 0:
                    inferred_num_quantizers = indices_l.shape[0] // base_data.natoms
                    if self.pretrained_vqvae_model is not None:
                        num_quantizers_l = getattr(self.pretrained_vqvae_model, 'num_quantizers_local', None)
                        if num_quantizers_l is not None and inferred_num_quantizers == num_quantizers_l:
                            num_quantizers = num_quantizers_l
                        else:
                            num_quantizers = inferred_num_quantizers  # fallback: 使用形状推断
                    else:
                        num_quantizers = inferred_num_quantizers  # fallback: 使用形状推断
                    # reshape 回 (n_atoms, num_quantizers)
                    indices_l = indices_l.view(base_data.natoms, num_quantizers)
                    indices_l_needs_expansion = True
        elif indices_l.dim() == 2:
            # 2D: 检查第一维是否等于 base_data.natoms
            if indices_l.shape[0] == base_data.natoms:
                indices_l_needs_expansion = True
        
        if indices_l_needs_expansion:
            # 保持原始维度，只在第一维 repeat
            if indices_l.dim() == 1:
                indices_l = indices_l.repeat(n_samples)
            else:
                indices_l = indices_l.repeat(n_samples, *([1] * (indices_l.dim() - 1)))

        if property_bins is not None:
            property_bins = torch.as_tensor(property_bins, device=device).view(-1)
            if property_bins.numel() == 1:
                property_bins = property_bins.repeat(n_samples)
            elif property_bins.numel() != n_samples:
                property_bins = property_bins[:1].repeat(n_samples)
            return expanded_batch, [indices_g, indices_l, property_bins]

        return expanded_batch, [indices_g, indices_l]

    def sample(
        self,
        batch: Batch,
        cond_inputs: Optional[List[torch.Tensor]] = None,
        n_samples: int = 1,
        cond_scale: float = 2.0,
        step_lr: float = 1e-5,
        return_trajectory: bool = False,
        stream: bool = False,
    ):
        """
        从模型采样
        
        Parameters
        ----------
        cond_inputs : List[torch.Tensor], optional
            条件输入 [indices_g, indices_l]
            - indices_g: (n_samples,) 全局 VQ indices
            - indices_l: (total_atoms,) 局部 VQ indices（可选，如果为 None 则使用零）
        cond_scale : float
            条件缩放因子（用于 classifier-free guidance）
        step_lr : float
            Langevin dynamics 的步长
        return_trajectory : bool
            是否返回完整轨迹
        stream : bool
            是否流式返回（生成器）
        """
        # 支持单结构 + n_samples 并行（对齐 0.0.1 的 sample 行为）
        batch, cond_inputs = self._expand_for_parallel_sampling(batch, cond_inputs, n_samples=n_samples)
        
        # 为了兼容性，暂时使用 None 作为 texts（_sample_generator 会处理）
        if stream:
            # _sample_generator的作用是对一个样本一个cond_inputs进行处理
            return self._sample_generator(
                batch=batch,
                cond_scale=cond_scale,
                step_lr=step_lr,
                cond_inputs=cond_inputs,
            )
        else:
            trajectory = list(
                self._sample_generator(
                    batch=batch,
                    cond_scale=cond_scale,
                    step_lr=step_lr,
                    cond_inputs=cond_inputs,
                )
            )
            if return_trajectory:
                return trajectory
            else:
                return trajectory[-1] if len(trajectory) > 0 else None
