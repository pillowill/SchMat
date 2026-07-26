import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat
from utils.cspnet_utils import radius_graph_pbc, repeat_blocks

MAX_ATOMIC_NUM = 100
DECODER_OUTPUTS = namedtuple(
    "DECODER_OUTPUTS", ["atom_types_out", "lattice_out", "coords_out", "node_features"]
)

class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SinusoidsEmbedding(nn.Module):
    "Embedding for periodic distance features."

    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

class FilmLayer(nn.Module):
    """FiLM layer for efficient incorporation of time or conditional embeddings."""

    def __init__(
            self,
            hidden_dim=128,
            time_dim=256,
            cond_dim=128,
            act_fn=nn.SiLU(),
            is_global=False,
    ):
        super(FilmLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.text_dim = cond_dim
        self.act_fn = act_fn
        self.mlp_cond = nn.Sequential(
            nn.Linear(time_dim + cond_dim, hidden_dim * 2),
            act_fn,
        )
        # block
        if is_global:
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, time_embeds=None, cond_embeds=None):
        # Feature-wise Linear Modulation
        if time_embeds is not None and cond_embeds is not None:
            cond_emb = self.mlp_cond(torch.cat([time_embeds, cond_embeds], dim=1))

        elif time_embeds is not None:
            cond_emb = self.mlp_cond(time_embeds)
        elif cond_embeds is not None:
            cond_emb = self.mlp_cond(cond_embeds)
        else:
            raise ValueError("Either time or text embeddings must be provided.")
        scale, shift = cond_emb.chunk(2, dim=1)

        # residual block
        x_init = x.clone()
        x = self.proj(x)
        x = self.norm(x)
        x = x * scale + shift
        x = self.act_fn(x)
        x = x + x_init
        return x


class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
            self,
            hidden_dim=128,
            act_fn=nn.SiLU(),
            dis_emb=None,
            ln=False,
            ip=True
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[0], dim=0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output

class CSPNet(nn.Module):

    def __init__(
            self,
            hidden_dim=128,
            cond_dim=256,
            time_dim=256,
            num_layers=4,
            max_atoms=100,
            act_fn='silu',
            dis_emb='sin',
            num_freqs=10,
            edge_style='fc',
            cutoff=6.0,
            max_neighbors=20,
            ln=False,
            ip=True,
            smooth=False,
            cond_guided=False,
            pred_atom_types=False,
            use_multiscale_film=False,
            local_film_end_layer=None,
            global_film_start_layer=None,
            hds_condition_mode="full",
    ):
        super(CSPNet, self).__init__()

        self.ip = ip
        self.smooth = smooth
        self.max_atoms = max_atoms  # 保存以便 one-hot 时使用
        self.cond_guided = cond_guided
        self.use_multiscale_film = bool(use_multiscale_film)
        valid_hds_condition_modes = {"full", "global_only", "local_only", "none"}
        self.hds_condition_mode = str(hds_condition_mode).strip().lower()
        if self.hds_condition_mode not in valid_hds_condition_modes:
            raise ValueError(
                "hds_condition_mode must be one of "
                f"{sorted(valid_hds_condition_modes)}, got {hds_condition_mode!r}"
            )
        # The complete HDS is still decoded and looked up before CSPNet. These
        # flags only gate the final FiLM injections at the diffusion interface.
        self.use_global_hds_condition = self.hds_condition_mode in {"full", "global_only"}
        self.use_local_hds_condition = self.hds_condition_mode in {"full", "local_only"}
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        if time_dim > 0 or cond_dim > 0:
            # node_features 在通过 atom_latent_emb 后是 hidden_dim 维
            self.local_film_layer = FilmLayer(hidden_dim, time_dim, hidden_dim)
            self.global_film_layer = FilmLayer(hidden_dim, time_dim, hidden_dim)
            self.local_film_layers = nn.ModuleList(
                [FilmLayer(hidden_dim, time_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.global_film_layers = nn.ModuleList(
                [FilmLayer(hidden_dim, time_dim, hidden_dim, is_global=True) for _ in range(num_layers)]
            )
        if self.cond_guided:
            # 使用 Linear 层将预训练模型的 cond_dim 投影到 hidden_dim
            # 输入：从预训练模型获取的连续向量 (hidden_dim)
            # 输出：条件向量 (cond_dim)
            self.global_sem_proj = nn.Linear(cond_dim, hidden_dim)
            self.local_sem_proj = nn.Linear(cond_dim, hidden_dim)
        self.coords_embedding = nn.Linear(3, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim * 2, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )
        self.num_layers = num_layers
        if local_film_end_layer is None:
            local_film_end_layer = max(1, num_layers // 3)
        if global_film_start_layer is None:
            global_film_start_layer = max(0, num_layers // 2)
        self.local_film_end_layer = int(local_film_end_layer)
        self.global_film_start_layer = int(global_film_start_layer)
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)
        # type_out 的创建应该同时依赖于 cond_guided 和 pred_atom_types
        # 只有当需要预测原子类型时才创建（无论是否有条件引导）
        if pred_atom_types:
            self.type_out = nn.Linear(hidden_dim, max_atoms)
        else:
            self.type_out = None
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.pred_atom_types = pred_atom_types

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
            self, edge_index, cell_offsets, neighbors, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
                (cell_offsets[:, 0] < 0)
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                | (
                        (cell_offsets[:, 0] == 0)
                        & (cell_offsets[:, 1] == 0)
                        & (cell_offsets[:, 2] < 0)
                )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == 'fc':
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)

            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, None, None, num_atoms, self.cutoff, self.max_neighbors,
                device=num_atoms.device, lattices=lattices)

            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds,
                                                                                 distance_vectors)

            return edge_index_new, -edge_vector_new

    def forward(self, atom_types, frac_coords, lattices, num_atoms, node2graph, t=None, cond_inputs=None):
        """
        前向传播
        
        Parameters
        ----------
        atom_types : Tensor
            原子类型 (total_atoms,) 或 (total_atoms, max_atoms) one-hot
        frac_coords : Tensor
            分数坐标 (total_atoms, 3)
        lattices : Tensor
            晶格矩阵 (batch_size, 3, 3)
        num_atoms : Tensor
            每个结构的原子数 (batch_size,)
        node2graph : Tensor
            节点到 batch 的映射 (total_atoms,)
        t : Tensor, optional
            时间嵌入 (batch_size, time_dim) 或整数时间步 (batch_size,)
        cond_inputs : List[Tensor], optional
            条件输入 [cond_vq_vector_g, cond_vq_vector_l]
            - cond_vq_vector_g: (batch_size, hidden_dim) 全局 VQ 向量
            - cond_vq_vector_l: (total_atoms, hidden_dim) 局部 VQ 向量
        """
        # 此处cond_inputs传入的已经是[cond_vq_vector_g, cond_vq_vector_l]，所以不需要再进行转换
        cond_global_semantic = None
        cond_local_semantic = None
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)  # []
        else:
            # Catch invalid atom indices early; CUDA device asserts from embedding
            # can otherwise surface at unrelated downstream ops.
            if atom_types.numel() > 0:
                at_min = int(atom_types.min().detach().cpu().item())
                at_max = int(atom_types.max().detach().cpu().item())
                if at_min < 1 or at_max > self.max_atoms:
                    raise ValueError(
                        f"atom_types out of range: min={at_min}, max={at_max}, expected in [1, {self.max_atoms}]"
                    )
            node_features = self.node_embedding(atom_types-1)

        coords_embedding = self.coords_embedding(frac_coords)

        # 此处处理时间嵌入：
        # 1. 当时间嵌入不为空，也就是当前模型的模式为晶体结构扩散建模，那么按照原DiffCSP的方式处理，即将时间嵌入扩展到每个原子，并拼接到节点特征中
        # 2. 当时间嵌入为空，也就是当前模型模式为VQVAE对晶体结构建模，那么按照原VQCrystal的方式处理，即将坐标与原子进行拼接
        if t is not None:
            t_per_atom = t.repeat_interleave(num_atoms, dim=0)  # (total_atoms,)
            node_features = torch.cat([node_features, t_per_atom], dim=1)
        else:
            node_features = torch.cat([node_features, coords_embedding], dim=-1)
            t_per_atom = None

        node_features = self.atom_latent_emb(node_features)  # (batch size, hidden_dim)


        if cond_inputs is not None and self.cond_guided:
            # cond_inputs: [cond_vq_vector_g, cond_vq_vector_l] VQ codebook中每个indices对应的连续向量作为条件
            cond_vq_vector_g, cond_vq_vector_l = cond_inputs[:2]
            # cond_vq_vector_g: (batch_size, vq_dim), cond_vq_vector_l: (total_atoms, vq_dim)
            
            # 使用 Linear 层将连续向量投影到cond_dim
            # 这样可以利用预训练模型的语义信息，同时学习到适合扩散模型的条件表示
            cond_global_semantic = self.global_sem_proj(cond_vq_vector_g)  # (batch_size, cond_dim)
            cond_local_semantic = self.local_sem_proj(cond_vq_vector_l)  # (total_atoms, cond_dim)
        """
        接下来分别对VQVAE和Diffusion进行建模
        """
        # 1. 条件建模与Diffusion建模进行结合(重点)，后续需要完成无条件的Diffusion
        # 基础局部 FiLM（兼容旧行为）
        if (
            self.use_local_hds_condition
            and (not self.use_multiscale_film)
            and t_per_atom is not None
            and cond_local_semantic is not None
        ):
            node_features = self.local_film_layer(
                node_features, time_embeds=t_per_atom, cond_embeds=cond_local_semantic
            )
        for i in range(0, self.num_layers):
            if (
                self.use_local_hds_condition
                and self.use_multiscale_film
                and t_per_atom is not None
                and cond_local_semantic is not None
            ):
                if i < self.local_film_end_layer:
                    node_features = self.local_film_layers[i](
                        node_features, time_embeds=t_per_atom, cond_embeds=cond_local_semantic
                    )
            node_features = self._modules[f"csp_layer_{i}"](
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff
            )

            if (
                self.use_global_hds_condition
                and self.use_multiscale_film
                and t is not None
                and cond_global_semantic is not None
            ):
                if i >= self.global_film_start_layer:
                    graph_features_i = scatter(node_features, node2graph, dim=0, reduce='mean')
                    graph_features_i = self.global_film_layers[i](
                        graph_features_i, time_embeds=t, cond_embeds=cond_global_semantic
                    )
                    node_features = node_features + graph_features_i[node2graph]

        local_features = node_features

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # 应用 FiLM 层进行条件注入到全局特征，同样对应于条件扩散
        if t_per_atom is not None:
            coord_out = self.coord_out(node_features)
            graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')


            if (
                self.cond_guided
                and self.use_global_hds_condition
                and (not self.use_multiscale_film)
                and cond_global_semantic is not None
            ):
                graph_features = self.global_film_layer(
                    graph_features, time_embeds=t, cond_embeds=cond_global_semantic
                )
            lattice_out = self.lattice_out(graph_features)
            lattice_out = lattice_out.view(-1, 3, 3)
            if self.ip:
                lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
            if self.pred_atom_types:
                type_out = self.type_out(node_features)
            else:
                type_out = None

            return DECODER_OUTPUTS(
                atom_types_out=type_out,  # [B_n, max_atoms]
                lattice_out=lattice_out,  # [B, 3, 3]
                coords_out=coord_out,  # [B_n, 3]
                node_features=node_features,  # [B_n, hidden_dim]
            )
        else:
            graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')
            return local_features, graph_features
