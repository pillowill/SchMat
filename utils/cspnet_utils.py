# -*- coding: utf-8 -*-
"""
VQCrystal model utility functions.
Copied from VQCrystal-main/utils.py for integration.
"""

import torch
import numpy as np
from torch_scatter import segment_coo, segment_csr
from pymatgen.core.lattice import Lattice

def load_pretrained_cryvqvae(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Additional utility functions needed by model2.py
def argmax_with_custom_value(tensor, custom_value=-1):
    equal_mask = torch.all(tensor == tensor[:, :, 0].unsqueeze(-1), dim=-1)
    result = torch.where(equal_mask, torch.full_like(equal_mask, custom_value, dtype=torch.int64), torch.argmax(tensor, dim=-1))
    return result


def tensor_to_list_based_on_mask(tensor, mask):
    b_size = tensor.shape[0] 
    result_list = []  
    for b in range(b_size): 
        selected_tensor = tensor[b][mask[b]] 
        result_list.append(selected_tensor)  
    return result_list


def int_to_element(lst):
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm']
    return [elements[i] for i in lst]


def batch_lattice_to_matrix_torch(params):
    B = params.size(0)
    matrices = []
    for i in range(B):
        a, b, c, alpha, beta, gamma = params[i].tolist()
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        matrix = torch.tensor(lattice.matrix, dtype=torch.float32)
        matrices.append(matrix)
    matrices_tensor = torch.stack(matrices, dim=0).to(params.device)
    return matrices_tensor


def custom_sort(tensor):
    indices = np.arange(len(tensor))
    tuples = [(tuple(vector), idx) for idx, vector in zip(indices, tensor)]
    tuples.sort()
    sorted_indices = [idx for _, idx in tuples]
    sorted_tensor = tensor[sorted_indices]
    return sorted_tensor


def crystal_position_constrain(tensor):
    transformed_tensor = tensor.clone()  
    shape = transformed_tensor.shape
    tensor_flat = transformed_tensor.reshape(-1)
    for idx in range(tensor_flat.shape[0]):
        value = tensor_flat[idx]
        if value > 0.9:
            tensor_flat[idx] = 1
        elif value < 0.1:
            tensor_flat[idx] = 0
        elif 0.45 <= value <= 0.55:
            tensor_flat[idx] = 0.5
    transformed_tensor = tensor_flat.reshape(shape)
    return transformed_tensor


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
    lattices=None
):
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattices, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc(pos, lengths, angles, natoms, radius, max_num_neighbors_threshold, device, lattices=None):
    
    batch_size = len(natoms)
    if lattices is None:
        cell = lattice_params_to_matrix_torch(lengths, angles)
    else:
        cell = lattices
    atom_pos = pos

    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1,1)

    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1,1)
    
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1,1)
    
    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1)
    
    unit_cell_all = []
    num_cells_all = []

    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    
    unit_cell = torch.cat([_.reshape(-1,1) for _ in torch.meshgrid(cells_per_dim, indexing='ij')], dim=-1)
    
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom

    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    radius_real = (min_dist.min(dim=-1)[0] + 0.01)
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)

    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    if max_num_neighbors_threshold is not None:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)
            
    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        image_indptr = torch.zeros(
            natoms.shape[0] + 1, device=device, dtype=torch.long
        )
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True,
    lattices = None
):
    if regularized:
        frac_coords = frac_coords % 1.
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def get_max_neighbors_mask(
    natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    max_num_neighbors = max_num_neighbors_threshold
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    distance_real_cutoff = distance_sort[:,max_num_neighbors_threshold].reshape(-1,1).expand(-1,max_num_neighbors) + 0.01
    
    mask_distance = distance_sort < distance_real_cutoff
    
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite & mask_distance)
    
    num_neighbor_per_node = (mask_finite & mask_distance).sum(dim=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)
    
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    N = (sizes * repeats).sum()

    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1

        insert_val[idx] += block_inc

    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    id_ar[0] += start_idx

    res = id_ar.cumsum(0)
    return res

