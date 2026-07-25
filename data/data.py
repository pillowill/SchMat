import functools
import json
import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from utils.data_utils import atoms_to_pyg_data
import pandas as pd
from pymatgen.io.cif import CifParser
from collections import Counter
from io import StringIO
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError
from pymatgen.core.lattice import Lattice
from pymatgen.core import Structure


def get_train_val_test_loader(
        dataset,
        collate_fn=default_collate,
        batch_size=64,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        return_test=False,
        num_workers=1,
        pin_memory=False,
        **kwargs,
):
    """
    从单个数据集按比例划分训练集、验证集和测试集。
    注意：如果数据已经分为三个独立的CSV文件，请使用 get_separate_loaders 函数。
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_size = total_size - train_size - valid_size
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size): -test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def get_separate_loaders(
        root_dir,
        collate_fn=default_collate,
        train_csv='train.csv',
        val_csv='val.csv',
        test_csv='test.csv',
        proper='formation_energy',
        cond_guide=False,
        train_batch_size=64,
        val_batch_size=64,
        test_batch_size=64,
        return_test=True,
        num_workers=1,
        pin_memory=False,
        random_seed=123,
        **kwargs,
):
    """
    从三个独立的CSV文件分别加载训练集、验证集和测试集。
    
    参数
    ----
    root_dir : str
        数据根目录，包含 train.csv, val.csv, test.csv
    collate_fn : callable
        批处理函数，默认为 default_collate
    train_csv : str
        训练集CSV文件名，默认为 'train.csv'
    val_csv : str
        验证集CSV文件名，默认为 'val.csv'
    test_csv : str
        测试集CSV文件名，默认为 'test.csv'
    proper : str
        性质名称，默认为 'formation_energy'
    cond_guide : bool
        是否使用条件引导（用于条件扩散模型），默认为 False
    train_batch_size : int
        训练集批次大小，默认为 64
    val_batch_size : int
        验证集批次大小，默认为 64
    test_batch_size : int
        测试集批次大小，默认为 64
    return_test : bool
        是否返回测试集DataLoader，默认为 True
    num_workers : int
        数据加载的worker数量，默认为 1
    pin_memory : bool
        是否使用pin_memory，默认为 False
    random_seed : int
        随机种子，用于打乱训练集，默认为 123
    
    返回
    ----
    train_loader : DataLoader
        训练集DataLoader
    val_loader : DataLoader
        验证集DataLoader
    test_loader : DataLoader (可选)
        测试集DataLoader（当 return_test=True 时返回）
    """
    # 创建训练集数据集（需要打乱）
    train_dataset = CIFData(
        root_dir=root_dir,
        proper=proper,
        cond_guide=cond_guide,
        random_seed=random_seed,
        csv_file=train_csv,
        shuffle=True,  # 训练集需要打乱
    )
    
    # 创建验证集数据集（不需要打乱）
    val_dataset = CIFData(
        root_dir=root_dir,
        proper=proper,
        cond_guide=cond_guide,
        random_seed=random_seed,
        csv_file=val_csv,
        shuffle=False,  # 验证集不需要打乱
    )
    
    # 创建训练集DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,  # DataLoader级别也打乱
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,  # 训练时丢弃最后一个不完整的batch
    )
    
    # 创建验证集DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    
    if return_test:
        # 创建测试集数据集（不需要打乱）
        test_dataset = CIFData(
            root_dir=root_dir,
            proper=proper,
            cond_guide=cond_guide,
            random_seed=random_seed,
            csv_file=test_csv,
            shuffle=False,  # 测试集不需要打乱
        )
        
        # 创建测试集DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,  # 测试集不打乱
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        
        print(f"数据集加载完成:")
        print(f"  训练集: {len(train_dataset)} 个样本")
        print(f"  验证集: {len(val_dataset)} 个样本")
        print(f"  测试集: {len(test_dataset)} 个样本")
        
        return train_loader, val_loader, test_loader
    else:
        print(f"数据集加载完成:")
        print(f"  训练集: {len(train_dataset)} 个样本")
        print(f"  验证集: {len(val_dataset)} 个样本")
        
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    将 CIFData 返回的单个样本 (data_dict, cif_id) 列表整理成一个 batch。

    输入
    ----
    dataset_list: List[Tuple[data_dict, cif_id]]
        每个元素来自 CIFData.__getitem__(idx)，形式为 (data_dict, cif_id)，其中：
        - data_dict['property']          : 标量性质（如 formation_energy），torch.tensor(1,)
        - data_dict['global_vq_targets'] : 标量全局 VQ 目标（当前为空间群号），当 cond_guide=False 时
        - data_dict['local_vq_targets']  : 长度为 N_i 的列表，每个元素是一个 dict，包含
                                           - "cn"                : 配位数
                                           - "neighbor_elements" : 邻居元素统计（已归一化）
                                          当 cond_guide=False 时
        - data_dict['pyg_data']          : PyG 数据对象，当 cond_guide=True 时

    输出
    ----
    result_dict: dict
        - 'property'          : (B,)  堆叠后的标量性质
        - 'global_vq_targets' : (B,)  每个结构一个标量的全局 VQ 目标（如空间群号），当 cond_guide=False 时
        - 'local_vq_targets'  : List[List[dict]]，保留 Python 嵌套列表形式，方便后续自定义处理
                                当 cond_guide=False 时

    batch_cif_ids: List[str]
        每个样本的 cif_id 列表，用于追踪/分析。
    """
    batch_cif_ids = []  # 每个样本的 ID
    batch_proper = []  # 每个样本的标量性质
    pyg_data_list = []
    global_vq_targets = []  # 每个样本的全局 VQ 目标（标量）
    local_vq_targets = []  # 每个样本的局部 VQ 目标（列表[list[dict]]）

    # ------------------------------------------------------------
    # 逐个样本处理：收集结构、性质、VQ 目标等信息
    # ------------------------------------------------------------
    for i, (data_dict, cif_id) in enumerate(dataset_list):
        proper = data_dict['property']  # 标量性质，torch.tensor(1,)
        # 获取 VQ 目标（当 cond_guide=False 时存在）
        global_vq_target = data_dict.get('global_vq_targets')  # 可能为 None（当 cond_guide=True）
        local_vq_target = data_dict.get('local_vq_targets')  # 可能为 None（当 cond_guide=True）
        pyg_data = data_dict.get('pyg_data')

        # --------------------------------------------------------
        # 收集其他信息：性质、结构、VQ 目标等
        # --------------------------------------------------------
        batch_proper.append(proper)  # 这里 proper 已经是 torch.tensor(1,)
        batch_cif_ids.append(cif_id)
        pyg_data_list.append(pyg_data)
        # 只有当存在时才添加（cond_guide=False 时）
        if global_vq_target is not None:
            global_vq_targets.append(global_vq_target)
        if local_vq_target is not None:
            local_vq_targets.append(local_vq_target)


    result_dict = {
        'pyg_data': pyg_data_list,
        # 标量属性，stack 成形状 (B,) 的张量
        'property': torch.stack(batch_proper),
    }

    # 只有当存在时才添加（cond_guide=False 时）
    if len(global_vq_targets) > 0:
        # 全局 VQ 目标是标量（如空间群号），直接转成 (B,) 张量
        result_dict['global_vq_targets'] = torch.tensor(global_vq_targets)

    if len(local_vq_targets) > 0:
        # 局部 VQ 目标是"每个原子一个 dict"的嵌套列表，保持 Python 结构
        result_dict['local_vq_targets'] = local_vq_targets

    # 返回 batch 数据和对应的 cif_id 列表
    return result_dict, batch_cif_ids


class CIFData(Dataset):
    def __init__(
            self,
            root_dir,
            proper='formation_energy',
            cond_guide=False,
            random_seed=123,
            csv_file='data.csv',  # 新增参数：指定CSV文件名，默认为'data.csv'保持向后兼容
            shuffle=True,  # 新增参数：是否打乱数据，验证集和测试集通常不需要打乱
            skip_bad_cif=True,
            bad_cif_log_file=None,
    ):
        self.root_dir = root_dir
        self.cond_guide = cond_guide  # 当训练条件扩散模型时启用
        self.proper = proper
        if self.proper == 'formation_energy':
            self.proper = 'formation_energy_per_atom'
        assert os.path.exists(root_dir), "root_dir does not exist!"
        
        csv_path = os.path.join(root_dir, csv_file)
        assert os.path.exists(csv_path), f"CSV file does not exist: {csv_path}"
        self.id_prop_data = pd.read_csv(csv_path)
        self.id_prop_data = self.id_prop_data.to_dict(orient='records')

        # 只在需要时打乱数据（训练集需要，验证集和测试集不需要）
        if shuffle:
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)

        # Guard rail: avoid training crash when a few CIFs have unstable symmetry.
        self._symmetry_fallback_count = 0
        self._symmetry_fallback_print_limit = 20
        self.skip_bad_cif = bool(skip_bad_cif)
        self._bad_cif_logged = set()
        if bad_cif_log_file is None:
            self.bad_cif_log_file = os.path.join(root_dir, f"bad_cif_{csv_file}.log")
        else:
            self.bad_cif_log_file = bad_cif_log_file

    def _log_bad_cif(self, cif_id: str, reason: str):
        key = f"{cif_id}\t{reason}"
        if key in self._bad_cif_logged:
            return
        self._bad_cif_logged.add(key)
        try:
            with open(self.bad_cif_log_file, "a", encoding="utf-8") as f:
                f.write(f"{cif_id}\t{reason}\n")
        except Exception:
            pass

    def calculate_global_vq_target(self, structure, cif_id=None):
        # CE 分类标签要求 [0, C-1]，空间群号为 [1, 230]
        try:
            analyzer = SpacegroupAnalyzer(structure)
            space_group_number = analyzer.get_space_group_number()
            return max(0, int(space_group_number) - 1)
        except SymmetryUndeterminedError:
            # Rare malformed / near-singular structures can fail symmetry detection.
            self._symmetry_fallback_count += 1
            if cif_id is not None:
                self._log_bad_cif(str(cif_id), "SymmetryUndeterminedError")
            if self._symmetry_fallback_count <= self._symmetry_fallback_print_limit:
                print(
                    f"[CIFData] SymmetryUndeterminedError -> fallback SG label=0 "
                    f"(count={self._symmetry_fallback_count})"
                )
            return 0
        except Exception:
            self._symmetry_fallback_count += 1
            if cif_id is not None:
                self._log_bad_cif(str(cif_id), "SpacegroupAnalyzerError")
            if self._symmetry_fallback_count <= self._symmetry_fallback_print_limit:
                print(
                    f"[CIFData] SpacegroupAnalyzer failed -> fallback SG label=0 "
                    f"(count={self._symmetry_fallback_count})"
                )
            return 0

    def calculate_local_vq_target(self, structure):
        # 计算每个原子的coordination_number（cn），
        # 邻居元素类型统计（Neighbor Element Histogram）
        local_targets = []
        for site_idx, site in enumerate(structure):
            neighbors = structure.get_neighbors(site, r=3.0)
            cn = len(neighbors)
            elem_counter = Counter(
                neigh.specie.symbol for neigh in neighbors
            )
            # 归一化直方图
            total = sum(elem_counter.values())
            if total > 0:
                elem_frac = {k: v / total for k, v in elem_counter.items()}
            else:
                elem_frac = {}
            local_targets.append({
                "cn": cn,
                "neighbor_elements": elem_frac,
            })
        return local_targets

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        global_vq_targets = None
        local_vq_targets = None
        max_retry = 8 if self.skip_bad_cif else 1
        structure = None
        pyg_data = None
        cry = None
        cif_id = None
        for retry in range(max_retry):
            ridx = (idx + retry) % len(self.id_prop_data)
            cry = self.id_prop_data[ridx]
            cif_id = cry.get('material_id', cry.get('id', f'crystal_{ridx}'))
            try:
                cif_io = StringIO(cry['cif'])
                parser = CifParser(cif_io)
                # Representation contract: all model inputs use the CIF's
                # current/full cell.  The legacy ``get_structures()`` API
                # implicitly returns a primitive structure in pymatgen, which
                # would desynchronise graph natoms from the HDS prompt formula.
                parse_structures = getattr(parser, "parse_structures", None)
                if parse_structures is None:
                    structure = parser.get_structures(primitive=False)[0]
                else:
                    structure = parse_structures(primitive=False)[0]
            except Exception as e:
                self._log_bad_cif(str(cif_id), f"CIFParseError:{type(e).__name__}")
                if retry == max_retry - 1:
                    raise
                continue

            # niggli reduction + graph conversion
            reduced_st = structure.get_reduced_structure()
            canonical_st = Structure(
                lattice=Lattice.from_parameters(*reduced_st.lattice.parameters),
                species=reduced_st.species,
                coords=reduced_st.frac_coords,
                coords_are_cartesian=False,
            )
            atoms = canonical_st.to_ase_atoms()
            pyg_data = atoms_to_pyg_data(atoms)

            # Skip malformed samples with invalid atomic numbers (e.g., 0).
            atom_types = getattr(pyg_data, "atom_types", None)
            if atom_types is None or int(atom_types.min().item()) < 1:
                self._log_bad_cif(str(cif_id), "InvalidAtomType")
                if retry == max_retry - 1:
                    raise ValueError(f"Invalid atom_types detected for {cif_id}")
                structure = None
                pyg_data = None
                continue

            if not self.cond_guide:
                global_vq_targets = self.calculate_global_vq_target(structure, cif_id=cif_id)
                local_vq_targets = self.calculate_local_vq_target(structure)
            break

        if structure is None or pyg_data is None:
            raise RuntimeError(f"Failed to build valid sample around index={idx}")
        try:
            if 'formation_energy' in self.proper:
                proper = torch.tensor(cry['formation_energy_per_atom'])
            else:
                proper = torch.tensor(cry[self.proper])
        except:
            proper = torch.tensor(-1.0)
        data_dict = {
            'pyg_data': pyg_data,
            'global_vq_targets': global_vq_targets,
            'local_vq_targets': local_vq_targets,
            'property': proper,
        }
        return data_dict, cif_id


