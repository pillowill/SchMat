"""
Tokenizer / vocabulary for simplified crystal VQ tokens.

默认与 scripts/cif2token.py 的简化输出一致：
<BOS> <CRYSTAL>
<PROPERTY> ... </PROPERTY>                 # optional
<FORMULA> <ATOM_X> <COUNT_n> ... </FORMULA>
<GLOBAL> <GLOBAL_CODE_i> <NUM_SITES_n> </GLOBAL>
<SITES> <SITE_i> <ATOM_X> <LOCAL_CODE_j> </SITE_i> ... </SITES>
</CRYSTAL> <EOS>

默认不注册 <SPACE_GROUP_*> 和 <COORD_*>，避免与简化 cif2token 输出不一致。
如需兼容旧实验，可在 CrystalVocabConfig 中设置 include_space_group_tokens=True、
include_crystal_system_tokens=True 或 include_coord_tokens=True。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


# Full atom list provided by the user
INTS = [str(d) for d in list(range(1, 101))]

ATOM_LIST = [
    "Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U",
    "Mn", "Lu", "Se", "Tl", "Hf", "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn",
    "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np", "Tc", "Hg",
    "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh",
    "Th", "Na", "Ru", "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F",
    "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb", "Dy", "Yb", "Nd", "Au",
    "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar",
]

CRYSTAL_SYSTEM_NAMES = [
    "TRICLINIC",
    "MONOCLINIC",
    "ORTHORHOMBIC",
    "TETRAGONAL",
    "TRIGONAL",
    "HEXAGONAL",
    "CUBIC",
]

LATTICE_SHAPE_NAMES = [
    "CUBIC",
    "TETRAGONAL",
    "ORTHORHOMBIC",
    "HEXAGONAL",
    "MONOCLINIC",
    "TRICLINIC",
]


def sanitize_token_name(name: str) -> str:
    """Convert arbitrary field names to a stable token suffix."""
    if name is None:
        return "UNKNOWN"
    s = str(name).strip().upper()
    chars = []
    prev_underscore = False
    for ch in s:
        if ch.isalnum():
            chars.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                chars.append("_")
                prev_underscore = True
    out = "".join(chars).strip("_")
    return out or "UNKNOWN"


@dataclass
class CrystalVocabConfig:
    num_quantizers_global: int = 2
    codebooksize_global: int = 512
    num_quantizers_local: int = 3
    codebooksize_local: int = 512
    max_sites: int = 100  # used for NUM_SITES_x and SITE_i markers
    max_coord: int = 20   # used only when include_coord_tokens=True
    include_space_group_tokens: bool = False
    include_crystal_system_tokens: bool = False
    max_space_group: int = 230
    include_coord_tokens: bool = False
    property_names: List[str] = None
    max_property_bins: int = 128
    atom_list: List[str] = None

    def __post_init__(self):
        if self.atom_list is None:
            self.atom_list = ATOM_LIST
        if self.property_names is None:
            self.property_names = []


class CrystalVocab:
    def __init__(self, config: CrystalVocabConfig):
        self.config = config
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build()

    def _add(self, token: str):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _build(self):
        cfg = self.config

        # ===== special =====
        for t in [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "<PROPERTY>", "</PROPERTY>",
            "<FORMULA>", "</FORMULA>",
            "<CRYSTAL>", "</CRYSTAL>",
            "<GLOBAL>", "</GLOBAL>",
            "<SITES>", "</SITES>",
        ]:
            self._add(t)
        # ===== FORMULA =====
        for int in INTS:
            self._add(f"<COUNT_{int}>")

        # ===== property =====
        for prop in self.config.property_names:
            prop_token = sanitize_token_name(prop)
            self._add(f"<PROPERTY_NAME_{prop_token}>")
        for i in range(cfg.max_property_bins):
            self._add(f"<PROPERTY_BIN_{i}>")

        # ===== GLOBAL RVQ codes (带 Q 前缀，多量化器版本) =====
        for q in range(cfg.num_quantizers_global):
            for i in range(cfg.codebooksize_global):
                self._add(f"<GLOBAL_CODE_Q{q}_{i}>")

        # ===== LOCAL RVQ codes (带 Q 前缀，多量化器版本) =====
        for q in range(cfg.num_quantizers_local):
            for i in range(cfg.codebooksize_local):
                self._add(f"<LOCAL_CODE_Q{q}_{i}>")

        # ===== GLOBAL / LOCAL codes（无 Q 前缀，简化版本）=====
        # 对应 scripts/cif2token.build_crystal_tokens 产生的：
        #   <GLOBAL_CODE_{code}> 和 <LOCAL_CODE_{code}>
        for i in range(cfg.codebooksize_global):
            self._add(f"<GLOBAL_CODE_{i}>")
        for i in range(cfg.codebooksize_local):
            self._add(f"<LOCAL_CODE_{i}>")

        # ===== crystal-level attributes =====
        for n in range(cfg.max_sites + 1):  # allow max_sites itself
            self._add(f"<NUM_SITES_{n}>")
        if cfg.include_space_group_tokens:
            for sg in range(1, cfg.max_space_group + 1):
                self._add(f"<SPACE_GROUP_{sg}>")
        if cfg.include_crystal_system_tokens:
            for name in CRYSTAL_SYSTEM_NAMES:
                self._add(f"<CRYSTAL_SYSTEM_{name}>")
        for name in LATTICE_SHAPE_NAMES:
            self._add(f"<LATTICE_SHAPE_{name}>")

        # ===== atom types =====
        for elem in cfg.atom_list:
            self._add(f"<ATOM_{elem}>")

        # ===== site markers =====
        for i in range(1, cfg.max_sites + 1):
            self._add(f"<SITE_{i}>")
            self._add(f"</SITE_{i}>")

        # ===== coord tokens (legacy/optional) =====
        if cfg.include_coord_tokens:
            for k in range(cfg.max_coord + 1):
                self._add(f"<COORD_{k}>")

    def __len__(self):
        return len(self.token_to_id)


class CrystalTokenizer:
    def __init__(self, vocab: CrystalVocab):
        self.vocab = vocab

    def encode(self, tokens: List[str], pad_to: Optional[int] = None) -> List[int]:
        ids = [self.vocab.token_to_id.get(t, self.vocab.token_to_id["<UNK>"]) for t in tokens]
        if pad_to is not None:
            if len(ids) > pad_to:
                ids = ids[:pad_to]
            else:
                pad_id = self.vocab.token_to_id["<PAD>"]
                ids = ids + [pad_id] * (pad_to - len(ids))
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        return [self.vocab.id_to_token.get(i, "<UNK>") for i in ids]


# Convenience builder from model/config parameters
def build_tokenizer_from_config(
    num_quantizers_global: int,
    codebooksize_global: int,
    num_quantizers_local: int,
    codebooksize_local: int,
    max_sites: int = 128,
    max_coord: int = 20,
    include_space_group_tokens: bool = False,
    include_crystal_system_tokens: bool = False,
    max_space_group: int = 230,
    include_coord_tokens: bool = False,
    property_names: Optional[Sequence[str]] = None,
    max_property_bins: int = 128,
    atom_list: Optional[List[str]] = None,
) -> CrystalTokenizer:
    cfg = CrystalVocabConfig(
        num_quantizers_global=num_quantizers_global,
        codebooksize_global=codebooksize_global,
        num_quantizers_local=num_quantizers_local,
        codebooksize_local=codebooksize_local,
        max_sites=max_sites,
        max_coord=max_coord,
        include_space_group_tokens=include_space_group_tokens,
        include_crystal_system_tokens=include_crystal_system_tokens,
        max_space_group=max_space_group,
        include_coord_tokens=include_coord_tokens,
        property_names=list(property_names) if property_names is not None else [],
        max_property_bins=max_property_bins,
        atom_list=atom_list or ATOM_LIST,
    )
    vocab = CrystalVocab(cfg)
    return CrystalTokenizer(vocab)
if __name__ == '__main__':

    # 创建词汇表实例
    config = CrystalVocabConfig()
    vocab = CrystalVocab(config)

    # 获取词汇表大小
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")