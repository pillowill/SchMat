"""
Tokenizer / vocabulary for simplified crystal VQ tokens.

默认与 scripts/cif2hds.py 的简化输出一致：
<BOS> <CRYSTAL>
<PROPERTY> ... </PROPERTY>                 # optional
<FORMULA> <ATOM_X> <COUNT_n> ... </FORMULA>
<GLOBAL> <GLOBAL_CODE_i> <NUM_SITES_n> </GLOBAL>
<SITES> <SITE_i> <ATOM_X> <LOCAL_CODE_j> </SITE_i> ... </SITES>
</CRYSTAL> <EOS>

默认不注册 <SPACE_GROUP_*> 和 <COORD_*>，避免与简化 cif2hds 输出不一致。
如需兼容旧实验，可在 CrystalVocabConfig 中设置 include_space_group_tokens=True、
include_crystal_system_tokens=True 或 include_coord_tokens=True。
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import floor
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


GLOBAL_CODE_RE = re.compile(r"^<GLOBAL_CODE(?:_Q(\d+))?_(\d+)>$")
LOCAL_CODE_RE = re.compile(r"^<LOCAL_CODE(?:_Q(\d+))?_(\d+)>$")
SITE_START_RE = re.compile(r"^<SITE_(\d+)>$")
ATOM_RE = re.compile(r"^<ATOM_([A-Za-z]{1,2})>$")
COUNT_RE = re.compile(r"^<COUNT_(\d+)>$")
PROPERTY_NAME_RE = re.compile(r"^<PROPERTY_NAME_([A-Z0-9_]+)>$")
PROPERTY_BIN_RE = re.compile(r"^<PROPERTY_BIN_(\d+)>$")
FORMATION_ENERGY_10_MIN = -4.5
FORMATION_ENERGY_10_MAX = 0.5
FORMATION_ENERGY_10_BIN_SIZE = 0.5


def bin_formation_energy_10(value: float) -> int:
    """Map [-4.5, 0.5) eV/atom strictly to one-based bins 1..10."""
    value = float(value)
    if not FORMATION_ENERGY_10_MIN <= value < FORMATION_ENERGY_10_MAX:
        raise ValueError(
            f"Formation energy {value} eV/atom is outside "
            f"[{FORMATION_ENERGY_10_MIN}, {FORMATION_ENERGY_10_MAX})"
        )
    return int(floor((value - FORMATION_ENERGY_10_MIN) / FORMATION_ENERGY_10_BIN_SIZE)) + 1


def hds_stage_names(num_quantizers_global: int, num_quantizers_local: int) -> Tuple[str, ...]:
    """Return the ordered RVQ stage schema for one HDS vocabulary."""
    return tuple(
        [f"GLOBAL_Q{q}" for q in range(int(num_quantizers_global))]
        + [f"LOCAL_Q{q}" for q in range(int(num_quantizers_local))]
    )


def parse_hds_code_token(token: str) -> Optional[Tuple[str, Optional[int], int]]:
    """Parse Q-tagged and legacy HDS code tokens.

    Returns ``(kind, quantizer, code_index)``.  ``quantizer`` is ``None`` for
    legacy tokens such as ``<LOCAL_CODE_12>``.
    """
    match = GLOBAL_CODE_RE.fullmatch(token)
    if match:
        return "global", int(match.group(1)) if match.group(1) is not None else None, int(match.group(2))
    match = LOCAL_CODE_RE.fullmatch(token)
    if match:
        return "local", int(match.group(1)) if match.group(1) is not None else None, int(match.group(2))
    return None


@lru_cache(maxsize=None)
def _atomic_number(symbol: str) -> int:
    """Resolve atomic number lazily so importing the tokenizer stays light."""
    try:
        from pymatgen.core.periodic_table import Element

        return int(Element(symbol).Z)
    except Exception as exc:
        raise ValueError(f"Unknown element symbol in HDS: {symbol}") from exc


@dataclass
class CompiledHDS:
    """Canonical HDS plus the metadata required by cached E2 training."""

    tokens: List[str]
    stage_ids: List[int]
    code_position_mask: List[bool]
    code_kinds: List[int]  # 0=non-code, 1=global, 2=local
    code_quantizers: List[int]  # -1=non-code, otherwise zero-based q
    code_targets: List[int]  # -1=non-code, otherwise codebook index
    composition_counts: List[Tuple[str, int]]
    atom_symbols: List[str]
    global_codes: List[int]
    local_codes: List[List[int]]
    canonical_atom_permutation: List[int]
    inverse_canonical_atom_permutation: List[int]


class HDSCompiler:
    """Build and validate the deterministic HDS skeleton used by all callers.

    The compiler is the shared seam between direct-HDS extraction, prepared
    caches, constrained generation, and diffusion-condition parsing.  Site
    atoms are the source of truth; reduced formula tokens are never padded or
    truncated to recover atom types.
    """

    def __init__(self, config: "CrystalVocabConfig"):
        self.config = config
        self.num_global = int(config.num_quantizers_global)
        self.num_local = int(config.num_quantizers_local)
        self.global_size = int(config.codebooksize_global)
        self.local_size = int(config.codebooksize_local)
        self.stage_names = hds_stage_names(self.num_global, self.num_local)
        self.stage_to_id = {name: i for i, name in enumerate(self.stage_names, start=1)}

    def _validate_codes(
        self,
        global_codes: Sequence[int],
        atom_symbols: Sequence[str],
        local_codes: Sequence[Sequence[int]],
    ) -> None:
        if len(global_codes) != self.num_global:
            raise ValueError(f"Expected {self.num_global} global codes, got {len(global_codes)}")
        if len(atom_symbols) != len(local_codes):
            raise ValueError(
                f"Site atom/code count mismatch: atoms={len(atom_symbols)}, local_rows={len(local_codes)}"
            )
        if not atom_symbols:
            raise ValueError("HDS requires at least one site")
        if len(atom_symbols) > int(self.config.max_sites):
            raise ValueError(f"HDS has {len(atom_symbols)} sites, max_sites={self.config.max_sites}")
        for code in global_codes:
            if not 0 <= int(code) < self.global_size:
                raise ValueError(f"Global code out of range: {code}")
        for site_index, row in enumerate(local_codes, start=1):
            if len(row) != self.num_local:
                raise ValueError(
                    f"Site {site_index} expected {self.num_local} local codes, got {len(row)}"
                )
            for code in row:
                if not 0 <= int(code) < self.local_size:
                    raise ValueError(f"Local code out of range at site {site_index}: {code}")
        for symbol in atom_symbols:
            _atomic_number(symbol)

    def normalize_composition_counts(
        self,
        composition_counts: Sequence[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        """Validate and atomically order one full-cell composition."""
        normalized: List[Tuple[str, int]] = []
        seen = set()
        for symbol, raw_count in composition_counts:
            symbol = str(symbol)
            count = int(raw_count)
            if symbol in seen:
                raise ValueError(f"Duplicate element in HDS composition: {symbol}")
            if count <= 0:
                raise ValueError(f"Composition count must be positive for {symbol}: {count}")
            if count > int(self.config.max_formula_count):
                raise ValueError(
                    f"Composition count {count} for {symbol} exceeds "
                    f"max_formula_count={self.config.max_formula_count}"
                )
            _atomic_number(symbol)
            seen.add(symbol)
            normalized.append((symbol, count))
        if not normalized:
            raise ValueError("HDS composition must contain at least one element")
        normalized.sort(key=lambda item: _atomic_number(item[0]))
        num_sites = sum(count for _, count in normalized)
        if num_sites > int(self.config.max_sites):
            raise ValueError(f"Composition has {num_sites} sites, max_sites={self.config.max_sites}")
        return normalized

    def parse_composition_tokens(self, tokens: Sequence[str]) -> List[Tuple[str, int]]:
        """Parse a complete FORMULA block without padding, truncation, or repair."""
        tokens = [token for token in tokens if token != "<PAD>"]
        if tokens.count("<FORMULA>") != 1 or tokens.count("</FORMULA>") != 1:
            raise ValueError("HDS must contain exactly one complete FORMULA block")
        start = tokens.index("<FORMULA>")
        end = tokens.index("</FORMULA>", start + 1)
        body = tokens[start + 1:end]
        if len(body) == 0 or len(body) % 2 != 0:
            raise ValueError("FORMULA must contain complete ATOM/COUNT pairs")
        parsed: List[Tuple[str, int]] = []
        for offset in range(0, len(body), 2):
            atom_match = ATOM_RE.fullmatch(body[offset])
            count_match = COUNT_RE.fullmatch(body[offset + 1])
            if atom_match is None or count_match is None:
                raise ValueError("FORMULA must alternate ATOM and COUNT tokens")
            parsed.append((atom_match.group(1), int(count_match.group(1))))
        normalized = self.normalize_composition_counts(parsed)
        if parsed != normalized:
            raise ValueError("FORMULA elements must be unique and sorted by atomic number")
        return normalized

    def compile_from_composition(
        self,
        composition_counts: Sequence[Tuple[str, int]],
        global_codes: Sequence[int],
        local_codes: Sequence[Sequence[int]],
        property_tokens: Optional[Sequence[str]] = None,
    ) -> CompiledHDS:
        """Compile CSP output using the full-cell composition as skeleton truth."""
        normalized = self.normalize_composition_counts(composition_counts)
        atom_symbols = [symbol for symbol, count in normalized for _ in range(count)]
        if len(local_codes) != len(atom_symbols):
            raise ValueError(
                f"Composition/local row mismatch: sites={len(atom_symbols)}, local_rows={len(local_codes)}"
            )
        return self.compile_from_codes(
            global_codes=global_codes,
            atom_symbols=atom_symbols,
            local_codes=local_codes,
            property_tokens=property_tokens,
        )

    @staticmethod
    def _inverse_permutation(permutation: Sequence[int]) -> List[int]:
        inverse = [0] * len(permutation)
        for canonical_index, original_index in enumerate(permutation):
            inverse[int(original_index)] = int(canonical_index)
        return inverse

    def compile_from_codes(
        self,
        global_codes: Sequence[int],
        atom_symbols: Sequence[str],
        local_codes: Sequence[Sequence[int]],
        property_tokens: Optional[Sequence[str]] = None,
    ) -> CompiledHDS:
        """Compile one canonical, stage-aware HDS record from discrete codes."""
        global_codes = [int(code) for code in global_codes]
        atom_symbols = [str(symbol) for symbol in atom_symbols]
        local_codes = [[int(code) for code in row] for row in local_codes]
        self._validate_codes(global_codes, atom_symbols, local_codes)
        property_tokens = list(property_tokens or [])
        if property_tokens:
            if len(property_tokens) != 4 or property_tokens[0] != "<PROPERTY>" or property_tokens[-1] != "</PROPERTY>":
                raise ValueError("PROPERTY block must contain exactly name and bin tokens")
            name_match = PROPERTY_NAME_RE.fullmatch(property_tokens[1])
            bin_match = PROPERTY_BIN_RE.fullmatch(property_tokens[2])
            if name_match is None or bin_match is None:
                raise ValueError("Malformed PROPERTY name/bin token")
            allowed_names = {sanitize_token_name(name) for name in self.config.property_names}
            if allowed_names and name_match.group(1) not in allowed_names:
                raise ValueError(
                    f"Property {name_match.group(1)} is not in configured property_names={sorted(allowed_names)}"
                )
            bin_index = int(bin_match.group(1))
            if not 0 <= bin_index < int(self.config.max_property_bins):
                raise ValueError(
                    f"Property bin {bin_index} is outside vocabulary size {self.config.max_property_bins}"
                )

        permutation = sorted(
            range(len(atom_symbols)),
            key=lambda index: (
                _atomic_number(atom_symbols[index]),
                tuple(local_codes[index]),
                int(index),
            ),
        )
        canonical_atoms = [atom_symbols[index] for index in permutation]
        canonical_local = [local_codes[index] for index in permutation]
        inverse = self._inverse_permutation(permutation)

        counts = Counter(canonical_atoms)
        composition_counts = sorted(counts.items(), key=lambda item: _atomic_number(item[0]))

        tokens: List[str] = ["<BOS>", "<CRYSTAL>"]
        if property_tokens:
            tokens.extend(property_tokens)
        tokens.append("<FORMULA>")
        for symbol, count in composition_counts:
            if count <= 0:
                raise ValueError(f"Invalid composition count for {symbol}: {count}")
            tokens.extend([f"<ATOM_{symbol}>", f"<COUNT_{count}>"])
        tokens.extend(["</FORMULA>", "<GLOBAL>", f"<NUM_SITES_{len(canonical_atoms)}>"])

        for q, code in enumerate(global_codes):
            tokens.extend([f"<STAGE_GLOBAL_Q{q}>", f"<GLOBAL_CODE_Q{q}_{code}>"])
        tokens.extend(["</GLOBAL>", "<SITES>"])

        for site_index, (symbol, row) in enumerate(zip(canonical_atoms, canonical_local), start=1):
            tokens.extend([f"<SITE_{site_index}>", f"<ATOM_{symbol}>"])
            for q, code in enumerate(row):
                tokens.extend([f"<STAGE_LOCAL_Q{q}>", f"<LOCAL_CODE_Q{q}_{code}>"])
            tokens.append(f"</SITE_{site_index}>")
        tokens.extend(["</SITES>", "</CRYSTAL>", "<EOS>"])

        stage_ids: List[int] = []
        code_position_mask: List[bool] = []
        code_kinds: List[int] = []
        code_quantizers: List[int] = []
        code_targets: List[int] = []
        for token in tokens:
            parsed = parse_hds_code_token(token)
            marker = re.fullmatch(r"<STAGE_(GLOBAL|LOCAL)_Q(\d+)>", token)
            if parsed:
                kind, quantizer, code_index = parsed
                assert quantizer is not None
                stage_name = f"{kind.upper()}_Q{quantizer}"
                stage_ids.append(self.stage_to_id[stage_name])
                code_position_mask.append(True)
                code_kinds.append(1 if kind == "global" else 2)
                code_quantizers.append(int(quantizer))
                code_targets.append(int(code_index))
            elif marker:
                stage_name = f"{marker.group(1)}_Q{int(marker.group(2))}"
                if stage_name not in self.stage_to_id:
                    raise ValueError(f"Stage marker is outside the configured schema: {token}")
                stage_ids.append(self.stage_to_id[stage_name])
                code_position_mask.append(False)
                code_kinds.append(0)
                code_quantizers.append(-1)
                code_targets.append(-1)
            else:
                stage_ids.append(0)
                code_position_mask.append(False)
                code_kinds.append(0)
                code_quantizers.append(-1)
                code_targets.append(-1)

        compiled = CompiledHDS(
            tokens=tokens,
            stage_ids=stage_ids,
            code_position_mask=code_position_mask,
            code_kinds=code_kinds,
            code_quantizers=code_quantizers,
            code_targets=code_targets,
            composition_counts=[(symbol, int(count)) for symbol, count in composition_counts],
            atom_symbols=canonical_atoms,
            global_codes=list(global_codes),
            local_codes=[list(row) for row in canonical_local],
            canonical_atom_permutation=[int(index) for index in permutation],
            inverse_canonical_atom_permutation=inverse,
        )
        self.validate_compiled(compiled)
        return compiled

    def compile_legacy_record(self, sample: Dict[str, Any]) -> CompiledHDS:
        """Rebuild a strict Q-tagged HDS from one legacy/direct token record."""
        tokens = list(sample.get("tokens", []))
        while tokens and tokens[-1] == "<PAD>":
            tokens.pop()
        if "<PAD>" in tokens:
            raise ValueError(f"Internal PAD token for id={sample.get('id')}")
        property_tokens: List[str] = []
        if tokens.count("<PROPERTY>") != tokens.count("</PROPERTY>") or tokens.count("<PROPERTY>") > 1:
            raise ValueError(f"Malformed or repeated property block for id={sample.get('id')}")
        if "<PROPERTY>" in tokens:
            start = tokens.index("<PROPERTY>")
            try:
                end = tokens.index("</PROPERTY>", start + 1)
            except ValueError as exc:
                raise ValueError(f"Unclosed property block for id={sample.get('id')}") from exc
            property_tokens = tokens[start:end + 1]

        global_codes: List[int] = []
        in_global = False
        for token in tokens:
            if token == "<GLOBAL>":
                in_global = True
                continue
            if token == "</GLOBAL>":
                in_global = False
                continue
            parsed = parse_hds_code_token(token)
            if in_global and parsed and parsed[0] == "global":
                global_codes.append(parsed[2])

        atom_symbols: List[str] = []
        local_codes: List[List[int]] = []
        cursor = 0
        while cursor < len(tokens):
            opening = SITE_START_RE.fullmatch(tokens[cursor])
            if not opening:
                cursor += 1
                continue
            site_number = int(opening.group(1))
            closing = f"</SITE_{site_number}>"
            try:
                end = tokens.index(closing, cursor + 1)
            except ValueError as exc:
                raise ValueError(f"Unclosed site block {site_number} for id={sample.get('id')}") from exc
            block = tokens[cursor + 1:end]
            atoms = [match.group(1) for token in block if (match := ATOM_RE.fullmatch(token))]
            if len(atoms) != 1:
                raise ValueError(
                    f"Site {site_number} must contain exactly one atom token for id={sample.get('id')}"
                )
            row = [
                parsed[2]
                for token in block
                if (parsed := parse_hds_code_token(token)) and parsed[0] == "local"
            ]
            atom_symbols.append(atoms[0])
            local_codes.append(row)
            cursor = end + 1

        return self.compile_from_codes(
            global_codes=global_codes,
            atom_symbols=atom_symbols,
            local_codes=local_codes,
            property_tokens=property_tokens,
        )

    def parse_strict_record(self, tokens: Sequence[str]) -> CompiledHDS:
        """Parse and validate one canonical Q-tagged HDS record exactly."""
        stripped = list(tokens)
        while stripped and stripped[-1] == "<PAD>":
            stripped.pop()
        compiled = self.compile_legacy_record({"tokens": stripped, "id": "strict-parse"})
        # Recompilation uses site atoms as the only structural source.  Exact
        # equality therefore checks formula/site consistency, stage order,
        # site numbering, delimiters, and canonical same-element ordering.
        if stripped != compiled.tokens:
            for index, (actual, expected) in enumerate(zip(stripped, compiled.tokens)):
                if actual != expected:
                    raise ValueError(
                        f"Non-canonical HDS token at position {index}: got {actual}, expected {expected}"
                    )
            raise ValueError(
                f"Non-canonical HDS length: got {len(stripped)}, expected {len(compiled.tokens)}"
            )
        self.parse_composition_tokens(stripped)
        return compiled

    def validate_compiled(self, compiled: CompiledHDS) -> None:
        """Enforce invariants required by E2 training and constrained decoding."""
        if not (
            len(compiled.tokens)
            == len(compiled.stage_ids)
            == len(compiled.code_position_mask)
            == len(compiled.code_kinds)
            == len(compiled.code_quantizers)
            == len(compiled.code_targets)
        ):
            raise ValueError("Compiled HDS metadata lengths do not match")
        if sum(count for _, count in compiled.composition_counts) != len(compiled.atom_symbols):
            raise ValueError("Formula count total does not match site atom count")
        expected_atoms: List[str] = []
        for symbol, count in compiled.composition_counts:
            expected_atoms.extend([symbol] * int(count))
        if expected_atoms != compiled.atom_symbols:
            raise ValueError("Canonical formula expansion does not match ordered site atoms")
        if compiled.tokens.count("<EOS>") != 1 or compiled.tokens[-1] != "<EOS>":
            raise ValueError("Compiled HDS must end with exactly one <EOS>")
        if "<UNK>" in compiled.tokens or "<PAD>" in compiled.tokens:
            raise ValueError("Compiled HDS may not contain <UNK> or internal <PAD>")
        global_slots = [
            (q, target)
            for kind, q, target in zip(
                compiled.code_kinds, compiled.code_quantizers, compiled.code_targets
            )
            if kind == 1
        ]
        if [q for q, _ in global_slots] != list(range(self.num_global)):
            raise ValueError("Global code stages are missing or out of order")
        local_slots = sum(1 for kind in compiled.code_kinds if kind == 2)
        if local_slots != len(compiled.atom_symbols) * self.num_local:
            raise ValueError("Local code slot count does not match sites x local quantizers")


@dataclass
class CrystalVocabConfig:
    num_quantizers_global: int = 2
    codebooksize_global: int = 512
    num_quantizers_local: int = 3
    codebooksize_local: int = 512
    max_sites: int = 100  # used for NUM_SITES_x and SITE_i markers
    # Kept separate from max_sites for checkpoint compatibility: historical
    # vocabularies reserved only COUNT_1..100 even when max_sites was larger.
    max_formula_count: int = 100
    max_coord: int = 20   # used only when include_coord_tokens=True
    include_space_group_tokens: bool = False
    include_crystal_system_tokens: bool = False
    max_space_group: int = 230
    include_coord_tokens: bool = False
    # HDS-reconstruction tokens. They are opt-in so existing checkpoints keep
    # their original vocabulary order and remain loadable.
    include_hds_stage_tokens: bool = False
    include_mask_token: bool = False
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
        if cfg.include_mask_token:
            self._add("<MASK>")
        if cfg.include_hds_stage_tokens:
            for stage in hds_stage_names(
                cfg.num_quantizers_global,
                cfg.num_quantizers_local,
            ):
                self._add(f"<STAGE_{stage}>")
        # ===== FORMULA =====
        # Keep COUNT_1..100 in their historical positions so legacy vocabularies
        # built with max_sites<=100 retain exactly the same token ids.
        for value in INTS:
            self._add(f"<COUNT_{value}>")

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
        # 对应 scripts/cif2hds.build_crystal_tokens 产生的：
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

        # Larger cells need counts beyond the historical fixed range. Append
        # only the extra tokens at the end, avoiding shifts to all existing ids.
        for value in range(len(INTS) + 1, cfg.max_formula_count + 1):
            self._add(f"<COUNT_{value}>")

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
    max_formula_count: int = 100,
    max_coord: int = 20,
    include_space_group_tokens: bool = False,
    include_crystal_system_tokens: bool = False,
    max_space_group: int = 230,
    include_coord_tokens: bool = False,
    include_hds_stage_tokens: bool = False,
    include_mask_token: bool = False,
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
        max_formula_count=max_formula_count,
        max_coord=max_coord,
        include_space_group_tokens=include_space_group_tokens,
        include_crystal_system_tokens=include_crystal_system_tokens,
        max_space_group=max_space_group,
        include_coord_tokens=include_coord_tokens,
        include_hds_stage_tokens=include_hds_stage_tokens,
        include_mask_token=include_mask_token,
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
