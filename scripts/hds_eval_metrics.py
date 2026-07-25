import itertools
import warnings
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

import smact
from smact.screening import pauling_test

# eval_utils.py style comp scaler (preferred)

from model.common.constants import CompScalerMeans, CompScalerStds
from model.common.data_utils import StandardScaler



from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")




def parse_structure_from_cif_str(cif_str: str) -> Optional[Structure]:
    if not isinstance(cif_str, str) or len(cif_str.strip()) == 0:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\\..*")
        parser = CifParser(StringIO(cif_str))
        try:
            # Reference CIFs are evaluated in the same full-cell
            # representation used by prompts, HDS and diffusion targets.
            return parser.parse_structures(primitive=False)[0]
        except Exception:
            try:
                return parser.get_structures(primitive=False)[0]
            except Exception:
                return None


def _is_odd(struct: Structure) -> bool:
    try:
        lengths = np.array(struct.lattice.abc, dtype=float)
        angles = np.array(struct.lattice.angles, dtype=float)
        n_atoms = max(1, len(struct))
        if np.any(angles < 10.0) or np.any(angles > 170.0):
            return True
        if np.any(lengths / np.power(n_atoms, 1.0 / 3.0) > 20.0):
            return True
        if np.any(lengths / np.power(n_atoms, 1.0 / 3.0) < 0.1):
            return True
        if not (0.1 < (struct.volume / n_atoms) < 100.0):
            return True
        return False
    except Exception:
        return True


def structure_validity(struct: Structure, cutoff: float = 0.5) -> bool:
    try:
        if _is_odd(struct):
            return False
        dist_mat = struct.distance_matrix
        dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * min(struct.lattice.abc))
        return bool(dist_mat.min() >= cutoff and struct.volume >= 0.1)
    except Exception:
        return False


def smact_validity_from_symbols_counts(
    elem_symbols: List[str],
    counts: List[int],
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> bool:
    if len(elem_symbols) == 0:
        return False
    if len(set(elem_symbols)) == 1:
        return True

    if include_alloys:
        is_metal = [s in smact.metals for s in elem_symbols]
        if all(is_metal):
            return True

    space = smact.element_dictionary(tuple(elem_symbols))
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False

    threshold = int(np.max(counts)) if len(counts) > 0 else 1
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(int(c),) for c in counts]
        cn_e, _ = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    en_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    en_ok = True
            else:
                en_ok = True
            if en_ok:
                return True
    return False


def composition_validity(struct: Structure) -> bool:
    try:
        comp = struct.composition.get_el_amt_dict()
        elems = sorted(comp.keys())
        if len(elems) >= 8:
            return False
        counts = [int(round(comp[e])) for e in elems]
        return smact_validity_from_symbols_counts(elems, counts)
    except Exception:
        return False


def structure_fp_simple(struct: Structure) -> Optional[np.ndarray]:
    if CrystalNNFP is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                site_fps = [CrystalNNFP.featurize(struct, i) for i in range(len(struct))]
            if len(site_fps) > 0:
                return np.array(site_fps, dtype=float).mean(axis=0)
        except Exception:
            pass
    try:
        a, b, c = struct.lattice.abc
        alpha, beta, gamma = struct.lattice.angles
        return np.array(
            [
                float(a), float(b), float(c),
                float(alpha), float(beta), float(gamma),
                float(struct.volume), float(struct.density), float(len(struct)),
            ],
            dtype=float,
        )
    except Exception:
        return None


def composition_fp_simple(struct: Structure) -> Optional[np.ndarray]:
    if CompFP is not None:
        try:
            elem_counter = {str(el): float(amt) for el, amt in struct.composition.get_el_amt_dict().items()}
            comp = Composition(elem_counter)
            return np.array(CompFP.featurize(comp), dtype=float)
        except Exception:
            pass
    try:
        vec = np.zeros((118,), dtype=float)
        total = 0.0
        for el, amt in struct.composition.get_el_amt_dict().items():
            z = int(Element(el).Z)
            if 1 <= z <= 118:
                vec[z - 1] = float(amt)
                total += float(amt)
        if total <= 0:
            return None
        return vec / total
    except Exception:
        return None


def normalize_features(train_like: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu = np.mean(train_like, axis=0, keepdims=True)
    std = np.std(train_like, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mu) / std


@dataclass
class CrystalRecord:
    struct: Structure
    struct_fp: np.ndarray
    comp_fp: np.ndarray
    density: float
    n_elements: float


class OnlineDssMetrics:
    def __init__(
        self,
        reference_structs: List[Structure],
        struc_cutoff: float,
        comp_cutoff: float,
        show_feature_cache_progress: bool = False,
        feature_cache_desc: str = "[OnlineMetrics] Building feature cache",
    ):
        self.struc_cutoff = float(struc_cutoff)
        self.comp_cutoff = float(comp_cutoff)

        self.ref_records: List[CrystalRecord] = []
        ref_iter = reference_structs
        if show_feature_cache_progress:
            ref_iter = tqdm(reference_structs, total=len(reference_structs), desc=feature_cache_desc, unit="ref", leave=False)
        for st in ref_iter:
            s_fp = structure_fp_simple(st)
            c_fp = composition_fp_simple(st)
            if s_fp is None or c_fp is None:
                continue
            self.ref_records.append(
                CrystalRecord(
                    struct=st,
                    struct_fp=s_fp,
                    comp_fp=c_fp,
                    density=float(st.density),
                    n_elements=float(len(st.composition.elements)),
                )
            )

        if len(self.ref_records) == 0:
            raise ValueError("No valid reference structures with computable fingerprints.")

        self.ref_struct_fp = np.array([r.struct_fp for r in self.ref_records], dtype=float)
        self.ref_comp_fp_raw = np.array([r.comp_fp for r in self.ref_records], dtype=float)

        self.comp_scaler = None
        if (
            StandardScaler is not None
            and CompScalerMeans is not None
            and CompScalerStds is not None
            and len(CompScalerMeans) == self.ref_comp_fp_raw.shape[1]
            and len(CompScalerStds) == self.ref_comp_fp_raw.shape[1]
        ):
            try:
                self.comp_scaler = StandardScaler(
                    means=np.array(CompScalerMeans),
                    stds=np.array(CompScalerStds),
                    replace_nan_token=0.0,
                )
            except Exception:
                self.comp_scaler = None

        if self.comp_scaler is not None:
            self.ref_comp_fp = self.comp_scaler.transform(self.ref_comp_fp_raw)
        else:
            self.ref_comp_fp = normalize_features(self.ref_comp_fp_raw, self.ref_comp_fp_raw)

        self.ref_density = np.array([r.density for r in self.ref_records], dtype=float)
        self.ref_nel = np.array([r.n_elements for r in self.ref_records], dtype=float)

        self.reset_generated_state()

    def reset_generated_state(self):
        # Reset generated-side running stats while keeping reference fingerprints/scaler cached.
        self.ref_min_struc = np.full((self.ref_struct_fp.shape[0],), np.inf, dtype=float)
        self.ref_min_comp = np.full((self.ref_struct_fp.shape[0],), np.inf, dtype=float)
        self.gen_struc_min = []
        self.gen_comp_min = []

        self.gen_density = []
        self.gen_nel = []
        self.gen_density_valid = []
        self.gen_nel_valid = []
        self.total_gen = 0
        self.struct_valid_count = 0
        self.comp_valid_count = 0

    def update_with_structure(self, st: Structure):
        self.total_gen += 1
        is_struct_valid = structure_validity(st)
        is_comp_valid = composition_validity(st)
        is_valid = is_struct_valid and is_comp_valid

        if is_struct_valid:
            self.struct_valid_count += 1
        if is_comp_valid:
            self.comp_valid_count += 1

        self.gen_density.append(float(st.density))
        self.gen_nel.append(float(len(st.composition.elements)))
        if is_valid:
            self.gen_density_valid.append(float(st.density))
            self.gen_nel_valid.append(float(len(st.composition.elements)))

        s_fp = structure_fp_simple(st)
        c_fp_raw = composition_fp_simple(st)
        if s_fp is None or c_fp_raw is None:
            return

        if self.comp_scaler is not None:
            c_fp = self.comp_scaler.transform(c_fp_raw[None, :])[0]
        else:
            c_fp = normalize_features(self.ref_comp_fp_raw, c_fp_raw[None, :])[0]

        ds = cdist(s_fp[None, :], self.ref_struct_fp, metric="euclidean")[0]
        dc = cdist(c_fp[None, :], self.ref_comp_fp, metric="euclidean")[0]

        self.gen_struc_min.append(float(np.min(ds)))
        self.gen_comp_min.append(float(np.min(dc)))

        self.ref_min_struc = np.minimum(self.ref_min_struc, ds)
        self.ref_min_comp = np.minimum(self.ref_min_comp, dc)

    def summary(self) -> Dict[str, float]:
        if self.total_gen == 0:
            return {
                "structure_valid_rate": 0.0,
                "composition_valid_rate": 0.0,
                "cov_recall": 0.0,
                "cov_precision": 0.0,
                "wdist_rho": float("nan"),
                "wdist_nel": float("nan"),
                "amsd_recall": float("nan"),
                "amsd_precision": float("nan"),
                "amcd_recall": float("nan"),
                "amcd_precision": float("nan"),
                "num_generated": 0.0,
            }

        struc_recall_dist = self.ref_min_struc
        comp_recall_dist = self.ref_min_comp
        cov_recall = float(
            np.mean((struc_recall_dist <= self.struc_cutoff) & (comp_recall_dist <= self.comp_cutoff))
        )

        if len(self.gen_struc_min) > 0:
            gs = np.array(self.gen_struc_min)
            gc = np.array(self.gen_comp_min)
            valid_mask = (gs <= self.struc_cutoff) & (gc <= self.comp_cutoff)
            cov_precision = float(np.sum((gs <= self.struc_cutoff) & (gc <= self.comp_cutoff)) / self.total_gen)
            amsd_precision = float(np.mean(gs))
            amcd_precision = float(np.mean(gc))
        else:
            cov_precision = 0.0
            amsd_precision = float("nan")
            amcd_precision = float("nan")

        # Keep CrystalFlow semantics: Wasserstein statistics over valid generated structures.
        wdist_rho = (
            float(wasserstein_distance(np.array(self.gen_density_valid), self.ref_density))
            if len(self.gen_density_valid) > 0
            else float("nan")
        )
        wdist_nel = (
            float(wasserstein_distance(np.array(self.gen_nel_valid), self.ref_nel))
            if len(self.gen_nel_valid) > 0
            else float("nan")
        )

        return {
            "structure_valid_rate": float(self.struct_valid_count / self.total_gen),
            "composition_valid_rate": float(self.comp_valid_count / self.total_gen),
            "cov_recall": cov_recall,
            "cov_precision": cov_precision,
            "wdist_rho": wdist_rho,
            "wdist_nel": wdist_nel,
            "amsd_recall": float(np.mean(struc_recall_dist)),
            "amsd_precision": amsd_precision,
            "amcd_recall": float(np.mean(comp_recall_dist)),
            "amcd_precision": amcd_precision,
            "num_generated": float(self.total_gen),
        }

