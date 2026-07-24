# -*- coding: utf-8 -*-
"""
train_ar_model.py

使用 `scripts/cif2hds.py` 预处理得到的 HDS 序列，训练 `model.crystal_token_gpt.GPT` 自回归模型。

数据格式（由 cif2hds 产生的 *.pkl 或 *.pkl.gz）：
    List[{"id": str, "tokens": List[str]}]
"""

import argparse
import gzip
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Sequence, Tuple
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

import yaml
from tqdm import tqdm


def load_pickle(path: str) -> List[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = pickle.load(f)
    return data


RVQ_STAGE_NAMES: Tuple[str, ...] = (
    "GLOBAL_Q0", "GLOBAL_Q1", "LOCAL_Q0", "LOCAL_Q1", "LOCAL_Q2",
)


def _stage_name(token: str) -> Optional[str]:
    """Return the RVQ stage encoded by a code or explicit stage marker."""
    marker = re.fullmatch(r"<STAGE_(GLOBAL|LOCAL)_Q(\d+)>", token)
    code = re.fullmatch(r"<(GLOBAL|LOCAL)_CODE_Q(\d+)_\d+>", token)
    match = marker or code
    if not match:
        return None
    name = f"{match.group(1)}_Q{match.group(2)}"
    return name if name in RVQ_STAGE_NAMES else None


def canonicalize_hds_tokens(tokens: Sequence[str]) -> List[str]:
    """Canonicalize formula and site-block ordering without touching RVQ values."""
    out = list(tokens)

    def span(open_token: str, close_token: str) -> Optional[Tuple[int, int]]:
        try:
            start = out.index(open_token)
            return start, out.index(close_token, start + 1)
        except ValueError:
            return None

    formula_span = span("<FORMULA>", "</FORMULA>")
    if formula_span:
        start, end = formula_span
        body = out[start + 1:end]
        pairs = [(body[i], body[i + 1]) for i in range(0, len(body) - 1, 2)
                 if body[i].startswith("<ATOM_") and body[i + 1].startswith("<COUNT_")]
        if len(pairs) * 2 == len(body):
            out[start + 1:end] = [token for pair in sorted(pairs) for token in pair]

    sites_span = span("<SITES>", "</SITES>")
    if sites_span:
        start, end = sites_span
        body = out[start + 1:end]
        blocks, cursor = [], 0
        while cursor < len(body):
            opening = re.fullmatch(r"<SITE_(\d+)>", body[cursor])
            if not opening:
                blocks = []
                break
            closing = f"</SITE_{opening.group(1)}>"
            try:
                block_end = body.index(closing, cursor + 1)
            except ValueError:
                blocks = []
                break
            blocks.append((int(opening.group(1)), body[cursor:block_end + 1]))
            cursor = block_end + 1
        if blocks and cursor == len(body):
            out[start + 1:end] = [token for _, block in sorted(blocks) for token in block]
    return out


def stage_mark_hds_tokens(tokens: Sequence[str]) -> List[str]:
    """Insert an explicit marker directly before each RVQ code token."""
    out: List[str] = []
    for token in tokens:
        stage = _stage_name(token)
        if stage and not token.startswith("<STAGE_"):
            out.append(f"<STAGE_{stage}>")
        out.append(token)
    return out


def build_stage_ids(tokens: Iterable[str]) -> List[int]:
    """Map syntax to 0 and the five RVQ stages to 1..5."""
    stage_to_id = {name: index for index, name in enumerate(RVQ_STAGE_NAMES, start=1)}
    return [stage_to_id.get(_stage_name(token), 0) for token in tokens]


class CrystalTokenDataset(Dataset):
    """One HDS dataset supporting causal and bidirectional masked objectives."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        max_seq_len: int,
        objective: str = "causal",
        stage_marked: bool = False,
        canonicalize: bool = False,
        mask_probability: float = 0.15,
        protected_token_ids: Optional[Sequence[int]] = None,
    ):
        if objective not in {"causal", "masked"}:
            raise ValueError(f"Unsupported HDS objective: {objective}")
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.objective = objective
        self.stage_marked = stage_marked
        self.canonicalize = canonicalize
        self.mask_probability = mask_probability
        self.pad_id = tokenizer.vocab.token_to_id["<PAD>"]
        self.mask_id = tokenizer.vocab.token_to_id.get("<MASK>")
        self.protected_token_ids = set(protected_token_ids or []) | {self.pad_id}
        if objective == "masked" and self.mask_id is None:
            raise ValueError("Masked-HDS requires <MASK> in the tokenizer vocabulary")
        self._indices = [i for i, sample in enumerate(samples) if len(sample.get("tokens", [])) >= 2]

    def __len__(self) -> int:
        return len(self._indices)

    def _prepare_tokens(self, sample: Dict[str, Any]) -> List[str]:
        tokens = list(sample["tokens"])
        if self.canonicalize:
            tokens = canonicalize_hds_tokens(tokens)
        return stage_mark_hds_tokens(tokens) if self.stage_marked else tokens

    def __getitem__(self, idx: int):
        tokens = self._prepare_tokens(self.samples[self._indices[idx]])
        stages = build_stage_ids(tokens)
        pad_to = self.max_seq_len + 1 if self.objective == "causal" else self.max_seq_len
        ids = torch.tensor(self.tokenizer.encode(tokens, pad_to=pad_to), dtype=torch.long)
        stage_ids = torch.tensor((stages + [0] * pad_to)[:pad_to], dtype=torch.long)

        if self.objective == "causal":
            return ids[:-1], ids[1:], stage_ids[:-1], stage_ids[1:], torch.ones_like(ids[:-1], dtype=torch.bool)

        target_ids = ids.clone()
        candidate = torch.ones_like(ids, dtype=torch.bool)
        for protected_id in self.protected_token_ids:
            candidate &= ids != protected_id
        candidate_indices = candidate.nonzero(as_tuple=False).flatten()
        if len(candidate_indices) == 0:
            raise ValueError("Masked-HDS sample has no maskable tokens")
        loss_mask = (torch.rand(len(ids)) < self.mask_probability) & candidate
        if not loss_mask.any():
            loss_mask[candidate_indices[torch.randint(len(candidate_indices), (1,))]] = True
        input_ids = ids.clone()
        input_ids[loss_mask] = int(self.mask_id)
        return input_ids, target_ids, stage_ids, stage_ids, loss_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Train crystal_token_gpt on crystal VQ token sequences")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI args will override YAML fields if both are provided.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        help="目录，包含 train_tokens.pkl(.gz) 和 val_tokens.pkl(.gz)，由 scripts/cif2hds.py 生成",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train_tokens.pkl.gz",
        help="训练集 token 文件名（默认: train_tokens.pkl.gz）",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="val_tokens.pkl.gz",
        help="验证集 token 文件名（默认: val_tokens.pkl.gz）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ckpts/cry2seq",
        help="模型 checkpoint 和日志保存目录",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="最大序列长度（不含最后一个被 shift 掉的位置）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="训练 batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="AdamW 的 weight decay",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.95),
        help="AdamW 的 betas 参数",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（<=0 表示不裁剪）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 的 num_workers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="GPT dropout 概率",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=8,
        help="Transformer 层数",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=8,
        help="多头注意力 head 数量",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=512,
        help="embedding 维度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--property_names",
        type=str,
        default=None,
        help="Comma-separated property names to reserve in the tokenizer vocab (e.g. band_gap,formation_energy).",
    )
    parser.add_argument(
        "--max_property_bins",
        type=int,
        default=None,
        help="Override tokenizer property-bin vocabulary size.",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=None,
        help="Codebook size used by cif2hds / Attn-RVQ. If not set, read from YAML or default to 512.",
    )
    parser.add_argument(
        "--codebooksize_global",
        type=int,
        default=None,
        help="Override global codebook size. Defaults to --codebook_size.",
    )
    parser.add_argument(
        "--codebooksize_local",
        type=int,
        default=None,
        help="Override local codebook size. Defaults to --codebook_size.",
    )
    parser.add_argument(
        "--num_quantizers_global",
        type=int,
        default=None,
        help="Number of global RVQ quantizers. Should match cif2hds/Attn-RVQ config.",
    )
    parser.add_argument(
        "--num_quantizers_local",
        type=int,
        default=None,
        help="Number of local RVQ quantizers. Should match cif2hds/Attn-RVQ config.",
    )
    parser.add_argument(
        "--max_sites",
        type=int,
        default=None,
        help="Maximum site index / NUM_SITES token reserved in vocabulary.",
    )
    parser.add_argument(
        "--include_space_group",
        action="store_true",
        help="Reserve legacy <SPACE_GROUP_*> tokens. Keep off for simplified DSS-seq.",
    )
    parser.add_argument(
        "--include_crystal_system",
        action="store_true",
        help="Reserve legacy <CRYSTAL_SYSTEM_*> tokens. Keep off for simplified DSS-seq.",
    )
    parser.add_argument(
        "--include_coord",
        action="store_true",
        help="Reserve legacy <COORD_*> tokens. Keep off for simplified DSS-seq.",
    )
    parser.add_argument(
        "--allow_unk",
        action="store_true",
        help="Allow unknown tokens instead of raising an error after vocabulary validation.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience measured in validation epochs. <=0 disables early stopping.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--save_last",
        action="store_true",
        help="Also save last_cry2seq.pt after every epoch.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Resume model, optimizer, best loss, and early-stop state from a checkpoint.",
    )
    parser.add_argument(
        "--objective",
        choices=("causal", "masked"),
        default="causal",
        help="Causal GPT baseline or single bidirectional Masked-HDS model.",
    )
    parser.add_argument(
        "--masked_probability",
        type=float,
        default=0.15,
        help="Fraction of non-structural HDS tokens supervised per Masked-HDS sample.",
    )
    parser.add_argument(
        "--stage_marked",
        action="store_true",
        help="Insert <STAGE_*> markers before RVQ code tokens at dataset read time.",
    )
    parser.add_argument(
        "--stage_specific_embeddings",
        action="store_true",
        help="Add a learned RVQ-stage embedding to each HDS token representation.",
    )
    parser.add_argument(
        "--stage_specific_heads",
        action="store_true",
        help="Route RVQ-code prediction positions through dedicated stage heads.",
    )
    parser.add_argument(
        "--canonicalize_hds",
        action="store_true",
        help="Canonicalize formula and numbered site blocks before tokenization.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=5,
        help="Target number of canonical HDS candidates for downstream generation metadata.",
    )
    args = parser.parse_args()

    # YAML config support (optional)
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"--config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # 将 YAML 中存在的字段写入 args（仅当 CLI 未显式提供时）
        # 规则：若 args 的值等于 argparse 默认值，则认为未显式提供，允许 YAML 覆盖。
        # 注意：dataset_dir 在 YAML 里通常必填，这里只在 args.dataset_dir 为空时覆盖。
        defaults = {
            "dataset_dir": None,
            "train_file": "train_tokens.pkl.gz",
            "val_file": "val_tokens.pkl.gz",
            "out_dir": "ckpts/cry2seq",
            "max_seq_len": 512,
            "batch_size": 32,
            "num_epochs": 50,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "grad_clip": 1.0,
            "num_workers": 4,
            "dropout": 0.0,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "seed": 42,
            "property_names": [],
            "max_property_bins": 128,
            "codebook_size": 512,
            "codebooksize_global": None,
            "codebooksize_local": None,
            "num_quantizers_global": None,
            "num_quantizers_local": None,
            "max_sites": 128,
            "include_space_group": False,
            "include_crystal_system": False,
            "include_coord": False,
            "allow_unk": False,
            "early_stop_patience": 0,
            "early_stop_min_delta": 0.0,
            "save_last": False,
            "objective": "causal",
            "masked_probability": 0.15,
            "stage_marked": False,
            "stage_specific_embeddings": False,
            "stage_specific_heads": False,
            "canonicalize_hds": False,
            "num_candidates": 5,
        }

        # CLI override for list-like property names
        if args.property_names is not None:
            args.property_names = [p.strip() for p in args.property_names.split(",") if p.strip()]

        for k, default_v in defaults.items():
            if k not in cfg:
                continue
            cur_v = getattr(args, k)
            # dataset_dir 的默认是 None
            if k == "dataset_dir":
                if cur_v is None:
                    setattr(args, k, cfg[k])
                continue
            # betas 在 argparse 里是 tuple
            if k == "betas":
                if cur_v == default_v:
                    b = cfg[k]
                    if isinstance(b, (list, tuple)) and len(b) == 2:
                        setattr(args, k, (float(b[0]), float(b[1])))
                continue
            if cur_v == default_v:
                value = cfg[k]
                if isinstance(default_v, float):
                    value = float(value)
                elif isinstance(default_v, int):
                    value = int(value)
                setattr(args, k, value)

    # dataset_dir 必填（从 CLI 或 YAML 来）
    if not args.dataset_dir:
        raise ValueError("--dataset_dir is required (provide via CLI or --config).")

    if args.property_names is not None and not isinstance(args.property_names, list):
        args.property_names = [p.strip() for p in str(args.property_names).split(",") if p.strip()]

    return args


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




def resolve_vocab_value(args, cfg: dict, key: str, default):
    """Resolve tokenizer/vocab hyperparameters from CLI first, then YAML, then default."""
    value = getattr(args, key, None)
    if value is not None:
        return value
    return cfg.get(key, default)


def validate_no_unknown_tokens(samples, tokenizer, split_name: str, max_examples: int = 20):
    """Return unknown token statistics and a few examples for debugging vocabulary mismatch."""
    unk = tokenizer.vocab.token_to_id.get("<UNK>")
    total_tokens = 0
    unk_count = 0
    examples = []
    unknown_types = set()

    for sample in samples:
        sample_id = sample.get("id", "UNKNOWN")
        tokens = sample.get("tokens", [])
        total_tokens += len(tokens)
        for tok in tokens:
            if tokenizer.vocab.token_to_id.get(tok, unk) == unk and tok != "<UNK>":
                unk_count += 1
                unknown_types.add(tok)
                if len(examples) < max_examples:
                    examples.append((sample_id, tok))

    print(
        f"[{split_name}] token check: total={total_tokens}, "
        f"unk={unk_count}, unique_unk={len(unknown_types)}"
    )
    if examples:
        print(f"[{split_name}] first unknown token examples:")
        for sample_id, tok in examples:
            print(f"  id={sample_id}, token={tok}")
    return unk_count, unknown_types


def main():
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    # 使得 `scripts`, `model` 等模块可被导入
    import sys

    sys.path.insert(0, str(project_root))

    from scripts.tokenizer import CrystalVocabConfig
    from model.crystal_token_gpt import GPT, GPTConfig

    cfg = {}
    if args.config:
        with open(Path(args.config), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / args.train_file
    val_path = dataset_dir / args.val_file

    assert train_path.is_file(), f"Train token file not found: {train_path}"
    assert val_path.is_file(), f"Val token file not found: {val_path}"

    print(f"Loading train tokens from {train_path}")
    train_samples = load_pickle(str(train_path))
    print(f"Loading val tokens from {val_path}")
    val_samples = load_pickle(str(val_path))

    # ===== 构建 tokenizer / vocab =====
    # 与当前 Attn-RVQ / cif2hds 约定保持一致：
    #   - codebook size 256
    #   - 单/多量化器都兼容
    property_names = args.property_names if args.property_names is not None else cfg.get("property_names", [])
    if isinstance(property_names, str):
        property_names = [property_names]
    max_property_bins = args.max_property_bins if args.max_property_bins is not None else int(cfg.get("max_property_bins", 128))
    # ===== IMPORTANT =====
    # 这些参数必须与 cif2hds / Attn-RVQ 的输出一致。
    # 之前写死 codebook size=256 会导致 <GLOBAL_CODE_398> 这类 token 变成 <UNK>。
    codebook_size = int(resolve_vocab_value(args, cfg, "codebook_size", 512))
    codebooksize_global = int(resolve_vocab_value(args, cfg, "codebooksize_global", codebook_size) or codebook_size)
    codebooksize_local = int(resolve_vocab_value(args, cfg, "codebooksize_local", codebook_size) or codebook_size)
    num_quantizers_global = int(resolve_vocab_value(args, cfg, "num_quantizers_global", cfg.get("num_quantizers", 1)) or 1)
    num_quantizers_local = int(resolve_vocab_value(args, cfg, "num_quantizers_local", cfg.get("num_quantizers", 1)) or 1)
    max_sites = int(resolve_vocab_value(args, cfg, "max_sites", 128))

    vocab_cfg = CrystalVocabConfig(
        num_quantizers_global=num_quantizers_global,
        codebooksize_global=codebooksize_global,
        num_quantizers_local=num_quantizers_local,
        codebooksize_local=codebooksize_local,
        max_sites=max_sites,
        max_coord=20,
        include_space_group_tokens=bool(args.include_space_group or cfg.get("include_space_group", False)),
        include_crystal_system_tokens=bool(args.include_crystal_system or cfg.get("include_crystal_system", False)),
        include_coord_tokens=bool(args.include_coord or cfg.get("include_coord", False)),
        include_hds_stage_tokens=bool(args.stage_marked),
        include_mask_token=args.objective == "masked",
        property_names=property_names,
        max_property_bins=max_property_bins,
    )
    from scripts.tokenizer import CrystalVocab, CrystalTokenizer  # type: ignore

    vocab = CrystalVocab(vocab_cfg)
    tokenizer = CrystalTokenizer(vocab)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    print(f"Vocab config: {vocab_cfg}")

    train_unk_count, train_unknown_types = validate_no_unknown_tokens(train_samples, tokenizer, "train")
    val_unk_count, val_unknown_types = validate_no_unknown_tokens(val_samples, tokenizer, "val")
    if (train_unk_count + val_unk_count) > 0 and not args.allow_unk:
        unknown_preview = sorted(train_unknown_types | val_unknown_types)[:30]
        raise ValueError(
            "Found unknown tokens. This usually means tokenizer config does not match cif2hds output. "
            f"Preview: {unknown_preview}. Use matching codebook_size/num_quantizers/max_sites/property_names, "
            "or pass the legacy include flags only if those tokens are truly present."
        )

    max_seq_len = args.max_seq_len

    # ===== Dataset / DataLoader =====
    # Structural delimiters are context anchors, not reconstruction targets in
    # Masked-HDS. RVQ / atom / count tokens remain eligible for masking.
    protected_tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<CRYSTAL>", "</CRYSTAL>",
        "<FORMULA>", "</FORMULA>", "<GLOBAL>", "</GLOBAL>",
        "<SITES>", "</SITES>", "<MASK>",
    ]
    protected_token_ids = [vocab.token_to_id[token] for token in protected_tokens if token in vocab.token_to_id]
    dataset_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        objective=args.objective,
        stage_marked=args.stage_marked,
        canonicalize=args.canonicalize_hds,
        mask_probability=args.masked_probability,
        protected_token_ids=protected_token_ids,
    )
    train_dataset = CrystalTokenDataset(train_samples, **dataset_kwargs)
    val_dataset = CrystalTokenDataset(val_samples, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ===== 构建 GPT 模型 =====
    gpt_cfg = GPTConfig(
        block_size=max_seq_len,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=True,
        attention_mode="bidirectional" if args.objective == "masked" else "causal",
        num_rvq_stages=len(RVQ_STAGE_NAMES),
        stage_specific_embeddings=args.stage_specific_embeddings,
        stage_specific_heads=args.stage_specific_heads,
    )
    model = GPT(gpt_cfg).to(device)

    # 优化器
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=tuple(args.betas),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = int(args.early_stop_patience or 0)
    early_stop_min_delta = float(args.early_stop_min_delta or 0.0)

    start_epoch = 1
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        saved_cfg = checkpoint.get("gpt_config", {})
        current_cfg = gpt_cfg.__dict__
        incompatible = {
            key: (saved_cfg.get(key), value)
            for key, value in current_cfg.items()
            if key in saved_cfg and saved_cfg[key] != value
        }
        if incompatible:
            raise ValueError(f"Resume checkpoint architecture differs from requested run: {incompatible}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        epochs_no_improve = int(checkpoint.get("epochs_no_improve", 0))
        print(f"Resumed from {resume_path}: next epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")

    print(
        f"Early stopping: patience={early_stop_patience}, "
        f"min_delta={early_stop_min_delta:g}, "
        f"enabled={early_stop_patience > 0}"
    )

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.num_epochs}", leave=False)
        for batch in pbar:
            input_ids, target_ids, input_stage_ids, target_stage_ids, loss_mask = batch
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            input_stage_ids = input_stage_ids.to(device, non_blocking=True)
            target_stage_ids = target_stage_ids.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            logits, loss = model(
                input_ids, targets=target_ids, stage_ids=input_stage_ids,
                head_stage_ids=target_stage_ids, loss_mask=loss_mask,
            )
            assert loss is not None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total_loss/max(num_batches,1)):.4f}")

        avg_train_loss = total_loss / max(num_batches, 1)

        # 验证
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val   {epoch}/{args.num_epochs}", leave=False)
            for batch in vbar:
                input_ids, target_ids, input_stage_ids, target_stage_ids, loss_mask = batch
                input_ids = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)
                input_stage_ids = input_stage_ids.to(device, non_blocking=True)
                target_stage_ids = target_stage_ids.to(device, non_blocking=True)
                loss_mask = loss_mask.to(device, non_blocking=True)
                _, loss = model(
                    input_ids, targets=target_ids, stage_ids=input_stage_ids,
                    head_stage_ids=target_stage_ids, loss_mask=loss_mask,
                )
                assert loss is not None
                val_loss_total += loss.item()
                val_batches += 1
                vbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(val_loss_total/max(val_batches,1)):.4f}")

        avg_val_loss = val_loss_total / max(val_batches, 1)

        print(
            f"Epoch {epoch}/{args.num_epochs} "
            f"- train loss: {avg_train_loss:.4f} "
            f"- val loss: {avg_val_loss:.4f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gpt_config": gpt_cfg.__dict__,
            "vocab_config": vocab_cfg.__dict__,
            "best_val_loss": best_val_loss,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "epochs_no_improve": epochs_no_improve,
            "hds_objective": args.objective,
            "stage_marked": args.stage_marked,
            "canonicalize_hds": args.canonicalize_hds,
            "num_candidates": args.num_candidates,
        }

        if args.save_last:
            last_path = Path(args.out_dir) / "last_cry2seq.pt"
            torch.save(ckpt_payload, last_path)

        # 保存最佳模型：只有超过 min_delta 的改善才算刷新 best。
        improved = avg_val_loss < (best_val_loss - early_stop_min_delta)
        if improved:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ckpt_payload.update(
                {
                    "best_val_loss": best_val_loss,
                    "epochs_no_improve": epochs_no_improve,
                }
            )
            save_path = Path(args.out_dir) / "best_cry2seq.pt"
            torch.save(ckpt_payload, save_path)
            print(f"  ✓ Saved best model to {save_path} (best_val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0:
                print(
                    f"  Early-stop counter: {epochs_no_improve}/{early_stop_patience} "
                    f"(best_val_loss={best_val_loss:.4f})"
                )

        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(
                f"[Early Stop] No validation improvement greater than {early_stop_min_delta:g} "
                f"for {early_stop_patience} consecutive epoch(s). "
                f"Stopped at epoch {epoch}. Best val loss: {best_val_loss:.4f}"
            )
            break


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python train_ar_model.py --config config/ar_model/train_cry2seq_mp2bandgap.yaml
    # CUDA_VISIBLE_DEVICES=2 python train_ar_model.py --config config/ar_model/train_cry2seq.yaml
    main()


