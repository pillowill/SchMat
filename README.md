# SchMat

SchMat: A Hierarchical Schema–Conditioned Generative Model for Crystalline Materials:
1) learn discrete structural semantics with Attn-RVQ,
2) model schema sequences with an autoregressive prior,
3) generate geometry with schema-conditioned diffusion.

This README is a concise usage guide aligned with the paper method and the current codebase.

## Method at a glance

- `Attn-RVQ` learns global/local discrete codes from crystal structures, with residual quantization and attention-based stage gating.
- `HDS-seq` organizes global and site-level codes into a structured token sequence.
- `AR prior` (GPT-style) models the sequence distribution `p_theta(S)`.
- `Conditional diffusion` maps sampled HDS codes back to semantic embeddings and generates lattice matrix + fractional coordinates.

## Repository map (main files)

- `train_attn_rvq.py`: train Attn-RVQ (VQ stage).
- `scripts/cif2token.py`: convert structures + VQ codes to HDS tokens.
- `train_ar_model.py`: train AR schema model on tokenized HDS sequences.
- `scripts/generate_tokens_from_prompts.py`: sample HDS tokens from trained AR model.
- `scripts/gpt_tokens_to_vq_indices.py`: parse generated tokens to VQ indices (for diffusion conditioning).
- `train_cond_diff.py`: train schema-conditioned diffusion model.

Configs:
- `config/attn_rvq/`
- `config/ar_model/`
- `config/cond_diffusion/`

## Environment

Recommended: create from `schmat_env.yml` if available in your setup.

```powershell
conda env create -f .\schmat_env.yml
conda activate schmat
```

Or with venv (minimal):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pyyaml numpy tqdm torch torchvision torchaudio
```

## Data requirement (minimum)

- CIF CSV mode is used by training scripts.
- For VQ and diffusion training, provide either:
  - split files: `train.csv`, `val.csv` (and optional `test.csv`), or
  - single file: `data.csv` with `use_single_csv: true` in config.
- Dataset parsing/targets are implemented in `data/data.py` (`CIFData`, `collate_pool`).

## Training flow

### Stage 1: Train Attn-RVQ

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python .\train_attn_rvq.py --config .\config\attn_rvq\train_cryvqvae_mpts_52.yaml --save_dir .\ckpts\mpts_52\attn_rvq --wandb
```

Output: `best_model.pth` in `--save_dir`.

### Stage 2: Build HDS-seqs + train AR model

1) Convert dataset to HDS token sequences using trained VQ model.

```powershell
python .\scripts\cif2token.py --config .\config\ar_model\train_cry2seq.yaml
```

2) Train AR schema model.

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python .\train_ar_model.py --config .\config\ar_model\train_cry2seq.yaml
```

Output: `best_cry2seq.pt` in AR output directory.

### Stage 3: Train conditional diffusion

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python .\train_cond_diff.py --config .\config\cond_diffusion\train_cond_diffusion_mpts_52.yaml --save_dir .\ckpts\mpts_52\cond_diffusion --wandb
```

Output: `best_diffusion.pth` in `--save_dir`.

## Inference-related scripts

- Generate schema tokens from prompts:
  - `scripts/generate_tokens_from_prompts.py`
- Convert generated tokens to diffusion conditions:
  - `scripts/gpt_tokens_to_vq_indices.py`
- Sample/evaluate from HDS:
  - `scripts/sample_from_hds.py`
  - `scripts/evaluate.py`

## Practical notes

- Windows DataLoader issues: set smaller `workers` (e.g., `0` or `1`) in YAML.
- GPU OOM: reduce `batch_size` and/or model dimensions.
- Keep tokenizer/codebook settings consistent across VQ, tokenization, and AR training.
- For exact equations and definitions, see `method.txt` and the paper Methods section.
