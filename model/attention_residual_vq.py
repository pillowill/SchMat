import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class ResidualStageAttentionGate(nn.Module):
    """
    QKV attention gate similar to CodebookAttentionAggregator but for one residual stage.

    It attends over 2 tokens: [residual_token, quantized_token], and uses the
    attention mass on quantized_token as alpha in (0, 1).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.learnable_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.scale = hidden_dim ** -0.5

    def forward(self, residual_flat: torch.Tensor, quantized_flat: torch.Tensor) -> torch.Tensor:
        # Inputs are flattened to (M, D)
        tokens = torch.stack([residual_flat, quantized_flat], dim=1)  # (M, 2, D)
        m = tokens.shape[0]

        query = self.query_proj(self.learnable_query.expand(m, -1, -1))
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)

        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # (M, 1, 2)
        attn_weights = F.softmax(scores, dim=-1)

        aggregated = torch.matmul(attn_weights, values).squeeze(1)  # (M, D)
        fused = self.output_proj(aggregated)                        # (M, D)
        alpha = self.alpha_head(fused)                              # (M, 1)
        return alpha


class AttentionResidualVQ(nn.Module):
    """
    Residual VQ variant with per-layer attention-weighted residual updates.

    Compared with vanilla ResidualVQ residual <- residual - quantized,
    this module uses residual <- residual - alpha * quantized where alpha is
    produced by a QKV attention gate per quantizer stage.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_quantizers: int,
        codebook_size: int,
        commitment_weight: float = 0.0,
        sample_codebook_temp: float = 0.1,
        gate_dropout: float = 0.1,
        **vq_kwargs,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.sample_codebook_temp = float(sample_codebook_temp)

        # Keep a VectorQuantize layer per residual stage for checkpoint compatibility.
        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=dim,
                    codebook_size=codebook_size,
                    commitment_weight=commitment_weight,
                    **vq_kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

        # Per-stage QKV attention gate to compute alpha in (0, 1).
        self.gates = nn.ModuleList(
            [
                ResidualStageAttentionGate(dim, dropout=gate_dropout)
                for _ in range(num_quantizers)
            ]
        )
        self.last_alpha_means: list[float] = []
        self.last_alpha_stds: list[float] = []

    @staticmethod
    def _flatten_seq(x: torch.Tensor):
        # Supports both (B, N, D) and (N, D)
        if x.dim() == 2:
            n, d = x.shape
            return x, (n,), False
        if x.dim() == 3:
            b, n, d = x.shape
            return x.reshape(b * n, d), (b, n), True
        raise ValueError(f"Unexpected input shape {tuple(x.shape)}")

    @staticmethod
    def _unflatten_seq(x: torch.Tensor, shape_meta, had_batch_seq: bool):
        if not had_batch_seq:
            return x
        b, n = shape_meta
        return x.reshape(b, n, -1)

    def _layer_embed(self, layer: nn.Module) -> torch.Tensor:
        embed = layer._codebook.embed
        if embed.dim() == 3 and embed.shape[0] == 1:
            embed = embed[0]
        return embed

    def forward(self, x: torch.Tensor, sample_codebook_temp: float | None = None):
        # x: (B, N, D) in current CrystalVQVAE usage
        residual = x
        quantized_out = torch.zeros_like(x)

        all_indices = []
        all_losses = []
        alpha_means = []
        alpha_stds = []

        for stage, (vq, gate) in enumerate(zip(self.layers, self.gates)):
            quantized, indices, loss = vq(
                residual,
                sample_codebook_temp=self.sample_codebook_temp if sample_codebook_temp is None else sample_codebook_temp,
            )

            residual_flat, shape_meta, had_batch_seq = self._flatten_seq(residual)
            quantized_flat, _, _ = self._flatten_seq(quantized)

            # alpha comes from a QKV gate similar to vq_aggregator attention style.
            alpha_flat = gate(residual_flat, quantized_flat)  # (M, 1)

            if had_batch_seq:
                b, n = shape_meta
                alpha = alpha_flat.reshape(b, n, 1)
            else:
                alpha = alpha_flat

            alpha_means.append(float(alpha.mean().detach().item()))
            alpha_stds.append(float(alpha.std().detach().item()))

            weighted_quantized = alpha * quantized

            residual = residual - weighted_quantized
            quantized_out = quantized_out + weighted_quantized

            all_indices.append(indices)
            all_losses.append(loss.mean())

        # (B, N, Q) or (N, Q)
        indices = torch.stack(all_indices, dim=-1)
        # scalar commitment-like objective compatible with existing training code
        loss = torch.stack(all_losses).mean()

        self.last_alpha_means = alpha_means
        self.last_alpha_stds = alpha_stds

        return quantized_out, indices, loss

    @torch.no_grad()
    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns per-quantizer codes in shape (Q, B, N, D) or (Q, N, D)
        to stay compatible with existing aggregator-side transpose handling.
        """
        if indices.dim() == 2 and indices.shape[-1] == self.num_quantizers:
            # (B, Q)
            b, q = indices.shape
            n = 1
            flat = indices.unsqueeze(1)
            out_has_seq = False
        elif indices.dim() == 3 and indices.shape[-1] == self.num_quantizers:
            # (B, N, Q)
            b, n, q = indices.shape
            flat = indices
            out_has_seq = True
        else:
            raise ValueError(f"Unexpected indices shape {tuple(indices.shape)}")

        if q != self.num_quantizers:
            raise ValueError(f"Expected last dim {self.num_quantizers}, got {q}")

        codes = []
        for stage, layer in enumerate(self.layers):
            embed = self._layer_embed(layer)
            stage_idx = flat[..., stage].long()  # (B, N)
            stage_codes = embed[stage_idx]       # (B, N, D)
            codes.append(stage_codes)

        stacked = torch.stack(codes, dim=0)      # (Q, B, N, D)

        if out_has_seq:
            return stacked

        # Keep parity with earlier usage where seq_len=1 is usually removed.
        return stacked.squeeze(2)                # (Q, B, D)

    @torch.no_grad()
    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_indices(indices)
        if codes.dim() == 3:
            # (Q, B, D)
            return codes.sum(dim=0)
        # (Q, B, N, D)
        return codes.sum(dim=0)


