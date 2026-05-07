"""
CrystalVQVAE with an additional physical-property prediction head.

Design goals:
- Keep the original `CrystalVQVAE` intact; this file provides a new model for
  pretrain->finetune workflows.
- The property head takes the *aggregated global quantized latent* `z_g` as input.
- The property is discretized (classification), so we use cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.crystal_vq_vae import CrystalVQVAE


class PropCrystalVQVAE(CrystalVQVAE):
    """
    Extends `CrystalVQVAE` by adding a classification head on the aggregated global latent `z_g`.

    Notes
    -----
    - `prop_num_classes`: number of bins/classes (e.g. band gap bins: 6).
    - During pretraining (no labels), call forward with `prop_targets=None` or `use_prop_head=False`.
    """

    def __init__(
        self,
        *args,
        prop_num_classes: int = 6,
        decoder_prop_dims: list[int] | None = None,
        prop_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prop_num_classes = int(prop_num_classes)
        self.prop_loss_weight = float(prop_loss_weight)

        # ------------------------------------------------------------
        # Property head (as requested): alternating MLP + Transformer layers.
        #
        # Input dim is latent_dim*2: we use concat([g, z_g]) where:
        #   - g   : encoder global embedding (B, hidden_dim)
        #   - z_g : aggregated quantized global latent (B, hidden_dim)
        # so input is (B, 2*hidden_dim)
        # ------------------------------------------------------------
        self.decoder_prop_dims: list[int] = list(decoder_prop_dims) if decoder_prop_dims is not None else [256, 256]
        self.decoder_prop = nn.ModuleList()
        self.decoder_prop_dims.insert(0, self.hidden_dim * 2)

        for i in range(len(self.decoder_prop_dims) - 1):
            self.decoder_prop.append(
                nn.Sequential(
                    nn.Linear(self.decoder_prop_dims[i], self.decoder_prop_dims[i + 1]),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.decoder_prop.append(
                nn.TransformerEncoderLayer(
                    d_model=self.decoder_prop_dims[i + 1],
                    nhead=4,
                    dim_feedforward=256,
                    batch_first=True,
                )
            )

        # last linear: output logits for classification
        self.decoder_prop.append(nn.Sequential(nn.Linear(self.decoder_prop_dims[-1], self.prop_num_classes)))

    def predict_prop_logits(self, g: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """
        g   : (B, D)
        z_g : (B, D)
        returns logits: (B, C)
        """
        x = torch.cat([g, z_g], dim=-1)  # (B, 2D)

        # MLP produces (B, d), Transformer expects (B, S, d); we use S=1.
        seq = None
        for layer in self.decoder_prop:
            if isinstance(layer, nn.TransformerEncoderLayer):
                if seq is None:
                    seq = x.unsqueeze(1)  # (B, 1, d)
                seq = layer(seq)  # (B, 1, d)
                x = seq.squeeze(1)  # (B, d)
            else:
                x = layer(x)
                seq = None
        return x

    def forward(
        self,
        atom_types,
        frac_coords,
        lattice,
        num_atoms,
        node2batch,
        global_targets,
        local_targets,
        prop_targets: torch.Tensor | None = None,
        use_prop_head: bool = True,
    ):
        # ---- base path (same as CrystalVQVAE.forward) ----
        g, h = self.encode(atom_types, frac_coords, lattice, num_atoms, node2batch)
        z_g, z_l, indices_g, indices_l, commitment_loss = self.quantize(g, h)
        outputs = self.decode(z_g, z_l)

        losses: dict[str, torch.Tensor] = {}

        # recon losses
        recon_global_loss = F.mse_loss(outputs["g_recon"], g)
        recon_local_loss = F.mse_loss(outputs["h_recon"], h)
        losses["recon_global_loss"] = recon_global_loss
        losses["recon_local_loss"] = recon_local_loss

        # aux: space group
        space_group_loss = F.cross_entropy(outputs["space_group_logits"], global_targets)
        losses["space_group_loss"] = space_group_loss

        # aux: CN
        cn_targets = []
        cn_mask = []
        for structure_targets in local_targets:
            for atom_target in structure_targets:
                cn = atom_target.get("cn", None)
                if cn is None or cn < 0 or cn >= self.max_cn:
                    cn_targets.append(0)
                    cn_mask.append(0)
                else:
                    cn_targets.append(int(cn))
                    cn_mask.append(1)
        cn_targets_t = torch.tensor(cn_targets, dtype=torch.long, device=outputs["cn_logits"].device)
        cn_mask_t = torch.tensor(cn_mask, dtype=torch.float, device=outputs["cn_logits"].device).bool()

        if cn_mask_t.sum() > 0:
            cn_loss = F.cross_entropy(outputs["cn_logits"][cn_mask_t], cn_targets_t[cn_mask_t])
        else:
            cn_loss = torch.tensor(0.0, device=outputs["cn_logits"].device, requires_grad=True)
        losses["cn_loss"] = cn_loss

        # vq commitment
        losses["commitment_loss"] = commitment_loss

        # bookkeeping
        losses["indices_g"] = indices_g
        losses["indices_l"] = indices_l
        losses["emb_global"] = g
        losses["emb_local"] = h

        # ---- property head (classification) ----
        prop_loss = None
        prop_logits = None
        if use_prop_head and (prop_targets is not None):
            prop_targets = prop_targets.long().to(z_g.device)
            prop_logits = self.predict_prop_logits(g, z_g)  # (B, C)
            prop_loss = F.cross_entropy(prop_logits, prop_targets)
            losses["prop_logits"] = prop_logits
            losses["prop_loss"] = prop_loss

        # ---- total loss ----
        total_loss = (
            self.loss_weights["recon_global"] * recon_global_loss
            + self.loss_weights["recon_local"] * recon_local_loss
            + self.loss_weights["space_group"] * space_group_loss
            + self.loss_weights["cn"] * cn_loss
            + self.loss_weights["commitment"] * commitment_loss
        )
        if prop_loss is not None:
            total_loss = total_loss + (self.prop_loss_weight * prop_loss)
        losses["total_loss"] = total_loss

        return losses

