"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py
"""
import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 371
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # ``causal`` preserves the original GPT baseline. ``bidirectional`` is
    # used by Masked-HDS so every unmasked token can be context.
    attention_mode: str = "causal"
    # RVQ stage conditioning is optional to retain checkpoint compatibility
    # with the original causal model.
    num_rvq_stages: int = 0
    stage_specific_embeddings: bool = False
    stage_specific_heads: bool = False


class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool):
        """
        Initialize the LayerNorm module.

        :param ndim: dimensionality of the input tensor
        :param bias: whether to add a learnable bias to the output
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.is_causal = config.attention_mode == "causal"
        if config.attention_mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Unsupported attention_mode: {config.attention_mode}")
        if not self.flash and self.is_causal:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """
        Applies causal or bidirectional self-attention to the given tensor.

        :param x: tensor of shape (batch size, sequence length, embedding dimension)
        :returns: result of applying the causal self-attention operation
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        key_padding_mask = None
        if attention_mask is not None:
            # SDPA accepts an additive mask. Padding is excluded as a key for
            # every query, while valid HDS tokens retain full bidirectional
            # context in Masked-HDS mode.
            key_padding_mask = torch.zeros(B, 1, 1, T, dtype=x.dtype, device=x.device)
            key_padding_mask.masked_fill_(~attention_mask[:, None, None, :].bool(), float("-inf"))

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=key_padding_mask, dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            if key_padding_mask is not None:
                att = att + key_padding_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


def gelu(x: Tensor) -> Tensor:
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function, as used in the Google BERT and
    OpenAI GPT models. See: "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415

    :param x: the tensor to which the GELU activation function will be applied
    :returns: the result tensor after applying the GELU activation function,
              possessing the same shape as the input tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """
        Forward pass for the Transformer Block module. A Block module includes causal self-attention,
        layer normalization, and MLP, and residual connections.

        :param x: input to the transformer block
        :returns: output of the transformer block, with the same shape as in the input
        """
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.stage_embeddings = (
            nn.Embedding(config.num_rvq_stages + 1, config.n_embd)
            if config.stage_specific_embeddings else None
        )
        # Each RVQ stage gets a dedicated classifier while non-code syntax
        # keeps the shared vocabulary head (stage id 0).  This makes the
        # prediction interface stage-aware without splitting HDS into models.
        self.stage_heads = (
            nn.ModuleList(
                [nn.Linear(config.n_embd, config.vocab_size, bias=False)
                 for _ in range(config.num_rvq_stages)]
            )
            if config.stage_specific_heads else None
        )
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight
        self.padding_token_id = 0
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :returns: the number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, stage_ids=None, head_stage_ids=None, loss_mask=None, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        if self.stage_embeddings is not None:
            if stage_ids is None:
                raise ValueError("stage_ids are required when stage_specific_embeddings is enabled")
            x = x + self.stage_embeddings(stage_ids.clamp(0, self.config.num_rvq_stages))
        x = self.transformer.drop(x)
        if attention_mask is None and self.config.attention_mode == "bidirectional":
            attention_mask = idx != self.padding_token_id
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        routing_stage_ids = head_stage_ids if head_stage_ids is not None else stage_ids
        if self.stage_heads is not None and routing_stage_ids is not None:
            # Route only RVQ-code positions to their matching dynamic head.
            # Stage ids are one based so zero remains the shared syntax head.
            for stage, head in enumerate(self.stage_heads, start=1):
                positions = routing_stage_ids == stage
                if positions.any():
                    logits[positions] = head(x[positions])

        if targets is not None:
            mask = targets != self.padding_token_id
            if loss_mask is not None:
                mask = mask & loss_mask.bool()
            if not mask.any():
                raise ValueError("No valid target positions remain after applying the loss mask")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            loss = (loss * mask.view(-1)).sum() / mask.sum()
        else:
            # Keep the original generation interface: inference returns only
            # the final position even for stage-aware models.
            logits = logits[:, [-1], :]
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        num_gens=1,
        forbidden_token_ids=None,
        eos_token_id=None,
        early_stop_eos=False,
    ):
        """
        从给定前缀 idx 继续自回归生成 token 序列。

        参数
        ----
        idx : LongTensor
            形状为 (B, T) 或 (T,) 的 token id 序列，作为生成的前缀。
        max_new_tokens : int
            需要生成的新 token 数量上限。
        temperature : float
            采样温度，>1 更随机，<1 更保守。
        top_k : Optional[int]
            若不为 None，则在每一步仅在 top_k 个最高概率 token 上做采样。
        num_gens : int
            当 B=1 时，可以通过 num_gens>1 复制前缀，得到多条独立采样的序列。

        返回
        ----
        LongTensor
            形状为 (B', T + max_new_tokens) 的生成序列，其中 B'=B 或 num_gens。
        """
        device = next(self.parameters()).device
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)  # (1, T)
        idx = idx.to(device)

        B, T = idx.shape
        if num_gens > 1 and B == 1:
            # 将单条前缀复制为多条，用于并行生成多条样本
            idx = idx.repeat(num_gens, 1)
            B, T = idx.shape

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # 截断到 block_size，避免位置 embedding 越界
            idx_cond = idx[:, -self.config.block_size :]

            # 前向得到最后一个位置的 logits
            logits, _ = self(idx_cond)  # logits: (B, 1, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # 应用温度
            if temperature is not None and temperature > 0:
                logits = logits / temperature

            # 可选 top-k 采样
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                min_topk = v[:, [-1]]
                logits = torch.where(
                    logits < min_topk,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            # 可选：屏蔽不希望采样到的 token（如 <UNK>）
            if forbidden_token_ids is not None:
                for tid in forbidden_token_ids:
                    if tid is None:
                        continue
                    if 0 <= int(tid) < logits.shape[-1]:
                        logits[:, int(tid)] = float("-inf")

            # 已经结束的样本保持输出 EOS（若提供）
            if eos_token_id is not None:
                logits[finished, :] = float("-inf")
                logits[finished, int(eos_token_id)] = 0.0

            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(1) == int(eos_token_id))
                if early_stop_eos and torch.all(finished):
                    break

        return idx
