"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from .randatt.blocks import EncoderBlock, LightEncoderBlock
from .randatt.tools import causal_mask, alibi_shift

from .tools import LayerNorm

from fvcore.nn import FlopCountAnalysis, flop_count_table


def analytical_flop_counter(model, batch_size, sequence_length):
    """
    Analytically estimate FLOPs per forward pass based on the model config.
    Includes attention (with different head dims/context), MLP, and projections.

    :param model: GPTBase model (or DDP-wrapped)
    :param batch_size: int
    :param sequence_length: int
    :return: total_flops (int)
    """
    # Unwrap DDP if needed
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    config = raw_model.config

    n_embd = config.n_embd
    n_layer = config.n_layer
    n_heads_short = config.n_heads_short
    n_heads_long = config.n_heads_long
    short_heads_dim = config.short_heads_dim
    long_heads_dim = config.long_heads_dim
    head_dim = n_embd // config.n_head  # Standard head dim for V and final concat

    context_short = config.context_short or sequence_length
    context_long = config.context_long or sequence_length

    B, T, C = batch_size, sequence_length, n_embd

    def compute_attention_flops(n_heads, qk_dim, v_dim, window_size):
        if n_heads == 0:
            return 0
        # Projection FLOPs: Q, K, V projections
        flops_q_proj = B * T * C * n_heads * qk_dim  # Q projection
        flops_k_proj = B * T * C * n_heads * qk_dim  # K projection
        flops_v_proj = B * T * C * n_heads * v_dim   # V projection

        # Attention computation FLOPs
        non_zero_elements = sum(min(i + 1, window_size) for i in range(T))

        # Q @ K^T and attention weighting
        flops_qk_matmul = B * n_heads * non_zero_elements * qk_dim
        flops_softmax = B * n_heads * non_zero_elements
        flops_v_weighted_sum = B * n_heads * non_zero_elements * v_dim

        return flops_q_proj + flops_k_proj + flops_v_proj + flops_qk_matmul + flops_softmax + flops_v_weighted_sum


    def compute_mlp_flops():
        # MLP: FC1 -> Activation -> FC2
        flops_fc1 = B * T * C * (4 * C)
        flops_fc2 = B * T * (4 * C) * C
        return flops_fc1 + flops_fc2

    # FLOP count for each layer
    total_flops_per_layer = 0
    for _ in range(n_layer):
        # Short attention heads
        flops_short = compute_attention_flops(n_heads_short, short_heads_dim, head_dim, context_short)
        # Long attention heads
        flops_long = compute_attention_flops(n_heads_long, long_heads_dim, head_dim, context_long)

        # Output projection once per block (after concatenation of heads)
        flops_output_proj = B * T * C * C

        # MLP FLOPs
        flops_mlp = compute_mlp_flops()

        total_flops_per_layer += flops_short + flops_long + flops_output_proj + flops_mlp

    # Embedding layer (lookup, not matmul, so FLOPs often ignored in papers)
    flops_embedding = 0

    # Final LayerNorm
    flops_ln_final = B * T * C * 2  # Normalization involves mean/variance + scale/shift

    total_flops = n_layer * total_flops_per_layer + flops_embedding + flops_ln_final

    print(f"[Analytical Profiling] Estimated FLOPs per forward pass: {total_flops:,}")
    return total_flops









class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        #self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.context_short = config.context_short or config.sequence_length
        self.context_long = config.context_long or config.sequence_length
        self.sequence_length = config.sequence_length
        self.head_dim = self.n_embd // self.n_head
        self.n_heads_short = config.n_heads_short
        self.n_heads_long = config.n_heads_long
        self.short_heads_dim = config.short_heads_dim
        self.long_heads_dim = config.long_heads_dim
        if self.n_heads_short > 0:
            self.qk_short = nn.Linear(self.n_embd, 2 * self.n_heads_short * self.short_heads_dim, bias=config.bias)
            self.v_short = nn.Linear(self.n_embd, self.n_heads_short * self.head_dim, bias=config.bias)
        if self.n_heads_long > 0:
            print(f"Short head dim: {self.short_heads_dim}, Using long head dim: {self.long_heads_dim} for {self.n_heads_long}/{self.n_heads_short + self.n_heads_long} heads")
            self.qk_long = nn.Linear(self.n_embd, 2 * self.n_heads_long * self.long_heads_dim, bias=config.bias)
            self.v_long = nn.Linear(self.n_embd, self.n_heads_long * self.head_dim, bias=config.bias)

        self.register_buffer("mask_short", self._build_causal_window_mask(self.context_short))
        self.register_buffer("mask_long", self._build_causal_window_mask(self.context_long))

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))
    def _build_causal_window_mask(self, context_window):
            T = self.sequence_length
            i_idx = torch.arange(T).unsqueeze(1)
            j_idx = torch.arange(T).unsqueeze(0)
            mask = ((i_idx - j_idx) < 0) | ((i_idx - j_idx) >= context_window)
            # Note: mask should be in float32 to match what F.scaled_dot_product_attention expects for `attn_mask`
            return mask.float() * torch.finfo(torch.float32).min  # shape: (T, T)
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        device = x.device

        y_short = None
        if self.n_heads_short > 0:
            qk_short = self.qk_short(x).split(self.n_heads_short * self.short_heads_dim, dim=2)
            q_short, k_short = [t.view(B, T, self.n_heads_short, self.short_heads_dim).transpose(1, 2) for t in qk_short]
            v_short = self.v_short(x).view(B, T, self.n_heads_short, self.head_dim).transpose(1, 2)
            mask_short = self.mask_short[:T, :T] if self.context_short < T else None
            y_short = F.scaled_dot_product_attention(q_short, k_short, v_short, attn_mask=mask_short, dropout_p=self.dropout, is_causal=mask_short is None)

        y_long = None
        if self.n_heads_long > 0:
            qk_long = self.qk_long(x).split(self.n_heads_long * self.long_heads_dim, dim=2)
            q_long, k_long = [t.view(B, T, self.n_heads_long, self.long_heads_dim).transpose(1, 2) for t in qk_long]
            v_long = self.v_long(x).view(B, T, self.n_heads_long, self.head_dim).transpose(1, 2)
            mask_long = self.mask_long[:T, :T] if self.context_long < T else None
            y_long = F.scaled_dot_product_attention(q_long, k_long, v_long, attn_mask=mask_long, dropout_p=self.dropout, is_causal=mask_long is None)

        # Combine the outputs from the two groups (concatenating along the head dimension)
        if y_short is not None and y_long is not None:
            y = torch.cat([y_short, y_long], dim=1)
        elif y_short is not None:
            y = y_short
        elif y_long is not None:
            y = y_long

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        #print(f"Initializing Block with attention_type={config.attention_type}")

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn_block = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x_attn = self.attn_block(self.ln_1(x))
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPTBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        # Select block type based on config
        # if config.attention_type == "random_block":
        #     block_cls = LightEncoderBlock
        # elif config.attention_type == "self":
        #     block_cls = EncoderBlock
        # elif config.attention_type == "base":
        #     block_cls = Block  # Default fallback or custom Block type

        if config.attention_type == "random_block":
            block_cls = lambda **kwargs: LightEncoderBlock(
                model_dim=config.n_embd,
                block_dim=getattr(config, "block_dim", 100),
                n_heads=config.n_head,
                dropout_rate=config.dropout,
                act=nn.GELU(),
                bias=config.bias,
                eps=getattr(config, "eps", 0.0),
                gamma=getattr(config, "gamma", 0.9),
                use_cumsum=getattr(config, "use_cumsum", False),
                trainable_cumsum=config.trainable_cumsum,
                use_softmax_cumsum=config.use_softmax_cumsum,
            )
        elif config.attention_type == "self":
            block_cls = lambda **kwargs: EncoderBlock(
                model_dim=config.n_embd,
                n_heads=config.n_head,
                dropout_rate=config.dropout,
                act=nn.GELU(),
                bias=config.bias,
                eps=getattr(config, "eps", 0.0),
                gamma=getattr(config, "gamma", 0.9),
                use_cumsum=getattr(config, "use_cumsum", False),
                trainable_cumsum=config.trainable_cumsum,
                use_softmax_cumsum=config.use_softmax_cumsum,
            )
        elif config.attention_type == "base":
            block_cls = lambda **kwargs: Block(config)


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) if config.attention_type == "base" 
            # else 
            # nn.ModuleList([block_cls(model_dim=config.n_embd, n_heads=config.n_head,dropout_rate=config.dropout,block_dim=getattr(config, "block_dim", None),bias=config.bias) for _ in range(config.n_layer) ]),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) if config.attention_type == "base" 
            else nn.ModuleList([block_cls() for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or (pn.endswith('lin_out.weight') and config.attention_type in {"self" , "random_block"}):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
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
        elif isinstance(module, LayerNorm):  # Explicitly handle your custom LayerNorm
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            if self.config.attention_type == "base":
                x = block(x)
            else:
                mask = causal_mask(x)
                x = block(x, mask=mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        return {'logits': logits, 'loss': loss}

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        # need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES


        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        if self.config.trainable_cumsum:
            for name, param in self.named_parameters():
                if "weights_eps" in name or "weights_gamma" in name:
                    no_decay.add(name)


        for mn, m in self.named_modules():
            if isinstance(m, LayerNorm):  # Match your custom LayerNorm explicitly
                for pn, p in m.named_parameters():
                    fpn = f"{mn}.{pn}" if mn else pn
                    no_decay.add(fpn)


        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
