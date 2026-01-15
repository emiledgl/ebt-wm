from typing import Optional

import math
import torch
from torch import nn
from torch.nn import functional as F

from models.utils import EBTModelArgs, init_whole_model_weights
from models.modules import DyT, RMSNorm, precompute_freqs_cis, apply_rotary_emb

class EBTAttention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: EBTModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of query heads.
            n_kv_heads (int): Number of key and value heads.
            head_dim (int): Dimension size of each attention head.
            wq (Linear): Linear transformation for queries.
            wk (Linear): Linear transformation for keys.
            wv (Linear): Linear transformation for values.
            wo (Linear): Linear transformation for output.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wq, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wk, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wv, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs (torch.Tensor): Precomputed frequency tensor.
            attn_mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        # """
        bsz, full_seqlen, _ = x.shape # full_seqlen includes real embeds and pred embeds -> 2S
        seqlen = (full_seqlen)//2 # S
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_kv_heads, self.head_dim)
        
        # _o is for original attention stuff
        xq_o = xq[:, :seqlen, :, :] #B, S, N, H
        xk_o = xk[:, :seqlen, :, :]
        xv_o = xv[:, :seqlen, :, :]
        
        # _p is for predicted attention stuff
        xq_p = xq[:, seqlen:, :, :] #B, S, N, H
        xk_p = xk[:, seqlen:, :, :]
        xv_p = xv[:, seqlen:, :, :]
        
        
        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs[:seqlen])
        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs[1:seqlen+1]) # use 2 since are the next preds and also have time embeddings and thus need to condition on two tokens        


        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S, S
        if attn_mask is not None:
            #this mask needs to be S, S, was S+1, S+1
            o_mask = attn_mask[:-1, :-1] 
            scores_o = scores_o + o_mask  # (bs, n_local_heads, S, S)
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, S, head_dim)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # B, S, D
        
        
        # seqlen here is S which = original_seqlen
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        
        values_p = xv_p.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S, S; this uses xq_p and keys_o since for every next pred calcs similarity to all prev words

        temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device) # B, N, S, 1; is used since context_length = original_length +1, superdiag needs this
        scores_p = torch.cat((scores_p, temp_append), dim = -1)# is B, N, S, S+1
        
        insertion_superdiagonal = (xq_p * keys_p).sum(dim = 3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype) # B, N, S
        
        superdiag_rows = torch.arange(scores_p.shape[2]) # [0, ..., S-1] len(S)
        superdiag_cols = torch.arange(1, scores_p.shape[3]) # [1, ..., S] len(S)
        
        # first remove superdiagonal values so doesnt use attention to future tokens--prevents leakage of probability mass
        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device) # for zeroing out superdiag since dont want to include in matmul, do this in differentiable way
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask
        
        # then set diagonal to next pred self attention scores in differentiable way
        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask         
        
        if attn_mask is not None:
            p_mask = attn_mask[1:, :]  #S, S+1
            scores_p = scores_p + p_mask
        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)
        
        #Q: why do I need to extract superdiagonal why cant i just do matmul after? A: its bc would need same subsequence in value matrix but dont have it, have original subsequence and then seperately all next preds
        scores_p_superdiagonal = scores_p.diagonal(offset=1, dim1=2, dim2=3).clone() # is B, N, S; basically how much each token on this superdiag should attent to itself; clone since dont want mask to change this
        
        scores_p = scores_p * diagonal_removal_mask # keeps scores_p as is except for superdiagonal which is next preds attention to selves, cant multiply these naively by values_p or values_o
        
        scores_p = scores_p[:, :, :, :-1] # B, N, S, S now; next preds/scores_p_superdiagonal was why needed extra col earlier (temp_append)
        output_p = torch.matmul(scores_p, values_o) # B, N, S, H; is how next preds attend to all original previous tokens;
        
        #next_pred_self_attention is to get self attention based on extracted superdiagonal and the values matrix (for predictions)
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim = -1) # B, N, S, H this is for weighted sum of each next pred to its final embed rep.
        
        output_p = output_p + next_pred_self_attention # B, N, S, H adding this is adding the aspect of each next pred embedding attending to itself
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # B, S, D
        
        #return linear projection of concatted outputs
        output = torch.cat((output_o, output_p), dim = 1) # B, 2S, D
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        ebt_act_func: str = "silu",
        weight_initialization_gain: float = 1.0
    ):
        super().__init__()
        hidden_dim = dim if ffn_dim_multiplier is None else int(dim*ffn_dim_multiplier)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w1, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(self.w2, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w3, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.act_func = {
            "silu": F.silu,
            "relu": F.relu,
            "gelu": F.gelu,
            "elu": F.elu
        }[ebt_act_func]

    def forward(self, x):
        return self.w2(self.act_func(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: EBTModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = EBTAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            weight_initialization=args.weight_initialization,
            ebt_act_func=args.ebt_act_func,
            weight_initialization_gain=args.weight_initialization_gain
        )
        self.layer_id = layer_id
        if args.ebt_norm == "rms":
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        elif args.ebt_norm == "layer":
            self.attention_norm = nn.LayerNorm(args.dim)
            self.ffn_norm = nn.LayerNorm(args.dim)
        elif args.ebt_norm == "none":
            self.attention_norm = nn.Identity()
            self.ffn_norm = nn.Identity()
        elif args.ebt_norm == "dyt":
            self.attention_norm = DyT(args.dim, alpha_init_value=args.dyt_alpha_init)
            self.ffn_norm = DyT(args.dim, alpha_init_value=args.dyt_alpha_init)
        else:
            raise ValueError(f"Invalid ebt_norm value: {args.ebt_norm}")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # x has shape B, 2S, D
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBT(nn.Module):
    def __init__(self, params: EBTModelArgs, max_mcmc_steps):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.

        Attributes:
            params (EBTModelArgs): Model configuration parameters.
            n_layers (int): Number of layers in the model.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        if params.ebt_norm == "rms":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        elif params.ebt_norm == "layer":
            self.norm = nn.LayerNorm(params.dim)
        elif params.ebt_norm == "none":
            self.norm = nn.Identity()
        elif params.ebt_norm == "dyt":
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable=False) # no learnable bias here since grad cant be computed for a final bias term in EBT
        else:
            raise ValueError(f"Invalid ebt_norm value: {params.ebt_norm}")

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias=False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(self, embeddings: torch.Tensor, mcmc_step = 0):
        """
        Perform a forward pass through the Transformer model.

        Args:
            embeds (torch.Tensor): Embeddings (instead of tokens since is for vision).
            mcmc_step (int): Current MCMC step. Defaults to 0.

        Returns:
            torch.Tensor: Output energies after applying the Transformer model.

        """
        _bsz, seqlen = embeddings.shape[:2]
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device=embeddings.device, dtype=torch.long)
        time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1) # needs to be expanded to B, 1, D
        embeddings = embeddings + time_embeddings # B, 2S, D
        
        seqlen = (seqlen+2) // 2 # do this since passed in seqlen is 2S so add 2 div 2 = S+1 which corresponds to concatting pred & next pred
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=embeddings.device
            )
            mask = torch.triu(mask, diagonal=1)

            for i, layer in enumerate(self.layers):
                embeddings = layer(embeddings, freqs_cis, mask)
            embeddings = self.norm(embeddings)
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies