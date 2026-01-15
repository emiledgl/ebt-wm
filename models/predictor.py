import math

import torch
import torch.nn as nn

from models.derf import Dynamic_erf
from models.modules import ACBlock as Block
from models.modules import precompute_freqs, build_action_block_causal_attention_mask
from timm.layers import trunc_normal_


class TransformerWM(nn.Module):
    """Transformer-based World Model for action-conditioned video prediction."""

    def __init__(
        self,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_std=0.02,
        use_silu=True,
        wide_silu=True,
        use_derf=False,
        use_activation_checkpointing=False,
        state_len=64,
        max_seq_len=1024,
        action_dim=8,
        **kwargs
    ):
        super().__init__()

        self.state_len = state_len

        T_max = max_seq_len // (state_len + 1)
        t_range = torch.arange(T_max)
        i_range = torch.arange(state_len + 1)
        t_coords, i_coords = torch.meshgrid(t_range, i_range, indexing='ij')
        t_coords = t_coords.flatten()[:max_seq_len]
        i_coords = i_coords.flatten()[:max_seq_len]
        
        # Compute frequency components for PoPE (half head_dim because 2D)
        t_freqs = precompute_freqs((predictor_embed_dim // num_heads) // 2, t_coords)
        i_freqs = precompute_freqs((predictor_embed_dim // num_heads) // 2, i_coords)
        self.register_buffer('t_freqs', t_freqs) # (2, max_seq_len, head_dim/2)
        self.register_buffer('i_freqs', i_freqs) # (2, max_seq_len, head_dim/2)

        # Causal attention mask
        self.attn_mask = build_action_block_causal_attention_mask(T_max, state_len, action_tokens=1)

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(action_dim, predictor_embed_dim)
        norm_layer = Dynamic_erf if use_derf else nn.RMSNorm

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        # Attention Blocks
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        # initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        g_block_id = 0 # global block counter across layers
        for block in self.predictor_blocks:
            rescale(block.attn.proj.weight.data, g_block_id + 1)
            rescale(block.mlp.fc2.weight.data, g_block_id + 1)
            g_block_id += 1

    def forward(self, states, actions):
        """
        Args:
            states: (B, (T*K), D) tensor of state embeddings
            actions: (B, T, A) tensor of action embeddings
        """
        # Map tokens to predictor dimensions
        s = self.predictor_embed(states)
        B, N, D = s.size()
        T = N // self.state_len

        # Interleave action tokens
        a = self.action_encoder(actions).unsqueeze(2)
        s = s.view(B, T, self.state_len, D)  # [B, T, K, D]
        x = torch.cat([s, a], dim=2).flatten(1, 2)  # [B, T*(K+1), D]

        freqs = (self.t_freqs[:, :x.size(1), :], self.i_freqs[:, :x.size(1), :])
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    freqs=freqs,
                    attn_mask=attn_mask,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    freqs=freqs,
                    attn_mask=attn_mask,
                )

        # Split out action and frame tokens
        x = x.view(B, T, self.state_len + 1, D)  # [B, T, K+1, D]
        x = x[:, :, :self.state_len, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x