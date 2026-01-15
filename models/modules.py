import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath
from models.derf import Dynamic_erf



def precompute_freqs(dim, pos, base_wavelength=10000.0):
        c = torch.arange(dim).float()
        freqs = base_wavelength ** (-c / dim)
        phases = pos[:, None] * freqs[None, :]
        return torch.stack([torch.cos(phases), torch.sin(phases)], dim=0)  # (2, seq_len, dim)



def rotate_queries_or_keys(x, freqs):
        """
        Apply PoPE encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, dim)
            freqs: Frequency components of shape (2, seq_len, dim)
        
        Returns:
            Complex-valued tensor in Cartesian form (real, imag)
            tuple of shape: (batch_size, num_heads, seq_len, dim)
        """
        
        # Apply softplus to get magnitudes
        magnitudes = F.softplus(x)  # (batch_size, num_heads, seq_len, dim)

        # Extract cosine and sine components
        cos, sin = freqs[0][None, None, :, :], freqs[1][None, None, :, :]  # (1, 1, seq_len, dim)

        # Convert to Cartesian coordinates
        real = magnitudes * cos # (batch_size, num_heads, seq_len, dim)
        imag = magnitudes * sin # (batch_size, num_heads, seq_len, dim)
        
        return real, imag



def apply_bias_to_keys(k_real, k_imag, bias):
        """
        Apply learnable phase bias to keys using complex rotation
        """

        # Clamp bias to [-2pi, 0]
        bias = torch.clamp(bias, min=-2 * math.pi, max=0)

        # Shape for broadcasting: (1, 1, 1, head_dim)
        bias = bias[None, None, None, :]

        # Precompute rotation terms
        cos_b = torch.cos(bias)
        sin_b = torch.sin(bias)

        # Complex rotation
        real_new = k_real * cos_b - k_imag * sin_b
        imag_new = k_real * sin_b + k_imag * cos_b

        return real_new, imag_new



def build_action_block_causal_attention_mask(T, K, action_tokens=1):
    N_T =  K + action_tokens
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)
    


class PoPE2DAttention(nn.Module):
    """
    Polar Coordinate Positional Embedding (PoPE) Attention Module
    
    Args:
        dim: Dimension of the input features
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to QKV projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        is_causal: If True, apply causal masking for autoregressive tasks
    """
    
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
    ):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_causal = is_causal
        
        # Learnable bias for each frequency component
        self.pope_bias1 = nn.Parameter(torch.zeros(self.head_dim // 2))
        self.pope_bias2 = nn.Parameter(torch.zeros(self.head_dim // 2))

    
    def forward(self, x, freqs, attn_mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            freqs: Frequency components for PoPE encoding, tuple of tensors with shape (2, seq_len, head_dim/2)
            attn_mask: Optional attention mask (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        
        # Apply PoPE encoding to Q and K
        freqs1, freqs2 = freqs # Each of shape (2, seq_len, head_dim/2)
        d_half = self.head_dim // 2
        q_real_1, q_imag_1 = rotate_queries_or_keys(q[..., :d_half], freqs=freqs1)
        k_real_1, k_imag_1 = rotate_queries_or_keys(k[..., :d_half], freqs=freqs1)
        q_real_2, q_imag_2 = rotate_queries_or_keys(q[..., d_half:], freqs=freqs2)
        k_real_2, k_imag_2 = rotate_queries_or_keys(k[..., d_half:], freqs=freqs2)

        # Apply bias to K phases (separate bias for each dimension)
        k_real_1, k_imag_1 = apply_bias_to_keys(k_real_1, k_imag_1, self.pope_bias1)
        k_real_2, k_imag_2 = apply_bias_to_keys(k_real_2, k_imag_2, self.pope_bias2)

        # Concatenate both dimensions
        q_real = torch.cat([q_real_1, q_real_2], dim=-1)
        q_imag = torch.cat([q_imag_1, q_imag_2], dim=-1)
        k_real = torch.cat([k_real_1, k_real_2], dim=-1)
        k_imag = torch.cat([k_imag_1, k_imag_2], dim=-1)

        # slower to compute with einsum
        # scores = (
        #     torch.einsum('bhqd,bhkd->bhqk', q_real, k_real) +
        #     torch.einsum('bhqd,bhkd->bhqk', q_imag, k_imag)
        # )
        
        # Compute attention scores
        q_combined = torch.cat([q_real, q_imag], dim=-1) # Shape: (b, h, q, 2d)
        k_combined = torch.cat([k_real, k_imag], dim=-1) # Shape: (b, h, k, 2d)
        scores = torch.matmul(q_combined, k_combined.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        # Apply custom mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
            # Support both boolean masks and float masks
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            else:
                scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        
        # Compute output
        output = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads and reshape
        output = output.transpose(1, 2).contiguous().view(B, N, self.dim)
        
        # Final projection
        output = self.proj(output)
        output = self.proj_drop(output)
        
        return output
    


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        wide_silu=True,
        norm_layer=Dynamic_erf,
        **kwargs,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = PoPE2DAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs, attn_mask=None):
        y = self.attn(self.norm1(x), freqs=freqs,attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=(self.attn_drop_prob if self.training else 0.0),
            is_causal=self.is_causal,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        wide_silu=True,
        norm_layer=Dynamic_erf,
        is_causal=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        y = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, attn_mask=None):
        B, n, C = x.shape
        N = context.shape[1]

        q = self.q(x).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_heads, N, D]

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=(self.attn_drop_prob if self.training else 0.0),
            is_causal=False,
        )

        x = x.transpose(1, 2).reshape(B, n, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.SiLU,
        wide_silu=True,
        norm_layer=Dynamic_erf,
        **kwargs,
    ):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_gate = nn.Parameter(torch.zeros(1))

        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context, attn_mask=None):
        x = x + self.cross_gate * self.cross_attn(self.norm_q(x), self.norm_kv(context), attn_mask=attn_mask)
        x = x + self.mlp(self.norm(x))
        return x