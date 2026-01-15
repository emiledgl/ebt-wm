import torch
import torch.nn as nn
import math

from models.derf import Dynamic_erf
from models.modules import AttentionBlock, CrossAttentionBlock
from models.modules import precompute_freqs
from timm.layers import trunc_normal_
from timm import create_model


class StateEncoder(nn.Module):
    """
    State Encoder that aggregates image and proprioception embeddings into query latents.
    """
    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        tubelet_size: int = 2,
        image_backbone: str = "resnet50", # supported by timm
        pretrained: bool = False,
        proprio_dim: int = 8,
        emb_dim: int = 768,
        num_queries: int = 64,
        num_heads: int = 12,
        depth: int = 2,
        num_cross_blocks_per_layer: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_silu: bool = True,
        wide_silu: bool = True,
        use_derf: bool = False,
        init_std: float = 0.02,
        
    ):
        super().__init__()
        
        self.image_size = image_size
        self.tubelet_size = tubelet_size
        self.proprio_dim = proprio_dim
        self.emb_dim = emb_dim
        self.num_queries = num_queries

        self.image_encoder = create_model(
            model_name=image_backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],  # Get only the final feature map
        )
        self.image_emb_dim, self.grid_size = self._get_encoder_info(image_size)


        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_queries, emb_dim))
        
        # Project inputs to common embedding dimension
        self.image_proj = nn.Linear(self.image_emb_dim, emb_dim)
        self.proprio_proj = nn.Linear(proprio_dim, emb_dim)

        norm_layer = Dynamic_erf if use_derf else nn.RMSNorm
        
        # Attention pooling layers
        self.pooling_layers = nn.ModuleList([
            StatePoolerLayer(
                emb_dim,
                num_heads,
                num_queries=num_queries,
                num_cross_blocks=num_cross_blocks_per_layer,
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
        ])

        self.final_norm = norm_layer(emb_dim)

        # Positional embeddings
        self.register_buffer('patch_pos_embed', self._create_3d_fourier_pos(emb_dim)) # (T*H*W, emb_dim)
        self.register_buffer('proprio_pos_embed', self._create_1d_temporal_pos(emb_dim)) # (T, emb_dim)
        
        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()


    def _get_encoder_info(self, input_size):
        dummy_input = torch.zeros(1, 3, *input_size)
        
        with torch.no_grad():
            output = self.image_encoder(dummy_input)[0]
        
        # Extract shapes
        channels = output.shape[1]
        grid_h, grid_w = output.shape[2], output.shape[3]
        
        return channels, (self.tubelet_size, grid_h, grid_w)
    
    
    def _create_3d_fourier_pos(self, dim: int) -> torch.Tensor:
        """Create 3D Fourier positional embeddings."""

        t, h, w = self.grid_size
        assert dim % 6 == 0, "Embedding dimension must be divisible by 6 for 3D Fourier features."
        num_freqs = dim // 6  # since we have sin and cos for each of x, y, t

        # Create normalized 3D grid from -1 to 1
        t_grid, y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, t), 
            torch.linspace(-1, 1, h), 
            torch.linspace(-1, 1, w), 
            indexing='ij'
        )
        stacked = torch.stack([t_grid, y_grid, x_grid], dim=-1).view(-1, 3)  # (T*H*W, 3)
        
        # Generate frequencies
        freqs = torch.exp(
            torch.arange(0, num_freqs).float() * -(math.log(10000.0) / num_freqs)
        )
        
        # Apply frequencies to positions
        args = stacked.unsqueeze(-1) * freqs  # (T*H*W, 3, dim/6)
        
        # Apply sin and cos
        pos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (T*H*W, 3, dim/3)
        pos = pos.view(t * h * w, dim)  # (T*H*W, dim)
        
        return pos
    

    def _create_1d_temporal_pos(self, dim: int) -> torch.Tensor:
        """Create 1D temporal positional embeddings for proprioception."""
        t = self.tubelet_size
        assert dim % 2 == 0, "Embedding dimension must be divisible by 2 for 1D Fourier features."
        num_freqs = dim // 2
        
        # Create normalized temporal positions from -1 to 1
        positions = torch.linspace(-1, 1, t).unsqueeze(-1)  # (T, 1)
        
        # Generate frequencies
        freqs = torch.exp(
            torch.arange(0, num_freqs).float() * -(math.log(10000.0) / num_freqs)
        )
        
        # Apply frequencies
        args = positions * freqs  # (T, num_freqs)
        
        # Apply sin and cos
        pos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (T, dim)
        
        return pos

    
    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        g_block_id = 0 # global block counter across layers
        for layer in self.pooling_layers:
            for block in layer.cross_blocks:
                rescale(block.cross_attn.proj.weight.data, g_block_id + 1)
                rescale(block.mlp.fc2.weight.data, g_block_id + 1)
                g_block_id += 1
            
            rescale(layer.attn_block.attn.proj.weight.data, g_block_id + 1)
            rescale(layer.attn_block.mlp.fc2.weight.data, g_block_id + 1)
            g_block_id += 1
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        images: torch.Tensor,
        proprios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Observation images of shape (B, tubelet_size, 3, H, W)
            proprios: Proprioception embeddings of shape (B, tubelet_size, proprio_emb_dim)
            
        Returns:
            Query latents of shape (B, K, emb_dim)
        """
        B, tubelet_size, _, H, W = images.shape
        # Encode images
        images = images.view(B * tubelet_size, 3, H, W)  # (B*tubelet_size, 3, H, W)
        img_emb = self.image_encoder(images)[0]  # (B*tubelet_size, C, H', W')
        C, H_enc, W_enc = img_emb.shape[1:]
        assert (tubelet_size, H_enc, W_enc) == self.grid_size, "Encoded image size does not match expected grid size."

        img_emb = img_emb.view(B, tubelet_size, C, H_enc, W_enc).permute(0, 1, 3, 4, 2)  # (B, tubelet_size, H', W', C)
        img_emb = img_emb.reshape(B, tubelet_size * H_enc * W_enc, C)  # (B, tubelet_size*H'*W', C)
        
        # Project inputs to common dimension
        x = self.image_proj(img_emb)  # (B, tubelet_size*H'*W', emb_dim)
        p = self.proprio_proj(proprios)  # (B, tubelet_size, emb_dim)
        
        # Add positional embeddings
        x = x + self.patch_pos_embed.unsqueeze(0)
        p = p + self.proprio_pos_embed.unsqueeze(0)
        
        # Concatenate observation and proprioception tokens
        x = x.view(B, tubelet_size, H_enc * W_enc, -1)  # (B, tubelet_size, H'*W', emb_dim)
        p = p.unsqueeze(2)  # (B, tubelet_size, 1, emb_dim)
        c = torch.cat([x, p], dim=2).flatten(1, 2)  # (B, tubelet_size*(H'*W'+1), emb_dim)
        
        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, K, emb_dim)
        
        # Apply attention pooling layers
        for layer in self.pooling_layers:
            queries = layer(queries, c)
        
        queries = self.final_norm(queries)
        return queries


class StatePoolerLayer(nn.Module):
    def __init__(
        self,
        emb_dim, 
        num_heads, 
        num_queries=64,
        num_cross_blocks=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0, 
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        wide_silu=True,
        norm_layer=Dynamic_erf,
        use_self_attn=True,
    ):
        super().__init__()

        self.use_self_attn = use_self_attn
    
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                emb_dim, 
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                wide_silu=wide_silu,
                norm_layer=norm_layer,
            ) for _ in range(num_cross_blocks)
        ])

        if self.use_self_attn:
            self.attn_block = AttentionBlock(
                emb_dim, 
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                wide_silu=wide_silu,
                norm_layer=norm_layer,
            )

            coords = torch.arange(num_queries)
            freqs = precompute_freqs(emb_dim // num_heads, coords)
            self.register_buffer('freqs', freqs) # (2, num_queries, head_dim)

        else:
            self.attn_block = nn.Identity()


    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for cross_block in self.cross_blocks:
            latent = cross_block(latent, context)
        if self.use_self_attn:
            latent = self.attn_block(latent, self.freqs)
        return latent