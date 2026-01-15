import math
import torch

import torch.nn.functional as F
import pytorch_lightning as pl

from models import StateEncoder, TransformerWM
from models.loss import SIGReg


class LightningWM(pl.LightningModule):
    def __init__(
        self,
        # State Encoder params
        image_size: tuple[int, int] = (384, 384),
        tubelet_size: int = 2,
        image_backbone: str = "resnet50",
        pretrained: bool = False,
        proprio_dim: int = 8,
        emb_dim: int = 768,
        num_queries: int = 64,
        num_heads_encoder: int = 12,
        encoder_depth: int = 4,
        num_cross_blocks_per_layer: int = 3,
        # Predictor params
        predictor_embed_dim: int = 1024,
        predictor_depth: int = 24,
        num_heads_predictor: int = 16,
        max_seq_len: int = 1024,
        action_dim: int = 8,
        # Training params
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_silu: bool = True,
        wide_silu: bool = True,
        use_derf: bool = False,
        init_std: float = 0.02,
        use_activation_checkpointing: bool = False,
        # SIGReg params
        sigreg_knots: int = 17,
        # Optimizer params
        learning_rate: float = 1e-4,
        lambd: float = 2e-2,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize state encoder
        self.state_encoder = StateEncoder(
            image_size=image_size,
            tubelet_size=tubelet_size,
            image_backbone=image_backbone,
            pretrained=pretrained,
            proprio_dim=proprio_dim,
            emb_dim=emb_dim,
            num_queries=num_queries,
            num_heads=num_heads_encoder,
            depth=encoder_depth,
            num_cross_blocks_per_layer=num_cross_blocks_per_layer,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            use_silu=use_silu,
            wide_silu=wide_silu,
            use_derf=use_derf,
            init_std=init_std,
        )
        
        # Initialize predictor
        self.predictor = TransformerWM(
            embed_dim=emb_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=num_heads_predictor,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            init_std=init_std,
            use_silu=use_silu,
            wide_silu=wide_silu,
            use_derf=use_derf,
            use_activation_checkpointing=use_activation_checkpointing,
            state_len=num_queries,
            max_seq_len=max_seq_len,
            action_dim=action_dim,
        )
        
        # SIGReg loss
        self.sigreg = SIGReg(knots=sigreg_knots)
        self.lambd = lambd
        
        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    
    def forward(self, video: torch.Tensor, proprios: torch.Tensor, actions: torch.Tensor):
        """
        Forward pass for autoregressive state prediction.
        
        Args:
            video: (B, T, C, H, W) video tensor
            proprios: (B, T, proprio_dim) proprioception tensor
            actions: (B, N, A) action tensor
            
        Returns:
            Dictionary with predictions and targets
        """
        # Encode all states with online encoder
        B, _, C, H, W = video.shape
        N = actions.size(1) # Number of prediction steps
        K = self.state_encoder.num_queries
        tubelet_size = self.state_encoder.tubelet_size
        
        input_video = video[:, :-tubelet_size].contiguous() # (B, N*tubelet_size, C, H, W)
        target_video = video[:, tubelet_size:].contiguous() # (B, N*tubelet_size, C, H, W)
        input_proprios = proprios[:, :-tubelet_size].contiguous() # (B, N*tubelet_size, P)
        target_proprios = proprios[:, tubelet_size:].contiguous() # (B, N*tubelet_size, P)

        input_video = input_video.view(B * N, tubelet_size, C, H, W)  # (B*N, tubelet_size, C, H, W)
        input_proprios = input_proprios.view(B * N, tubelet_size, proprios.size(-1))  # (B*N, tubelet_size, P)
        target_video = target_video.view(B * N, tubelet_size, C, H, W)  # (B*N, tubelet_size, C, H, W)
        target_proprios = target_proprios.view(B * N, tubelet_size, proprios.size(-1))  # (B*N, tubelet_size, P)

        # Encode input
        input_embeds = self.state_encoder(input_video, input_proprios) # (B*N, K, D)
        # Encode target
        with torch.no_grad():
            target_embeds = self.state_encoder(target_video, target_proprios) # (B*N, K, D)
        
        # Predict next states
        input_embeds = input_embeds.view(B, N * K, input_embeds.size(-1))  # (B, N*K, D)
        target_embeds = target_embeds.view(B, N * K, target_embeds.size(-1))  # (B, N*K, D)

        pred_embeds = self.predictor(input_embeds, actions)  # (B, N*K, D)
        proj = input_embeds.view(B, N, K, input_embeds.size(-1)).permute(0, 2, 1, 3).reshape(B * K, N, input_embeds.size(-1))
        
        return pred_embeds, target_embeds, proj
    
    def compute_loss(
        self,
        pred_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        proj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total loss with reconstruction and SIGReg regularization.
        
        Args:
            pred_embeds: (B, N*K, D) predicted embeddings
            target_embeds: (B, N*K, D) target embeddings
            proj: (B*K, N, D) proj for SIGReg
        Returns:
            Dictionary with total loss, reconstruction loss, and SIGReg loss
        """
        # Reconstruction loss (SmoothL1)
        recon_loss = F.smooth_l1_loss(pred_embeds, target_embeds)
        # SIGReg loss
        #sigreg_loss = self.sigreg(proj)
        # Total loss
        #total_loss = self.lambd * sigreg_loss + (1 - self.lambd) * recon_loss
        total_loss = recon_loss
        sigreg_loss = torch.tensor(0.0, device=total_loss.device)
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'sigreg_loss': sigreg_loss,
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        video, proprios, actions = batch
        
        # Forward pass
        pred_embeds, target_embeds, proj = self(video, proprios, actions)
        
        # Compute losses
        losses = self.compute_loss(pred_embeds, target_embeds, proj)
        
        # Log metrics
        self.log('train/loss', losses['loss'], prog_bar=True)
        self.log('train/recon_loss', losses['recon_loss'])
        self.log('train/sigreg_loss', losses['sigreg_loss'])
        self.log('train/lr', self.optimizers().param_groups[0]['lr'])
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        video, proprios, actions = batch
        
        # Forward pass
        pred_embeds, target_embeds, proj = self(video, proprios, actions)
        
        # Compute losses
        losses = self.compute_loss(pred_embeds, target_embeds, proj)
        
        # Log metrics
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', losses['recon_loss'], sync_dist=True)
        self.log('val/sigreg_loss', losses['sigreg_loss'], sync_dist=True)
        
        return losses['loss']
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and norms
            if 'bias' in name or 'norm' in name or 'erf' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.learning_rate, betas=(0.9, 0.95))
        
        # Cosine learning rate schedule with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }