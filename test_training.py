import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import DROIDDataset

from models import StateEncoder, TransformerWM
from models.loss import SIGReg

torch.autograd.set_detect_anomaly(True)
    

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

batch_size = 2
image_size = 224
tubelet_size = 2
tubelet_per_clip = 4
lambd = 2e-2
use_derf = False

dataset = DROIDDataset(
    num_episodes=10,
    tubelets_per_clip=tubelet_per_clip,
    image_size=image_size,
    tubelet_size=tubelet_size,
    camera_keys=["observation.images.wrist_left"],
    normalize=True,
    video_backend="pyav",
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


state_encoder = StateEncoder(
    image_size=(image_size, image_size),
    tubelet_size=tubelet_size,
    proprio_dim=8,
    emb_dim=768,
    num_queries=64,
    num_heads=12,
    depth=2,
    num_cross_blocks_per_layer=3,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
    use_silu=True,
    wide_silu=True,
    use_derf=use_derf,
)

predictor = TransformerWM(
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
    use_derf=use_derf,
    use_activation_checkpointing=False,
    state_len=64,
    max_seq_len=1024,
    action_embed_dim=8,
)

state_encoder.to(device)
predictor.to(device)

def debug_nan_hook(module, input, output):
    # Handle case where output might be a tuple
    outputs = output if isinstance(output, tuple) else (output,)
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor) and torch.isnan(out).any():
            print(f"!!! NAN detected in output of layer: {module.__class__.__name__}")
            # print(f"Layer details: {module}") 
            raise RuntimeError("Stopping due to NaN")

# Register the hook to every submodule in your models
for name, layer in state_encoder.named_modules():
    layer.register_forward_hook(debug_nan_hook)

for name, layer in predictor.named_modules():
    layer.register_forward_hook(debug_nan_hook)

# print number of parameters
se_num_params = sum(p.numel() for p in state_encoder.parameters() if p.requires_grad)
print(f"Number of trainable parameters in StateEncoder: {se_num_params/1e6:.2f}M")

pred_num_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
print(f"Number of trainable parameters in Predictor: {pred_num_params/1e6:.2f}M")

optimizer = torch.optim.AdamW(
    list(state_encoder.parameters()) + list(predictor.parameters()),
    lr=1e-4,
    weight_decay=1e-2,
)

# Initialize GradScaler for mixed precision training
scaler = torch.amp.GradScaler(device=device)

sigreg = SIGReg().to(device)

for step, batch in enumerate(loader):
    optimizer.zero_grad()
    video, proprios, actions = batch
    B, _, C, H, W = video.shape
    N = actions.size(1) # Number of prediction steps
    K = state_encoder.num_queries
    tubelet_size = state_encoder.tubelet_size
    
    print(f"Video Shape: {video.shape}")
    print(f"Proprios Shape: {proprios.shape}")
    print(f"Actions Shape: {actions.shape}")

    video = video.to(device)
    proprios = proprios.to(device)
    actions = actions.to(device)

    input_video = video[:, :-tubelet_size].contiguous() # (B, N*tubelet_size, C, H, W)
    target_video = video[:, tubelet_size:].contiguous() # (B, N*tubelet_size, C, H, W)
    input_proprios = proprios[:, :-tubelet_size].contiguous() # (B, N*tubelet_size, P)
    target_proprios = proprios[:, tubelet_size:].contiguous() # (B, N*tubelet_size, P)

    input_video = input_video.view(B * N, tubelet_size, C, H, W)  # (B*N, tubelet_size, C, H, W)
    input_proprios = input_proprios.view(B * N, tubelet_size, proprios.size(-1))  # (B*N, tubelet_size, P)
    target_video = target_video.view(B * N, tubelet_size, C, H, W)  # (B*N, tubelet_size, C, H, W)
    target_proprios = target_proprios.view(B * N, tubelet_size, proprios.size(-1))  # (B*N, tubelet_size, P)

    # Wrap forward pass in autocast context
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        input_embeds = state_encoder(input_video, input_proprios) # (B*N, K, D)
        with torch.no_grad():
            target_embeds = state_encoder(target_video, target_proprios) # (B*N, K, D)

        input_embeds = input_embeds.view(B, N*K, input_embeds.size(-1))  # (B, N*K, D)
        target_embeds = target_embeds.view(B, N*K, target_embeds.size(-1))  # (B, N*K, D)

        pred_embeds = predictor(input_embeds, actions)
        proj = input_embeds.view(B, N, K, input_embeds.size(-1)).permute(0, 2, 1, 3).reshape(B * K, N, input_embeds.size(-1))  # (B*K, N, D)

        print(f"Input Embedding Shape: {input_embeds.shape}")
        print(f"Target Embedding Shape: {target_embeds.shape}")
        print(f"Predicted States Shape: {pred_embeds.shape}")

        print(f"Pred embeds max: {pred_embeds.abs().max().item()}")
        print(f"Target embeds max: {target_embeds.abs().max().item()}")

        loss = lambd * sigreg(proj) + (1 - lambd) * F.smooth_l1_loss(pred_embeds, target_embeds)
        print(f"Loss: {loss.item():.6f}")
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    
    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(state_encoder.parameters()) + list(predictor.parameters()), 
        max_norm=1.0
    )
    
    # Step optimizer with scaler
    scaler.step(optimizer)
    scaler.update()
    
    if step >= 3:
        break