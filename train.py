import argparse
from pathlib import Path

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import DROIDDataset
from models import LightningWM


def parse_args():
    parser = argparse.ArgumentParser(description="Train World Model with SIGReg")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing DROID dataset")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to use")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--tubelets_per_clip", type=int, default=8, help="Number of tubelets per clip (context length)")
    parser.add_argument("--camera_keys", type=str, nargs="+", default=["observation.images.wrist_left"], help="Camera keys to use")
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize inputs")
    parser.add_argument("--video_backend", type=str, default="torchcodec", choices=["pyav", "torchcodec"], help="Video backend")
    
    # Model parameters - State Encoder
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument("--tubelet_size", type=int, default=2, help="Tubelet size")
    parser.add_argument("--image_backbone", type=str, default="resnet50", help="Image backbone model")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained image backbone")
    parser.add_argument("--proprio_dim", type=int, default=8, help="Proprioception dimension")
    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_queries", type=int, default=64, help="Number of query tokens")
    parser.add_argument("--num_heads_encoder", type=int, default=12, help="Number of attention heads in encoder")
    parser.add_argument("--encoder_depth", type=int, default=2, help="Encoder depth")
    parser.add_argument("--num_cross_blocks_per_layer", type=int, default=3, help="Number of cross attention blocks per layer")
    
    # Model parameters - Predictor
    parser.add_argument("--predictor_embed_dim", type=int, default=1024, help="Predictor embedding dimension")
    parser.add_argument("--predictor_depth", type=int, default=24, help="Predictor depth")
    parser.add_argument("--num_heads_predictor", type=int, default=16, help="Number of attention heads in predictor")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--action_dim", type=int, default=8, help="Action dimension")
    
    # Training parameters
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP ratio")
    parser.add_argument("--qkv_bias", action="store_true", default=True, help="Use QKV bias")
    parser.add_argument("--drop", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument("--drop_path", type=float, default=0.0, help="Drop path rate")
    parser.add_argument("--use_silu", action="store_true", default=True, help="Use SiLU activation")
    parser.add_argument("--wide_silu", action="store_true", default=True, help="Use wide SiLU")
    parser.add_argument("--use_derf", action="store_true", default=False, help="Use Dynamic erf activation")
    parser.add_argument("--init_std", type=float, default=0.02, help="Weight initialization std")
    parser.add_argument("--use_activation_checkpointing", action="store_true", help="Use activation checkpointing to save memory")

    # Optimizer parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambd", type=float, default=2e-2, help="Weight for SIGReg loss component")
    parser.add_argument("--sigreg_knots", type=int, default=17, help="Number of knots for SIGReg")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="32-true", choices=["32-true", "16-mixed", "bf16-mixed"], help="Training precision")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    
    # Checkpoint and logging
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--tensorboard", action="store_true", default=False, help="Use TensorBoard for logging")
    parser.add_argument("--exp_name", type=str, default="transformer_wm", help="Experiment name")
    parser.add_argument("--wandb_project", type=str, default="ebt_wm", help="W&B project name")
    parser.add_argument("--save_top_k", type=int, default=1, help="Save top k checkpoints")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validation frequency in epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Logging frequency in steps")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Hardware
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "cpu", "mps"], help="Accelerator type")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "fsdp"], help="Training strategy")
    
    # Debugging
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast dev run for debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def create_datasets(args):
    """Create train and validation datasets."""
    dataset = DROIDDataset(
        num_episodes=args.num_episodes,
        tubelets_per_clip=args.tubelets_per_clip,
        image_size=args.image_size,
        tubelet_size=args.tubelet_size,
        camera_keys=args.camera_keys,
        normalize=args.normalize,
        video_backend=args.video_backend,
    )
    
    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, args):
    """Create train and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    return train_loader, val_loader



def main():
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(args)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args)
    
    # Create model
    model = LightningWM(
        # State Encoder params
        image_size=(args.image_size, args.image_size),
        tubelet_size=args.tubelet_size,
        num_views=len(args.camera_keys),
        image_backbone=args.image_backbone,
        pretrained=args.pretrained,
        proprio_dim=args.proprio_dim,
        emb_dim=args.emb_dim,
        num_queries=args.num_queries,
        num_heads_encoder=args.num_heads_encoder,
        encoder_depth=args.encoder_depth,
        num_cross_blocks_per_layer=args.num_cross_blocks_per_layer,
        # Predictor params
        predictor_embed_dim=args.predictor_embed_dim,
        predictor_depth=args.predictor_depth,
        num_heads_predictor=args.num_heads_predictor,
        max_seq_len=args.max_seq_len,
        action_dim=args.action_dim,
        # Training params
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        drop=args.drop,
        attn_drop=args.attn_drop,
        drop_path=args.drop_path,
        use_silu=args.use_silu,
        wide_silu=args.wide_silu,
        use_derf=args.use_derf,
        init_std=args.init_std,
        use_activation_checkpointing=args.use_activation_checkpointing,
        # Loss weights
        lambd=args.lambd,
        # SIGReg params
        sigreg_knots=args.sigreg_knots,
        # Optimizer params
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    
    # Create callbacks and logger
    log_dir = Path(args.output_dir) / args.exp_name / "logs"
    if args.tensorboard:
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name=args.exp_name,
        )
    else:
        wandb.login()
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.exp_name,
            save_dir=log_dir,
            log_model=False,
        )

    # Model checkpoint
    checkpoint_dir = Path(args.output_dir) / args.exp_name / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        fast_dev_run=args.fast_dev_run,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()