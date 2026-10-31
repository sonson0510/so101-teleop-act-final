#!/usr/bin/env python3
"""Train ACT only on a validated, isolated SO-101 teleop dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def delta(indices, fps):
    return [index / fps for index in indices] if indices is not None else [0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--pretrained-backbone-weights",
                        default="ResNet18_Weights.IMAGENET1K_V1",
                        help="torchvision weights enum, or 'none' for random initialization")
    parser.add_argument("--no-vae", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--init-policy", type=Path,
                        help="Initialize from a saved ACT policy; optimizer starts fresh")
    parser.add_argument("--allow-diagnostic-synthetic", action="store_true",
                        help="One-step pipeline smoke tests only; output path must contain diagnostic")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    manifest_path = root / "teleop_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing teleop conversion manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") not in ("train", "diagnostic"):
        raise ValueError("ACT training requires a train split")
    if manifest.get("diagnostic_synthetic"):
        if not args.allow_diagnostic_synthetic or "diagnostic" not in str(output).lower() or args.steps != 1:
            raise ValueError("Synthetic dataset is restricted to one-step diagnostic output")
    if (args.batch_size < 1 or args.steps < 1 or args.num_workers < 0 or
            args.learning_rate <= 0 or args.backbone_learning_rate <= 0):
        raise ValueError("Batch size/steps must be positive and num-workers nonnegative")
    output.mkdir(parents=True, exist_ok=False)

    metadata = LeRobotDatasetMetadata(args.repo_id, root=root)
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    input_features = {
        key: value for key, value in features.items()
        if value.type in (FeatureType.VISUAL, FeatureType.STATE)
    }
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=(None if args.pretrained_backbone_weights.lower() == "none"
                                     else args.pretrained_backbone_weights),
        dim_model=256,
        n_heads=8,
        dim_feedforward=1024,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=not args.no_vae,
        latent_dim=32,
        n_vae_encoder_layers=4,
        optimizer_lr=args.learning_rate,
        optimizer_lr_backbone=args.backbone_learning_rate,
        device=str(device),
        push_to_hub=False,
    )
    delta_timestamps = {
        "action": delta(config.action_delta_indices, metadata.fps),
        **{
            key: delta(config.observation_delta_indices, metadata.fps)
            for key in config.image_features
        },
    }
    dataset = LeRobotDataset(args.repo_id, root=root, delta_timestamps=delta_timestamps)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if args.init_policy is not None:
        policy = ACTPolicy.from_pretrained(args.init_policy.resolve()).to(device).train()
        if (policy.config.input_features != config.input_features or
                policy.config.output_features != config.output_features or
                policy.config.chunk_size != config.chunk_size):
            raise ValueError("Initial policy is incompatible with this dataset/configuration")
    else:
        policy = ACTPolicy(config).to(device).train()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=metadata.stats)
    # Preserve generic ACT learning rate while fine-tuning the pretrained visual
    # backbone more conservatively through ACTPolicy's parameter groups.
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())

    history = []
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        loss, loss_info = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint = output / "checkpoints" / f"step_{step:06d}"
            policy.save_pretrained(checkpoint / "policy")
            preprocessor.save_pretrained(checkpoint)
            postprocessor.save_pretrained(checkpoint)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.item()), **loss_info}
            history.append(record)
            print(json.dumps(record), flush=True)

    policy.save_pretrained(output / "policy")
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    with (output / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "repo_id": args.repo_id,
        "dataset_root": str(root),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "use_vae": not args.no_vae,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "pretrained_backbone_weights": config.pretrained_backbone_weights,
        "init_policy": str(args.init_policy.resolve()) if args.init_policy else None,
        "final_loss": history[-1]["loss"],
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
