import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

from utils import barrier, reduce_mean, update_loss_info


def quantize_to_bins(target_density: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """
    Helper function to map continuous density values to the closest discrete bin indices.
    Required for the CMAR loss to establish rank comparisons.
    """
    # Reshape target_density to [Batch*Patches, 1] and anchors to [1, Num_Bins]
    flat_density = target_density.reshape(-1, 1)
    flat_anchors = anchor_points.reshape(1, -1).to(target_density.device)

    # Find the index of the closest anchor point (L1 distance)
    distances = torch.abs(flat_density - flat_anchors)
    bin_indices = torch.argmin(distances, dim=1)

    return bin_indices.view(target_density.shape)


def train(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        grad_scaler: GradScaler,
        device: torch.device,
        rank: int,
        nprocs: int,
        cmar_criterion: nn.Module = None,  # NEW: CMAR Loss
        lambda_cmar: float = 0.1  # NEW: CMAR Weight
) -> Tuple[nn.Module, Optimizer, GradScaler, Dict[str, float]]:
    model.train()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    # Extract configurations from the model (handle DDP wrapper if present)
    base_model = model.module if ddp else model
    regression = base_model.bins is None
    anchor_points = base_model.anchor_points if hasattr(base_model, 'anchor_points') else None

    for image, target_points, target_density in data_iter:
        image = image.to(device)
        target_points = [p.to(device) for p in target_points]
        target_density = target_density.to(device)

        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    if not regression:
                        # NEW: Unpack the features from the modified model wrapper
                        pred_class, pred_density, img_feats, txt_feats = model(image)

                        # FOR DEBUGGING: Print the absolute sum of differences between the first two text embeddings to verify they are not identical (which would be a red flag)
                        print("First 5 values of Text 0:", txt_feats[0][:5])
                        print("Text Embeddings diff:", (txt_feats[0] - txt_feats[1]).abs().sum())

                        loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
                    else:
                        pred_density = model(image)
                        loss, loss_info = loss_fn(pred_density, target_density, target_points)
                        img_feats, txt_feats = None, None

                    # NEW: Calculate and add CMAR loss
                    if cmar_criterion is not None and not regression and anchor_points is not None:
                        # 1. Get the spatial dimensions of the feature map (H=56, W=56)
                        B, H, W, C = img_feats.shape

                        # 2. Calculate the reduction factor between the target map and feature map
                        pool_h = target_density.shape[-2] // H
                        pool_w = target_density.shape[-1] // W

                        # 3. Shrink the target density using average pooling.
                        # CRITICAL: Multiply by the area (pool_h * pool_w) to preserve the total crowd count!
                        target_density_shrunk = torch.nn.functional.avg_pool2d(
                            target_density,
                            kernel_size=(pool_h, pool_w),
                            stride=(pool_h, pool_w)
                        ) * (pool_h * pool_w)

                        # 4. Grab the anchor points (handling DDP module wrapper if necessary)
                        anchor_points = model.module.anchor_points if hasattr(model, 'module') else model.anchor_points

                        # Push anchor_points to the GPU!
                        anchor_points = anchor_points.to(target_density_shrunk.device)

                        # 5. Quantize the shrunk density map to get the correct gt_bins
                        diff = torch.abs(target_density_shrunk - anchor_points)
                        gt_bins = torch.argmin(diff, dim=1)  # Shape becomes [B, H, W]

                        # FOR DEBUGGING: Print unique bin indices to verify they are within expected range
                        print("Unique Ground Truth Bins:", gt_bins.unique())

                        # 6. Calculate CMAR loss
                        cmar_loss = cmar_criterion(img_feats, txt_feats, gt_bins)

                        loss = loss + (lambda_cmar * cmar_loss)
                        loss_info["cmar_loss"] = cmar_loss  # Track in logs

            else:
                if not regression:
                    # NEW: Unpack the features
                    pred_class, pred_density, img_feats, txt_feats = model(image)
                    loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
                else:
                    pred_density = model(image)
                    loss, loss_info = loss_fn(pred_density, target_density, target_points)
                    img_feats, txt_feats = None, None

                # NEW: Calculate and add CMAR loss (No AMP)
                if cmar_criterion is not None and not regression and anchor_points is not None:
                    gt_bins = quantize_to_bins(target_density, anchor_points)
                    cmar_loss = cmar_criterion(img_feats, txt_feats, gt_bins)

                    loss = loss + (lambda_cmar * cmar_loss)
                    loss_info["cmar_loss"] = cmar_loss  # Track in logs

        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()

            # PREVENT NAN LOSSES: Unscale the gradients and clip them so they can't explode
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in
                     loss_info.items()}
        info = update_loss_info(info, loss_info)

        barrier(ddp)

    return model, optimizer, grad_scaler, {k: np.mean(v) for k, v in info.items()}