from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import Compose
import os, sys
from argparse import ArgumentParser
from typing import Union, Tuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import datasets
# from nvidia.dali import pipeline_def, fn, types

# import numpy as np
# from scipy.ndimage import gaussian_filter



# from nvidia.dali import pipeline_def
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy

from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy
import torch

@pipeline_def
def crowd_dali_pipeline(args, split="train"):
    # 1. Inputs
    encoded_images = fn.external_source(name="images", no_copy=True)
    points = fn.external_source(name="labels", no_copy=True)
    
    # 2. Get the shape ON THE CPU before decoding (Fixes the ValueError & DeprecationWarning)
    # fn.peek_image_shape returns a 1D CPU tensor: [H, W, C]
    shape = fn.peek_image_shape(encoded_images)
    h = fn.cast(fn.slice(shape, start=[0], shape=[1], axes=[0]), dtype=types.FLOAT)
    w = fn.cast(fn.slice(shape, start=[1], shape=[1], axes=[0]), dtype=types.FLOAT)

    # 3. Decode the image to GPU
    images = fn.decoders.image(
        encoded_images, 
        device="mixed", 
        output_type=types.RGB, 
        hw_decoder_load=0.9
    )

    # 4. Spatial Transform Math (All safely executing on CPU)
    if split == "train":
        crop_scale = fn.random.uniform(range=[args.min_scale, args.max_scale])
        crop_w = crop_scale * w
        crop_h = crop_scale * h

        crop_x = fn.random.uniform(range=[0.0, 1.0]) * (w - crop_w)
        crop_y = fn.random.uniform(range=[0.0, 1.0]) * (h - crop_h)
        
        flip_coin = fn.random.coin_flip(probability=0.5)
    else:
        crop_w, crop_h = w, h
        # 1. Use fn.constant to provide a stable 0.0 tensor
        # This cancels the 'offset' effect in: points = points - offset
        crop_x = fn.constant(fdata=0.0)
        crop_y = fn.constant(fdata=0.0)
        
        # 2. Use a constant 0 for the flip coin
        # This ensures (1 - mask) * points always equals 1.0 * points
        flip_coin = fn.constant(idata=0)

    # 5. Apply to Images
    start = fn.cast(fn.cat(crop_y, crop_x), dtype=types.INT32)
    crop_shape = fn.cast(fn.cat(crop_h, crop_w), dtype=types.INT32)

    images = fn.slice(images, start=start, shape=crop_shape, axes=[0, 1], 
                      out_of_bounds_policy="pad",
                      fill_values=0)
    images = fn.resize(images, size=[args.input_size, args.input_size])
    images = fn.flip(images, horizontal=flip_coin)

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )

    # 6. Apply identical math to Points
    offset = fn.cat(crop_x, crop_y, axis=0)
    points = points - offset

    scale_x = args.input_size / crop_w
    scale_y = args.input_size / crop_h
    scale = fn.cat(scale_x, scale_y, axis=0)
    points = points * scale

    x = fn.slice(points, start=[0], shape=[1], axes=[1])
    y = fn.slice(points, start=[1], shape=[1], axes=[1])
    
    x_flip = (args.input_size - 1.0) - x
    flipped_points = fn.cat(x_flip, y, axis=1)

    mask = fn.cast(flip_coin, dtype=types.FLOAT)
    transformed_points = mask * flipped_points + (1.0 - mask) * points

    #FIX: Pad ragged arrays to a uniform dense shape for the batch
    padded_points = fn.pad(transformed_points, fill_value=-1.0)
    
    return images, padded_points


def generate_density_map_gpu(label: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
    # Ensure standard PyTorch formatting
    label_tensor = torch.as_tensor(label, device=device)
    density_map = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    
    if len(label_tensor) > 0:
        label_ = label_tensor.long()
        label_[:, 0] = label_[:, 0].clamp(min=0, max=width - 1)
        label_[:, 1] = label_[:, 1].clamp(min=0, max=height - 1)
        density_map[0, label_[:, 1], label_[:, 0]] = 1.0
        
    return density_map


class ProxyOutputWrapper:
    """Safely converts the 2-output proxy batch into the required 3-output PyTorch batch."""
    def __init__(self, loader, input_size):
        self.loader = loader
        self.input_size = input_size

    def __iter__(self):
        # The proxy native DataLoader guarantees these are PyTorch GPU Tensors
        for images, points_batch in self.loader:
            device = images.device
            
            densities = []
            points_list = []

            # CRITICAL FIX: Extract dynamic H and W from the evaluation images 
            img_h, img_w = images.shape[-2], images.shape[-1]
            #----------------------------------------------------------------

            # Unbind if DALI returned a perfectly stacked tensor, otherwise iterate the list
            if isinstance(points_batch, torch.Tensor):
                points_iterable = list(torch.unbind(points_batch, dim=0))
            else:
                points_iterable = points_batch

            # for pts in points_iterable:
            #     pts_tensor = torch.as_tensor(pts, device=device)
            #     points_list.append(pts_tensor)
            #     densities.append(generate_density_map_gpu(pts_tensor, self.input_size, self.input_size, device))
            for pts in points_iterable:
                pts_tensor = torch.as_tensor(pts, device=device)
                
                valid_mask = pts_tensor[:, 0] != -1.0
                valid_pts = pts_tensor[valid_mask]
                points_list.append(valid_pts)
                
                # Pass dynamic H and W so density maps scale flawlessly to the sliding window
                densities.append(generate_density_map_gpu(valid_pts, img_h, img_w, device))
            
            densities = torch.stack(densities, dim=0)
            
            yield images, points_list, densities

    def __len__(self):
        return len(self.loader)
def get_dataloader_dali(args, split="train", ddp=False):
    if split == "train":
        pipe = crowd_dali_pipeline(
            args=args,
            split=split,
            batch_size=args.batch_size,
            num_threads=3,
            device_id=0,
            prefetch_queue_depth=2 * args.num_workers,
        )
    else:
        pipe = crowd_dali_pipeline(
            args=args,
            split=split,
            batch_size=1,
            num_threads=3,
            device_id=0,
            prefetch_queue_depth=2 * args.num_workers,
        )

    dali_server = dali_proxy.DALIServer(pipe)

    # 1. Pass the proxy directly to the Dataset
    dataset = datasets.CrowdDali(
        dataset=args.dataset,
        split=split,
        input_size=args.input_size,
        return_filename=False,
        transform=dali_server.proxy,
    )

    # 2. No custom collate functions. The proxy natively hooks default_collate.
    loader = dali_proxy.DataLoader(
        dali_server,
        dataset,
        batch_size=args.batch_size if split == "train" else 1,
        num_workers=args.num_workers,
        drop_last=(split == "train"),
    )

    return ProxyOutputWrapper(loader, args.input_size), None

def get_dataloader(args: ArgumentParser, split: str = "train", ddp: bool = False) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    if split == "train":  # train, strong augmentation
        transforms = Compose([
            datasets.RandomResizedCrop((args.input_size, args.input_size), scale=(args.min_scale, args.max_scale)),
            datasets.RandomHorizontalFlip(),
            datasets.RandomApply([
                datasets.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
                datasets.GaussianBlur(kernel_size=args.kernel_size, sigma=(0.1, 5.0)),
                datasets.PepperSaltNoise(saltiness=args.saltiness, spiciness=args.spiciness),
            ], p=(args.jitter_prob, args.blur_prob, args.noise_prob)),
        ])

    elif args.sliding_window:
        if args.resize_to_multiple:
            transforms = datasets.Resize2Multiple(args.window_size, stride=args.stride)
        elif args.zero_pad_to_multiple:
            transforms = datasets.ZeroPad2Multiple(args.window_size, stride=args.stride)
        else:
            transforms = None

    else:
        transforms = None

    dataset = datasets.Crowd(
        dataset=args.dataset,
        split=split,
        transforms=transforms,
        sigma=None,     #sigma always none --> no qaussian blur on density map
        return_filename=False,
        num_crops=args.num_crops if split == "train" else 1,
    )

    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader, sampler

    elif split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader, None

    else:  # data_loader for evaluation
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader
    
