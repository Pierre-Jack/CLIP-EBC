from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import Compose
import os, sys
from argparse import ArgumentParser
from typing import Union, Tuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import datasets
from nvidia.dali import pipeline_def, fn, types

import numpy as np
from scipy.ndimage import gaussian_filter


# def transform_points(points, orig_w, orig_h, new_size, flip):

#     if len(points) == 0:
#         return points.astype(np.float32)

#     pts = points.copy()

#     # resize scale
#     sx = new_size / orig_w
#     sy = new_size / orig_h

#     pts[:,0] *= sx
#     pts[:,1] *= sy

#     if flip:
#         pts[:,0] = new_size - pts[:,0]

#     return pts.astype(np.float32)


# def points_to_dotmap(points, H, W):

#     density = np.zeros((1, H, W), dtype=np.float32)

#     if len(points) > 0:
#         pts = points.astype(np.int64)

#         pts[:,0] = np.clip(pts[:,0], 0, W-1)
#         pts[:,1] = np.clip(pts[:,1], 0, H-1)

#         density[0, pts[:,1], pts[:,0]] = 1.0

#     return density


# def blur_density(dotmap, sigma):

#     return gaussian_filter(dotmap, sigma=sigma)

# @pipeline_def
# def crowd_dali_pipeline(args, split="train"):
#     # Receives the file path or raw bytes from the Dataset
#     encoded_images = fn.external_source(name="images", no_copy=True)
    
#     if split == "train":
#         # Decode and RandomResizedCrop in one shot on GPU
#         images = fn.decoders.image_random_crop(
#             encoded_images,
#             device="mixed",
#             output_type=types.RGB,
#             random_area=[args.min_scale, args.max_scale],
#             # DALI handles aspect ratio slightly differently; usually [0.75, 1.33]
#         )
#         images = fn.resize(images, size=[args.input_size, args.input_size])
#         images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
        
#         # Color Jitter
#         images = fn.color_twist(
#             images,
#             brightness=fn.random.uniform(range=[1-args.brightness, 1+args.brightness]),
#             contrast=fn.random.uniform(range=[1-args.contrast, 1+args.contrast]),
#             saturation=fn.random.uniform(range=[1-args.saturation, 1+args.saturation]),
#             hue=fn.random.uniform(range=[-args.hue, args.hue])
#         )
        
#         # Gaussian Blur
#         # 1. Define a random distribution for sigma
#         # This creates a different value for every image in the batch
#         sigma_val = fn.random.uniform(range=[0.1, 5.0])

#         # 2. Apply it to the images
#         # window_size is usually the 'kernel_size' from your args
#         images = fn.gaussian_blur(
#             images, 
#             window_size=args.kernel_size, 
#             sigma=sigma_val
# )
        
#     else:
#         # Evaluation mode: Just decode and resize
#         images = fn.decoders.image(encoded_images, device="mixed", output_type=types.RGB)
#         if args.sliding_window:
#             # You'd implement Resize2Multiple logic here or via fn.resize
#             pass

#     # Final Normalization (CLIP / ImageNet)
#     output = fn.crop_mirror_normalize(
#         images,
#         dtype=types.FLOAT,
#         output_layout="CHW",
#         mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#         std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
#     )
#     return output



# @pipeline_def
# def crowd_dali_pipeline(args, split="train"):

#     # -------------------------------------------------
#     # INPUTS
#     # -------------------------------------------------

#     encoded_images = fn.external_source(name="images", no_copy=True)
#     points = fn.external_source(name="labels")   # Nx2 float


#     # -------------------------------------------------
#     # DECODE
#     # -------------------------------------------------

#     images = fn.decoders.image(
#         encoded_images,
#         device="mixed",
#         output_type=types.RGB
#     )


#     # -------------------------------------------------
#     # IMAGE SIZE
#     # -------------------------------------------------

#     shape = fn.shapes(images)

#     height = fn.slice(shape, 0, 1, axes=[0])
#     width  = fn.slice(shape, 1, 1, axes=[0])


#     # -------------------------------------------------
#     # RANDOM CROP PARAMETERS
#     # -------------------------------------------------

#     if split == "train":

#         crop_scale = fn.random.uniform(
#             range=[args.min_scale, args.max_scale]
#         )

#         crop_w = crop_scale * width
#         crop_h = crop_scale * height

#         crop_x = fn.random.uniform(range=[0.0, 1.0]) * (width - crop_w)
#         crop_y = fn.random.uniform(range=[0.0, 1.0]) * (height - crop_h)

#         images = fn.slice(
#             images,
#             start=fn.cat(crop_y, crop_x),
#             shape=fn.cat(crop_h, crop_w),
#             axes=[0, 1]
#         )

#         flip_coin = fn.random.coin_flip(probability=0.5)

#     else:

#         crop_x = 0
#         crop_y = 0
#         crop_w = width
#         crop_h = height
#         flip_coin = 0


#     # -------------------------------------------------
#     # RESIZE + FLIP
#     # -------------------------------------------------

#     images = fn.resize(images, size=[args.input_size, args.input_size])

#     images = fn.flip(images, horizontal=flip_coin)


#     # -------------------------------------------------
#     # LABEL TRANSFORM
#     # -------------------------------------------------

#     scale_x = args.input_size / crop_w
#     scale_y = args.input_size / crop_h

#     offset = fn.cat(crop_x, crop_y)

#     points = points - offset

#     scale = fn.cat(scale_x, scale_y)

#     points = points * scale


#     # split coordinates
#     x = fn.slice(points, 0, 1, axes=[1])
#     y = fn.slice(points, 1, 1, axes=[1])


#     # flip transform
#     x_flip = args.input_size - 1 - x

#     flipped = fn.cat(x_flip, y, axis=1)

#     mask = fn.cast(flip_coin, dtype=types.FLOAT)

#     points = mask * flipped + (1 - mask) * points

#     transformed_points = points


#     # -------------------------------------------------
#     # NORMALIZATION
#     # -------------------------------------------------

#     output_image = fn.crop_mirror_normalize(
#         images,
#         dtype=types.FLOAT,
#         output_layout="CHW",
#         mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#         std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
#     )


#     return output_image, transformed_points

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy

@pipeline_def
def crowd_dali_pipeline(args, split="train"):

    # -------------------------------------------------
    # INPUTS
    # -------------------------------------------------

    encoded_images = fn.external_source(name="images", no_copy=True)
    points = fn.external_source(name="labels", no_copy=True)
    impulses = fn.external_source(name="impulses", no_copy=True)
    # encoded_images, points = fn.external_source(num_outputs=2)
    jpegs = fn.io.file.read(encoded_images)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB, hw_decoder_load=0.9)


    # -------------------------------------------------
    # DECODE
    # -------------------------------------------------

    # images = fn.decoders.image(
    #     encoded_images,
    #     device="mixed",
    #     output_type=types.RGB
    # )


    # -------------------------------------------------
    # IMAGE SIZE
    # -------------------------------------------------

    # shape = fn.shapes(images)

    # height = fn.slice(shape, 0, 1, axes=[0])
    # width  = fn.slice(shape, 1, 1, axes=[0])
    shape = images.shape()

    height = fn.slice(shape, 0, 1, axes=[0])
    width  = fn.slice(shape, 1, 1, axes=[0])


    # -------------------------------------------------
    # RANDOM CROP PARAMETERS
    # -------------------------------------------------

    if split == "train":

        # crop_scale = fn.random.uniform(
        #     range=[args.min_scale, args.max_scale]
        # )

        # crop_w = crop_scale * width
        # crop_h = crop_scale * height

        # crop_x = fn.random.uniform(range=[0.0, 1.0]) * (width - crop_w)
        # crop_y = fn.random.uniform(range=[0.0, 1.0]) * (height - crop_h)

        # images = fn.slice(
        #     images,
        #     start=fn.cat(crop_y, crop_x),
        #     shape=fn.cat(crop_h, crop_w),
        #     axes=[0, 1]
        # )

        crop_scale = fn.random.uniform(range=[args.min_scale, args.max_scale])

        crop_w = crop_scale * width
        crop_h = crop_scale * height

        crop_x = fn.random.uniform(range=[0.0, 1.0]) * (width - crop_w)
        crop_y = fn.random.uniform(range=[0.0, 1.0]) * (height - crop_h)

        start = fn.cat(crop_y, crop_x).cpu()
        crop_shape = fn.cat(crop_h, crop_w).cpu()

        images = fn.slice(
            images,
            start=start,
            shape=crop_shape,
            axes=[0,1]
        )
        impulses = fn.slice(
            impulses,
            start=start,
            shape=crop_shape,
            axes=[0,1]
        )
        flip_coin = fn.random.coin_flip(probability=0.5)

    else:

        crop_x = 0
        crop_y = 0
        crop_w = width
        crop_h = height
        flip_coin = 0


    # -------------------------------------------------
    # RESIZE + FLIP
    # -------------------------------------------------

    images = fn.resize(images, size=[args.input_size, args.input_size])

    images = fn.flip(images, horizontal=flip_coin)
    impulses = fn.resize(impulses, size=[args.input_size, args.input_size], interp_type=types.INTERP_NN)

    impulses = fn.flip(impulses, horizontal=flip_coin)


    # -------------------------------------------------
    # LABEL TRANSFORM
    # -------------------------------------------------

    scale_x = args.input_size / crop_w
    scale_y = args.input_size / crop_h

    offset = fn.cat(crop_x, crop_y)

    points = points - offset

    scale = fn.cat(scale_x, scale_y)

    points = points * scale


    # split coordinates
    x = fn.slice(points, 0, 1, axes=[1])
    y = fn.slice(points, 1, 1, axes=[1])


    # flip transform
    x_flip = args.input_size - 1 - x

    flipped = fn.cat(x_flip, y, axis=1)

    mask = fn.cast(flip_coin, dtype=types.FLOAT)

    points = mask * flipped + (1 - mask) * points

    transformed_points = points

    # -------------------------------------------------
    # DENSITY MAP GENERATION
    # -------------------------------------------------

    # points_clamped = fn.clip(
    #     transformed_points,
    #     min=0.0,
    #     max=float(args.input_size - 1)
    # )

    x = fn.slice(transformed_points, 0, 1, axes=[1])
    y = fn.slice(transformed_points, 1, 1, axes=[1])

    x = fn.cast(x, dtype=types.INT32)
    y = fn.cast(y, dtype=types.INT32)

    # density_map = fn.zeros(
    #     shape=[args.input_size, args.input_size],
    #     dtype=types.FLOAT
    # )

    # density_map = fn.coord_transform(
    #     density_map,
    #     x=x,
    #     y=y,
    #     value=1.0
    # )

    # density_map = fn.coord_transform(density_map, x=x, y=y, value=1.0)
    # density_map = fn.expand_dims(density_map, axes=[0])
    density_map = fn.expand_dims(impulses, axes=[0])

    # -------------------------------------------------
    # DENSITY MAP GENERATION (GPU)
    # -------------------------------------------------

    # density_map = fn.splat(
    #     transformed_points,
    #     shape=[args.input_size, args.input_size],
    #     dtype=types.FLOAT
    # )

    # if args.sigma is not None:

    #     density_map = fn.gaussian_blur(
    #         density_map,
    #         sigma=args.sigma
    #     )

    # density_map = fn.expand_dims(density_map, axes=[0])


    # impulses = fn.external_source(name="impulses")

    # density_map = impulses

    # if args.sigma is not None:    # sigma always none
    #     density_map = fn.gaussian_blur(
    #         density_map,
    #         sigma=args.sigma
    #     )

    # density_map = fn.expand_dims(density_map, axes=[0])

    # -------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------

    output_image = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )


    # -------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------

    return output_image, transformed_points, density_map




def get_dataloader_dali(args, split="train", ddp=False):

    dataset = datasets.CrowdDali(
        dataset=args.dataset,
        split=split,
        return_filename=False,
        input_size=args.input_size,
    )

    # ----------------------------
    # Build pipeline
    # ----------------------------

    pipe = crowd_dali_pipeline(
        args=args,
        split=split,
        batch_size=args.batch_size,
        num_threads=3,
        device_id=0,
        prefetch_queue_depth=2 * args.num_workers,
    )

    # ----------------------------
    # DALI server
    # ----------------------------

    dali_server = dali_proxy.DALIServer(pipe)

    # ----------------------------
    # DataLoader
    # ----------------------------

    loader = dali_proxy.DataLoader(
        dali_server,
        dataset,
        batch_size=args.batch_size if split == "train" else 1,
        num_workers=args.num_workers,
        drop_last=(split == "train"),
    )

    return loader

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
    
# def get_dataloader_dali(args: ArgumentParser, split: str = "train", ddp: bool = False) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
#     if split == "train":  # train, strong augmentation
#         transforms = Compose([
#             datasets.RandomResizedCrop((args.input_size, args.input_size), scale=(args.min_scale, args.max_scale)),
#             datasets.RandomHorizontalFlip(),
#             datasets.RandomApply([
#                 datasets.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
#                 datasets.GaussianBlur(kernel_size=args.kernel_size, sigma=(0.1, 5.0)),
#                 datasets.PepperSaltNoise(saltiness=args.saltiness, spiciness=args.spiciness),
#             ], p=(args.jitter_prob, args.blur_prob, args.noise_prob)),
#         ])

#     # elif args.sliding_window:
#     #     if args.resize_to_multiple:
#     #         transforms = datasets.Resize2Multiple(args.window_size, stride=args.stride)
#     #     elif args.zero_pad_to_multiple:
#     #         transforms = datasets.ZeroPad2Multiple(args.window_size, stride=args.stride)
#     #     else:
#     #         transforms = None

#     else:
#         transforms = None

#     dataset = datasets.CrowdDali(
#         dataset=args.dataset,
#         split=split,
#         transforms=None,
#         sigma=None,
#         return_filename=False,
#         num_crops= 1,
#     )

#     if ddp and split == "train":  # data_loader for training in DDP
#         sampler = DistributedSampler(dataset)
#         data_loader = DataLoader(
#             dataset,
#             batch_size=args.batch_size,
#             sampler=sampler,
#             num_workers=args.num_workers,
#             pin_memory=True,
#             collate_fn=datasets.collate_fn,
#         )
#         return data_loader, sampler

#     elif split == "train":  # data_loader for training
#         data_loader = DataLoader(
#             dataset,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=args.num_workers,
#             pin_memory=True,
#             collate_fn=datasets.collate_fn,
#         )
#         return data_loader, None

#     else:  # data_loader for evaluation
#         data_loader = DataLoader(
#             dataset,
#             batch_size=1,  # Use batch size 1 for evaluation
#             shuffle=False,
#             num_workers=args.num_workers,
#             pin_memory=True,
#             collate_fn=datasets.collate_fn,
#         )
#         return data_loader
