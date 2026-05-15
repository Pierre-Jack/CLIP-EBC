import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import os, json
import wandb

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model

from utils import setup, cleanup, init_seeds, get_logger, barrier
from utils import get_dataloader, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint
from utils import get_writer, update_train_result, update_eval_result, log
from train import train
from eval import evaluate
from losses.cmar_loss import CrossModalAlignmentRankingLoss

parser = ArgumentParser(description="Train an EBC model.")

# Parameters for model
parser.add_argument("--model", type=str, default="vgg19_ae", help="The model to train.")
parser.add_argument("--input_size", type=int, default=448, help="The size of the input image.")
parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32], help="The reduction factor of the model.")
parser.add_argument("--regression", action="store_true", help="Use blockwise regression instead of classification.")
parser.add_argument("--truncation", type=int, default=None, help="The truncation of the count.")
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"],
                    help="The representative count values of bins.")
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"],
                    help="The type of the prompt.")
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "coarse"],
                    help="The granularity of the bins.")
parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0., help="The dropout rate for visual prompt tokens.")
parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")
parser.add_argument("--lambda_cmar", type=float, default=0.1, help="Weight for the CMAR loss.")

# Parameters for dataset
parser.add_argument("--dataset", type=str, required=True, help="The dataset to train on.")
parser.add_argument("--dataset_path", type=str, required=True, help="The path to the dataset.")
parser.add_argument("--num_crops", type=int, default=1, help="The number of crops for multi-crop training.")
parser.add_argument("--min_scale", type=float, default=1.0, help="The minimum scale for random scale augmentation.")
parser.add_argument("--max_scale", type=float, default=2.0, help="The maximum scale for random scale augmentation.")
parser.add_argument("--brightness", type=float, default=0.1, help="The brightness factor for random color jitter.")
parser.add_argument("--contrast", type=float, default=0.1, help="The contrast factor for random color jitter.")
parser.add_argument("--saturation", type=float, default=0.1, help="The saturation factor for random color jitter.")
parser.add_argument("--hue", type=float, default=0.0, help="The hue factor for random color jitter.")
parser.add_argument("--kernel_size", type=int, default=5, help="The kernel size for Gaussian blur.")
parser.add_argument("--saltiness", type=float, default=1e-3, help="The saltiness for pepper salt noise.")
parser.add_argument("--spiciness", type=float, default=1e-3, help="The spiciness for pepper salt noise.")
parser.add_argument("--jitter_prob", type=float, default=0.2, help="The probability for random color jitter.")
parser.add_argument("--blur_prob", type=float, default=0.2, help="The probability for random gaussian blur.")
parser.add_argument("--noise_prob", type=float, default=0.2, help="The probability for random noise.")
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window for evaluation.")
parser.add_argument("--window_size", type=int, default=None, help="The size of the sliding window.")
parser.add_argument("--stride", type=int, default=None, help="The stride of the sliding window.")
parser.add_argument("--zero_pad_to_multiple", action="store_true",
                    help="Zero pad the image to a multiple of the window size.")
parser.add_argument("--resize_to_multiple", action="store_true",
                    help="Resize the image to a multiple of the window size.")

# Parameters for training
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the dataloader.")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size for training.")
parser.add_argument("--epochs", type=int, default=1000, help="The number of epochs to train.")
parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay.")
parser.add_argument("--weight_count_loss", type=float, default=1.0, help="The weight for the count loss.")
parser.add_argument("--count_loss", type=str, default="mae", choices=["mae", "mse", "none"], help="The type of count loss.")
parser.add_argument("--weight_ot", type=float, default=0.1, help="The weight for Optimal Transport loss.")
parser.add_argument("--weight_tv", type=float, default=0.01, help="The weight for Total Variation loss.")
parser.add_argument("--warmup_epochs", type=int, default=10, help="The number of warmup epochs.")
parser.add_argument("--warmup_lr", type=float, default=1e-6, help="The learning rate during warmup.")
parser.add_argument("--T_0", type=int, default=5, help="The number of epochs for the first restart.")
parser.add_argument("--T_mult", type=int, default=2, help="The factor that increases T_i after a restart.")
parser.add_argument("--eta_min", type=float, default=1e-7, help="The minimum learning rate.")
parser.add_argument("--loss_fn", type=str, default="ce", choices=["ce", "bce", "dace"],
                    help="The loss function to use.")
parser.add_argument("--label_smoothing", type=float, default=0.0, help="The label smoothing parameter for CE loss.")
parser.add_argument("--entropy_weight", type=float, default=0.0, help="The weight of the entropy loss.")
parser.add_argument("--count_weight", type=float, default=0.0, help="The weight of the count loss.")
parser.add_argument("--tau", type=float, default=1.0, help="The temperature for DACE loss.")
parser.add_argument("--dm_loss", type=str, default="none", choices=["none", "l1", "l2", "dmcount"],
                    help="The density map loss to use.")
parser.add_argument("--dm_weight", type=float, default=0.0, help="The weight of the density map loss.")
parser.add_argument("--color_jitter", action="store_true", help="Use color jitter data augmentation.")
parser.add_argument("--gray_scale", action="store_true", help="Use gray scale data augmentation.")

# Parameters for logging
parser.add_argument("--output_dir", type=str, default=os.path.join(current_dir, "outputs"),
                    help="The directory to save the outputs.")

# Parameters for DDP
parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel.")
parser.add_argument("--port", type=str, default="12345", help="The port for DDP.")


def worker(rank, args, config):
    if args.ddp:
        print(f"Starting DDP worker {rank}.")
        setup(rank, args.world_size, args.port)
        init_seeds(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        init_seeds(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Corrected get_logger (takes 1 argument: the file path)
    model_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(model_dir, f"train_rank{rank}.log")
    logger = get_logger(log_file)

    log(logger, epoch=None, total_epochs=None, message=f"Args: {json.dumps(vars(args), indent=4)}")

    # 2. Corrected get_dataloader (returns tuple for train, single for val)
    train_loader_tuple = get_dataloader(args, split="train", ddp=args.ddp)
    if args.ddp:
        train_loader, train_sampler = train_loader_tuple
    else:
        train_loader, train_sampler = train_loader_tuple, None

    # Force batch_size to 1 for validation so PyTorch doesn't try to stack images of different sizes!
    original_bs = args.batch_size
    args.batch_size = 1
    val_loader = get_dataloader(args, split="val", ddp=False)
    args.batch_size = original_bs  # restore it back to 8 for the next training epoch

    # 3. Load model correctly
    model = get_model(
        backbone=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        bins=args.bins,
        anchor_points=args.anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    ).to(device)

    # load optimizer, scheduler, loss_fn
    optimizer, scheduler = get_optimizer(args, model)
    try:
        loss_fn = get_loss_fn(args, device)
    except TypeError:
        loss_fn = get_loss_fn(args)

    cmar_criterion = CrossModalAlignmentRankingLoss(margin=0.1)

    start_epoch = 1
    best_val_scores = {
        "mae": [float("inf")],
        "mse": [float("inf")],
        "rmse": [float("inf")],
        "nae": [float("inf")]
    }
    hist_val_scores = {
        "mae": [], "mse": [], "rmse": [], "nae": []
    }

    ckpt_dir = "/kaggle/working/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = "/kaggle/working/checkpoints/latest.pth"

    if os.path.exists(ckpt_path):
        start_epoch, best_val_scores = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        log(logger, epoch=None, total_epochs=None,
            message=f"Loaded checkpoint from {ckpt_path}. Resuming from epoch {start_epoch}.")

    if args.ddp:
        model = DDP(model, device_ids=[rank])
    scaler = GradScaler()

    # 4. Initialize W&B and TensorBoard writer (ONLY on the main GPU!)
    if rank == 0 or not args.ddp:
        # sync_tensorboard=True automatically uploads everything your writer logs!
        wandb.init(
            project="clip-ebc-de-clip",
            name=args.model_name,
            config=vars(args),
            sync_tensorboard=True
        )
        writer = get_writer(model_dir)
    else:
        writer = None

    for epoch in range(start_epoch, args.epochs + 1):
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        log(logger, epoch, args.epochs)

        # 5. Corrected train() call to match the updated train.py signature
        model, optimizer, scaler, train_result = train(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            grad_scaler=scaler,
            device=device,
            rank=rank,
            nprocs=args.world_size if args.ddp else 1,
            cmar_criterion=cmar_criterion,
            lambda_cmar=args.lambda_cmar
        )

        if scheduler is not None:
            scheduler.step()

        # 6. Corrected update_train_result (takes 3 arguments)
        if writer is not None:
            update_train_result(epoch, train_result, writer)
        log(logger, epoch, args.epochs, loss_info=train_result)

        if epoch % args.eval_freq == 0 or epoch == args.epochs:

            # 1. Split the validation dataset evenly across all GPUs
            from torch.utils.data.distributed import DistributedSampler
            val_sampler = DistributedSampler(val_loader.dataset, shuffle=False) if args.ddp else None

            # 2. Rebuild the safe dataloader with the DistributedSampler
            safe_val_loader = DataLoader(
                val_loader.dataset,
                batch_size=1,
                shuffle=False,  # Must be False when using a sampler
                sampler=val_sampler,
                num_workers=0,
                pin_memory=False,
                collate_fn=val_loader.collate_fn
            )

            # 3. BOTH GPUs run evaluation in parallel (doing half the work each)
            val_result = evaluate(
                model,
                safe_val_loader,
                device,
                sliding_window=args.sliding_window,
                window_size=args.window_size,
                stride=args.stride
            )

            # 4. Sync and average the results across GPUs so the metrics are perfectly accurate
            if args.ddp:
                for k, v in val_result.items():
                    tensor = torch.tensor(v, dtype=torch.float32).to(device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    val_result[k] = (tensor / args.world_size).item()

            # 5. Only Rank 0 handles the logging and saving
            if rank == 0 or not args.ddp:
                state_dict = model.module.state_dict() if args.ddp else model.state_dict()

                hist_val_scores, best_val_scores = update_eval_result(
                    epoch, val_result, hist_val_scores, best_val_scores, writer, state_dict, ckpt_dir
                )
                log(logger, epoch, args.epochs, curr_scores=val_result, best_scores=best_val_scores)

                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    train_result,
                    hist_val_scores,
                    best_val_scores,
                    ckpt_dir
                )

            # A safe barrier that won't timeout because both GPUs arrive here at the exact same time!
            barrier(args.ddp)

    if writer is not None:
        writer.close()

    if rank == 0 or not args.ddp:
        print("Training completed. Best scores:")
        for k in best_val_scores.keys():
            scores = " ".join([f"{best_val_scores[k][i]:.4f};" for i in range(len(best_val_scores[k]))])
            print(f"    {k}: {scores}")

        # safely upload final logs before exiting
        wandb.finish()

    cleanup(ddp=args.ddp)


def main():
    args = parser.parse_args()
    args.model = args.model.lower()
    args.dataset = standardize_dataset_name(args.dataset)

    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None

    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None

    if "clip" not in args.model:
        args.prompt_type = None

    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (
                    args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both zero pad and resize to multiple."

    else:
        args.window_size = None
        args.stride = None
        args.zero_pad_to_multiple = False
        args.resize_to_multiple = False

    args.num_workers = min(args.num_workers, os.cpu_count())
    args.eval_freq = 1

    if not args.regression:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)][args.dataset]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else \
        config["anchor_points"][args.granularity]["middle"]

        args.bins = [(float(b[0]), float(b[1])) for b in bins]
        args.anchor_points = [float(p) for p in anchor_points]
    else:
        args.bins = None
        args.anchor_points = None

    args.model_name = f"{args.model}_{args.input_size}_{args.reduction}"

    if not args.regression:
        args.model_name += f"_{args.truncation}_{args.granularity}"

        if "clip" in args.model:
            args.model_name = args.model_name.replace(f"{args.model}_", f"{args.model}_{args.prompt_type}_")
            if "vit" in args.model:
                args.model_name += f"_vpt{args.num_vpt}"
                if args.shallow_vpt:
                    args.model_name += "s"

                if args.vpt_drop > 0:
                    args.model_name += f"_drop{args.vpt_drop}"

    else:
        args.model_name += "_reg"

    args.model_name += f"_{args.tau}"

    if args.entropy_weight > 0:
        args.model_name += f"_ent{args.entropy_weight}"
    if args.count_weight > 0:
        args.model_name += f"_cnt{args.count_weight}"

    if args.dm_loss != "none":
        args.model_name += f"_{args.dm_loss}"
        if args.dm_loss != "dmcount":
            args.model_name += f"{args.dm_weight}"

    if args.color_jitter:
        args.model_name += "_cj"
    if args.gray_scale:
        args.model_name += "_gs"

    if args.ddp:
        args.world_size = torch.cuda.device_count()
        mp.spawn(worker, args=(args, config), nprocs=args.world_size, join=True)
    else:
        worker(0, args, config)


if __name__ == "__main__":
    main()