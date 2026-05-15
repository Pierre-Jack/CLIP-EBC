import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from argparse import ArgumentParser
import os, json

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
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
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"], help="The representative count values of bins.")
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"], help="The type of the prompt.")
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "coarse"], help="The granularity of the bins.")
parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0., help="The dropout rate for visual prompt tokens.")
parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")
parser.add_argument("--lambda_cmar", type=float, default=0.1, help="Weight for the CMAR loss.")

# Parameters for dataset
parser.add_argument("--dataset", type=str, required=True, help="The dataset to train on.")
parser.add_argument("--dataset_path", type=str, required=True, help="The path to the dataset.")
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window for evaluation.")
parser.add_argument("--window_size", type=int, default=None, help="The size of the sliding window.")
parser.add_argument("--stride", type=int, default=None, help="The stride of the sliding window.")
parser.add_argument("--zero_pad_to_multiple", action="store_true", help="Zero pad the image to a multiple of the window size.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to a multiple of the window size.")

# Parameters for training
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the dataloader.")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size for training.")
parser.add_argument("--epochs", type=int, default=1000, help="The number of epochs to train.")
parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay.")
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
parser.add_argument("--output_dir", type=str, default=os.path.join(current_dir, "outputs"), help="The directory to save the outputs.")

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

    logger = get_logger(args.output_dir, args.model_name, rank)

    log(logger, rank, f"Args: {json.dumps(vars(args), indent=4)}")

    # load dataset
    train_loader, val_loader = get_dataloader(args, config, rank)

    # load model
    model = get_model(args, config, device)

    # load optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, args)

    # load loss function
    loss_fn = get_loss_fn(args, config, device)

    cmar_criterion = CrossModalAlignmentRankingLoss(margin=0.1)

    start_epoch = 1
    best_val_scores = {
        "mae": [float("inf")],
        "mse": [float("inf")],
        "nae": [float("inf")]
    }

    # load checkpoint
    ckpt_path = os.path.join(args.output_dir, args.model_name, "checkpoints", "latest.pth")
    if os.path.exists(ckpt_path):
        start_epoch, best_val_scores = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        log(logger, rank, f"Loaded checkpoint from {ckpt_path}. Resuming from epoch {start_epoch}.")

    if args.ddp:
        model = DDP(model, device_ids=[rank])
    scaler = GradScaler()

    # setup tensorboard
    writer = get_writer(args.output_dir, args.model_name, rank)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        log(logger, rank, f"Epoch {epoch}/{args.epochs}")

        # train
        train_result = train(model, train_loader, loss_fn, cmar_criterion, args.lambda_cmar, optimizer, scaler, device)
        scheduler.step()
        update_train_result(train_result, epoch, writer, logger, rank)

        # evaluate
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_result = evaluate(model, val_loader, device)

            if args.ddp:
                for k, v in val_result.items():
                    tensor = torch.tensor(v).to(device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    val_result[k] = (tensor / args.world_size).item()

            update_eval_result(val_result, epoch, best_val_scores, writer, logger, rank)

            if rank == 0 or not args.ddp:
                if val_result["mae"] == best_val_scores["mae"][-1]:
                    save_checkpoint(
                        epoch, model, optimizer, scheduler, best_val_scores,
                        os.path.join(args.output_dir, args.model_name, "checkpoints", "best_mae.pth")
                    )

            save_checkpoint(
                epoch, model, optimizer, scheduler, best_val_scores,
                os.path.join(args.output_dir, args.model_name, "checkpoints", "latest.pth")
            )

            barrier(args.ddp)

    if writer is not None:
        writer.close()

    if rank == 0 or not args.ddp:
        print("Training completed. Best scores:")
        for k in best_val_scores.keys():
            scores = " ".join([f"{best_val_scores[k][i]:.4f};" for i in range(len(best_val_scores[k]))])
            print(f"    {k}: {scores}")

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

    config = get_config(args.dataset, args.input_size, args.reduction, args.truncation, args.granularity,
                        args.anchor_points)

    args.bins = config["bins"]
    args.truncation = config["truncation"]
    args.anchor_points = config["anchor_points"]

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