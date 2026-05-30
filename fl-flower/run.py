"""
Flower-based FL compression study.

Usage examples:
  python run.py --dataset cifar10 --compressor none
  python run.py --dataset cifar10 --compressor quantization --bits 8
  python run.py --dataset cifar10 --compressor sz --error-bound 0.01
  python run.py --dataset cifar10 --compressor none --lr-decay
  python run.py --dataset cifar10 --compressor sz --error-bound 0.001 --lr-decay
  python run.py --dataset cifar10 --schedule A
  python run.py --dataset cifar10 --schedule C --lr-decay

Key flags:
  --num-clients   number of simulated clients  (default: 10)
  --alpha         Dirichlet concentration param (default: 0.5, lower = more non-IID)
  --rounds        FL communication rounds       (default: 50)
  --local-epochs  local SGD epochs per round    (default: 2)
  --lr-decay      cosine LR decay 0.01 → 0.001 over --rounds (client-side)
  --schedule      adaptive SZ error-bound schedule: A, B, or C
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import flwr as fl

# nnpack crashes on Apple Silicon (Trace/BPT trap: 5); disable it globally
torch.backends.nnpack.enabled = False

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from src.compressors.base import Compressor
from src.compressors.no_compression import NoCompression
from src.compressors.quantization import QuantizationCompressor
from src.compressors.sz import SZCompressor, SZRelCompressor
from src.compressors.sz_usnr import SZUsnrRmsCompressor
from src.compressors.hybrid import HybridCompressor

from models import build_model, get_parameters, set_parameters
from data import load_datasets
from client import FlowerClient
from strategy import LoggingFedAvg, find_checkpoint
from adaptive_strategy import AdaptiveSZStrategy


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FL compression study with Flower")
    p.add_argument("--dataset",      choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--compressor",   choices=["none", "quantization", "sz", "sz_rel", "sz_usnr_rms", "hybrid"], default="none")
    p.add_argument("--bits",         type=int,   default=8, choices=[1, 2, 4, 8, 16])
    p.add_argument("--error-bound",  type=float, default=0.01)
    p.add_argument("--num-clients",  type=int,   default=10)
    p.add_argument("--alpha",        type=float, default=0.5,
                   help="Dirichlet alpha: lower=more non-IID, 1000=IID")
    p.add_argument("--rounds",       type=int,   default=50)
    p.add_argument("--local-epochs", type=int,   default=2)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--output",       type=str,   default=None)
    p.add_argument("--lr-decay",     action="store_true",
                   help="Cosine LR decay from 0.01 to 0.001 over --rounds (client-side)")
    p.add_argument("--schedule",     choices=["A", "B", "C"], default=None,
                   help="Adaptive SZ error-bound schedule (implies --compressor sz)")
    p.add_argument("--seed",         type=int, default=0,
                   help="Random seed for data partition and model initialisation")
    # USNR-SZ options
    p.add_argument("--usnr-alpha",   type=float, default=0.1,
                   help="USNR-RMS alpha: eb = clip(alpha*rms, eb_min, eb_max)")
    p.add_argument("--usnr-eb-min",  type=float, default=1e-6,
                   help="USNR-RMS minimum error bound")
    p.add_argument("--usnr-eb-max",  type=float, default=1.0,
                   help="USNR-RMS maximum error bound")
    p.add_argument("--usnr-diagnostics", action="store_true",
                   help="Write per-tensor diagnostics CSV (smoke tests only)")
    return p.parse_args()


def _build_compressor(args: argparse.Namespace) -> Compressor:
    if args.schedule is not None:
        return SZCompressor(error_bound=0.001)  # initial; overridden each round via config
    if args.compressor == "none":
        return NoCompression()
    if args.compressor == "quantization":
        return QuantizationCompressor(bits=args.bits)
    if args.compressor == "sz":
        return SZCompressor(error_bound=args.error_bound)
    if args.compressor == "sz_rel":
        return SZRelCompressor(error_bound=args.error_bound)
    if args.compressor == "sz_usnr_rms":
        return SZUsnrRmsCompressor(
            alpha=args.usnr_alpha,
            eb_min=args.usnr_eb_min,
            eb_max=args.usnr_eb_max,
            diagnostics=args.usnr_diagnostics,
        )
    if args.compressor == "hybrid":
        return HybridCompressor(
            sz_error_bound=args.error_bound,
            quant_bits=args.bits,
        )
    raise ValueError(args.compressor)


def _run_name(args: argparse.Namespace, compressor: Compressor) -> str:
    if args.schedule is not None:
        name = f"sz_schedule_{args.schedule.lower()}"
    elif args.compressor == "sz_usnr_rms":
        name = f"sz_usnr_rms_a{args.usnr_alpha}"
    else:
        name = compressor.name
    if args.lr_decay:
        name += "_cosine"
    return name


# ── Centralized evaluation (runs on server after every round) ─────────────────

def make_evaluate_fn(model, testloader, device, strategy_holder):
    def evaluate_fn(server_round, parameters, config):
        set_parameters(model, parameters)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_sum += criterion(out, y).item() * y.size(0)
                correct  += out.argmax(1).eq(y).sum().item()
                total    += y.size(0)
        loss = loss_sum / total
        acc  = 100.0 * correct / total

        strategy = strategy_holder[0]
        if strategy is not None:
            strategy.log_round(server_round, acc, loss)
            ratio = strategy.round_fit_metrics.get(server_round, {}).get("compression_ratio", 1.0)
        else:
            ratio = 1.0
        print(f"  [server] round {server_round:>3}  acc={acc:.2f}%  loss={loss:.4f}  ratio={ratio:.2f}x")
        return loss, {"val_accuracy": acc}

    return evaluate_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def _already_done(output_path: str, run_name: str, seed: int, rounds: int) -> bool:
    if not os.path.exists(output_path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(output_path)
        mask = (df["compressor"] == run_name) & (df["seed"] == seed)
        done = df[mask]["round"].max()
        return int(done) >= rounds
    except Exception:
        return False


def main() -> None:
    args = _parse()
    compressor = _build_compressor(args)
    run_name = _run_name(args, compressor)

    if args.output is None:
        os.makedirs("results", exist_ok=True)
        if args.schedule is not None:
            args.output = f"results/flower_{args.dataset}_adaptive.csv"
        elif args.lr_decay:
            args.output = f"results/flower_{args.dataset}_lr_decay.csv"
        else:
            args.output = f"results/flower_{args.dataset}_sweep.csv"

    if _already_done(args.output, run_name, args.seed, args.rounds):
        print(f"[skip] {run_name} seed={args.seed} already has {args.rounds} rounds in {args.output}")
        return

    if args.schedule is not None:
        checkpoint_dir = f"checkpoints/flower_{args.dataset}_adaptive"
    elif args.lr_decay:
        checkpoint_dir = f"checkpoints/flower_{args.dataset}_lr_decay"
    else:
        checkpoint_dir = f"checkpoints/flower_{args.dataset}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed data partition and model initialisation
    import numpy as _np
    torch.manual_seed(args.seed)
    _np.random.seed(args.seed)

    ckpt_arrays, start_round = find_checkpoint(checkpoint_dir, run_name, seed=args.seed)
    remaining_rounds = args.rounds - start_round
    if ckpt_arrays is not None:
        print(f"[resume] Found checkpoint at round {start_round} — running {remaining_rounds} more rounds")
        initial_params = fl.common.ndarrays_to_parameters(ckpt_arrays)
    else:
        initial_params = None

    if remaining_rounds <= 0:
        print(f"[skip] {run_name} checkpoint already at round {start_round} >= {args.rounds}")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Run:        {run_name}")
    print(f"  Clients:    {args.num_clients}  |  alpha={args.alpha}")
    print(f"  Rounds:     {start_round}+{remaining_rounds}={args.rounds}  |  local_epochs={args.local_epochs}")
    print(f"  LR decay:   {args.lr_decay}  |  Schedule: {args.schedule}  |  Seed: {args.seed}")
    print(f"  Device:     {device}")
    print(f"  Output:     {args.output}")
    print(f"{'='*60}\n")

    train_loaders, test_loader = load_datasets(
        args.dataset, args.num_clients, args.alpha, args.batch_size, seed=args.seed
    )

    server_model = build_model(args.dataset).to(device)
    if ckpt_arrays is not None:
        set_parameters(server_model, ckpt_arrays)

    if initial_params is None:
        initial_params = fl.common.ndarrays_to_parameters(get_parameters(server_model))

    strategy_holder = [None]

    diag_path = None
    if getattr(args, "usnr_diagnostics", False):
        diag_stem = os.path.splitext(args.output)[0]
        diag_path = f"{diag_stem}_diag_{run_name}_s{args.seed}.csv"

    strategy_kwargs = dict(
        output_path=args.output,
        compressor_name=run_name,
        num_clients=args.num_clients,
        alpha=args.alpha,
        checkpoint_dir=checkpoint_dir,
        round_offset=start_round,
        lr_decay=args.lr_decay,
        seed=args.seed,
        diagnostics_path=diag_path,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=initial_params,
        evaluate_fn=make_evaluate_fn(server_model, test_loader, device, strategy_holder),
    )

    if args.schedule is not None:
        strategy = AdaptiveSZStrategy(schedule=args.schedule, **strategy_kwargs)
    else:
        strategy = LoggingFedAvg(**strategy_kwargs)

    strategy_holder[0] = strategy

    def client_fn(context) -> fl.client.Client:
        try:
            from flwr.common import Context
            cid = int(context.node_config.get("partition-id", context.node_id))
        except Exception:
            cid = int(context)
        model = build_model(args.dataset)
        return FlowerClient(
            cid=cid,
            model=model,
            trainloader=train_loaders[cid],
            local_epochs=args.local_epochs,
            compressor=compressor,
            device=device,
        ).to_client()

    import torch as _torch
    import ray as _ray
    _gpu_frac = (1.0 / args.num_clients) if _torch.cuda.is_available() else 0.0
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=remaining_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": _gpu_frac},
            ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
        )
    finally:
        # Shut Ray down cleanly so the next run starts with a fresh cluster.
        # Without this, leftover worker processes cause lock timeouts on macOS.
        if _ray.is_initialized():
            _ray.shutdown()

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
