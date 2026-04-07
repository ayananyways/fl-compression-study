import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compressors.hybrid import HybridCompressor
from src.compressors.no_compression import NoCompression
from src.compressors.quantization import QuantizationCompressor
from src.compressors.sz import SZCompressor
from src.training.trainer import Trainer
from src.utils.dist_utils import is_main_process
from src.utils.logger import ResultsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] rank=%(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    base_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)
    config.update(override)
    return config


def _build_compressor(config: dict):
    kind = config.get("compressor", "none")
    if kind == "none":
        return NoCompression()
    elif kind == "quantization":
        bits = config.get("bits", 8)
        return QuantizationCompressor(bits=bits)
    elif kind == "sz":
        error_bound = config.get("error_bound", 0.01)
        return SZCompressor(error_bound=error_bound)
    elif kind == "hybrid":
        bits = config.get("bits", 8)
        error_bound = config.get("error_bound", 0.01)
        return HybridCompressor(bits=bits, error_bound=error_bound)
    else:
        raise ValueError(f"Unknown compressor: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    config = _load_config(args.config)
    config["rank"] = rank
    config["world_size"] = world_size

    compressor = _build_compressor(config)
    trainer = Trainer(config=config, rank=rank, world_size=world_size, compressor=compressor)

    results_logger = None
    if is_main_process(rank):
        results_logger = ResultsLogger(
            output_path=config["output_path"],
            experiment_id=config["experiment_id"],
        )

    try:
        trainer.setup()
        total_rounds = config.get("total_rounds", 50)

        for round_num in range(total_rounds):
            metrics = trainer.train_one_round(round_num)

            if is_main_process(rank):
                results_logger.log(metrics)
                logger.info(
                    "Round %d/%d — loss=%.4f val_loss=%.4f val_acc=%.4f "
                    "compress=%.3fs decompress=%.3fs bytes=%d ratio=%.2f",
                    round_num + 1,
                    total_rounds,
                    metrics.train_loss,
                    metrics.val_loss,
                    metrics.val_accuracy,
                    metrics.compress_time_s,
                    metrics.decompress_time_s,
                    metrics.bytes_sent,
                    metrics.compression_ratio,
                )

    except KeyboardInterrupt:
        if is_main_process(rank):
            logger.info("Training interrupted by user.")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
