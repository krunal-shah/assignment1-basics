import json
import sys
import torch
import functools
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
import logging
from dataclasses import dataclass, field
from jsonargparse import ArgumentParser, ActionConfigFile
from pathlib import Path
from cs336_basics.tokenizer import train_bpe, Tokenizer, get_training_batch, get_validation_batch_itr
from cs336_basics.transformer import TransformerLM, cross_entropy, save_checkpoint
from cs336_basics.optimizers import AdamW, gradient_clipping, cosine_lr_scheduler
import numpy as np


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    datapath: str = None
    val_datapath: str = None
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])


@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    context_length: int = 512
    num_layers: int = 8
    d: int = 512
    d_ff: int = 512 * 8 // 3
    num_heads: int = 32
    rope_theta: int = 10000
    device: str = "mps"


@dataclass
class OptimizerConfig:
    batch_size: int = 16
    num_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    max_norm: float = 1.0
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    warmup_itr: int = 50
    annealing_itr: int = 250


@dataclass
class LoggingConfig:
    run_id: str = None
    save_every_n: int = 200
    log_loss_every_n: int = 2
    run_validation_every_n: int = 5
    save_filepath: str = "data/models/"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--data", type=DataConfig, default=DataConfig())
    parser.add_argument("--model", type=TransformerConfig, default=TransformerConfig())
    parser.add_argument("--optimizer", type=OptimizerConfig, default=OptimizerConfig())
    parser.add_argument("--log", type=LoggingConfig, default=LoggingConfig())
    args = parser.parse_args()
    return args


def save_checkpoint_dir(args, step, model, optimizer):
    dirname = Path(args.log.save_filepath) / args.log.run_id
    filepath = dirname / f"step_{step}.ckpt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Step[{step}] Saving checkpoint to {filepath}")
    with open(
        filepath,
        "wb",
    ) as out:
        save_checkpoint(model, optimizer, step, out)
    with open(Path(dirname / "config.json"), "w") as fout:
        json.dump(args.as_dict(), fout, default=str, indent=4)
    logger.info("Checkpoint saved")


if __name__ == "__main__":
    args = parse_args()

    datapath = Path(args.data.datapath)
    tokenizer_path = Path(datapath.with_suffix(".tokenizer"))
    encoded_datapath = Path(datapath.with_suffix(".encoded"))

    val_datapath = Path(args.data.val_datapath)
    val_encoded_datapath = Path(val_datapath.with_suffix(".encoded"))

    if not tokenizer_path.exists():
        logger.info("Tokenizer not found, training")
        vocab, merges = train_bpe(datapath, args.model.vocab_size, args.data.special_tokens)
        tokenizer = Tokenizer(vocab, merges, args.data.special_tokens)
        tokenizer.save(tokenizer_path)

    if not encoded_datapath.exists():
        logger.info("Encoded data not found, encoding")
        tokenizer = Tokenizer.load(tokenizer_path)
        tokenizer.encode_file(datapath, encoded_datapath)
    if not val_encoded_datapath.exists():
        logger.info("Val encoded data not found, encoding")
        tokenizer = Tokenizer.load(tokenizer_path)
        tokenizer.encode_file(val_datapath, val_encoded_datapath)

    run_id = args.log.run_id

    if not run_id:
        run_id = f"{datetime.now(UTC).astimezone(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')}"
        args.log.run_id = run_id

    model = TransformerLM(
        vocab_size=args.model.vocab_size,
        context_length=args.model.context_length,
        num_layers=args.model.num_layers,
        d_model=args.model.d,
        num_heads=args.model.num_heads,
        d_ff=args.model.d_ff,
        rope_theta=args.model.rope_theta,
        device=args.model.device,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.optimizer.lr_max,
        betas=(args.optimizer.beta1, args.optimizer.beta2),
        weight_decay=args.optimizer.weight_decay,
    )

    lr_schedule = functools.partial(
        cosine_lr_scheduler,
        lr_max=args.optimizer.lr_max,
        lr_min=args.optimizer.lr_min,
        warmup_itr=args.optimizer.warmup_itr,
        annealing_itr=args.optimizer.annealing_itr,
    )

    loss_fn = cross_entropy

    encoded_data = np.memmap(encoded_datapath, dtype=np.int32, mode="r")
    val_encoded_data = np.memmap(val_encoded_datapath, dtype=np.int32, mode="r")

    for step in range(args.optimizer.num_steps):
        batch, targets = get_training_batch(
            encoded_data, args.optimizer.batch_size, args.model.context_length, args.model.device
        )

        model.zero_grad()
        loss = loss_fn(model(batch), targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.optimizer.max_norm)
        lr_scheduled_value = lr_schedule(step)
        optimizer.update_lr(lr_scheduled_value)
        optimizer.step()

        if step % args.log.log_loss_every_n == 0:
            logger.info(f"Step[{step}] Loss={loss.item()} LR={lr_scheduled_value}")

        if step % args.log.save_every_n == 0 and step:
            save_checkpoint_dir(args, step, model, optimizer)

        if step % args.log.run_validation_every_n == 0:
            logger.info(f"Step[{step}] Running validation")
            val_loss = 0.0
            num_val_examples = 0
            with torch.no_grad():
                model.eval()
                for val_batch, val_targets in get_validation_batch_itr(
                    val_encoded_data, args.optimizer.batch_size, args.model.context_length, args.model.device
                ):
                    val_loss_batch = loss_fn(model(val_batch), val_targets)
                    val_loss += val_loss_batch * val_batch.shape[0]
                    num_val_examples += val_batch.shape[0]
                model.train()

            logger.info(f"Step[{step}] Val loss: {val_loss.item() / num_val_examples}")

    save_checkpoint_dir(args, step, model, optimizer)
