import json
import logging
import random
import selectors
import sys
from functools import cache
from typing import Protocol

import humanize
import numpy as np
import torch
from torch.distributed import broadcast_object_list
from torch.utils.data import DataLoader

from .config import Config
from .distributed import (
    global_leader_only,
    global_rank,
    is_global_leader,
    is_local_leader,
    local_leader_only,
)
from .engines import Engine, Engines, TrainFeeder
from .utils import to_device

_logger = logging.getLogger(__name__)
_engines: Engines
_command: str


def get_global_step():
    try:
        return _engines.global_step
    except:
        return None


def get_cfg():
    try:
        return _engines.cfg
    except:
        raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


def get_cmd():
    try:
        return _command
    except:
        raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


get_iteration = get_global_step


class EnginesLoader(Protocol):
    def __call__(self) -> Engines:
        ...


def load_engines(engines: dict[str, Engine], config: Config):
    engines = Engines(engines)
    engines.setup(config)
    engines.load_checkpoint()
    return engines


class EvalFn(Protocol):
    def __call__(self, *, engines: Engines):
        ...


class Logger(Protocol):
    def __call__(self, *, data: dict):
        ...


@cache
def _get_stdin_selector():
    selector = selectors.DefaultSelector()
    selector.register(fileobj=sys.stdin, events=selectors.EVENT_READ)
    return selector


def _non_blocking_input():
    global _command
    l = [""]
    if is_global_leader():
        s = ""
        selector = _get_stdin_selector()
        events = selector.select(timeout=0)
        for key, _ in events:
            s: str = key.fileobj.readline().strip()
            _logger.info(f'Get stdin "{s}".')
        l[0] = s
    broadcast_object_list(l, src=0)
    _command = l[0]
    return _command


def _make_infinite_epochs(dl):
    while True:
        _logger.info("New epoch starts.")
        yield from dl


@local_leader_only(default=None)
def logger(data):
    return _logger.info(json.dumps(data, indent=2, default=str))


def seed(seed):
    # Set up random seeds, after fork()
    random.seed(seed + global_rank())
    np.random.seed(seed + global_rank())
    torch.manual_seed(seed + global_rank())


def train(
    engines_loader: EnginesLoader,
    train_dl: DataLoader,
    train_feeder: TrainFeeder,
    eval_fn: EvalFn,
    logger: Logger = logger,
    use_tb: bool = False
):  
    engines = engines_loader()  # load ckpts
    cfg = engines.cfg

    if use_tb:
        logger("Use Tensorboard")
        import tensorflow as tf
        import datetime
        # Hide GPU from visible devices
        log_dir = f"{cfg.log_root}/{cfg.relpath}/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(log_dir)

    if is_local_leader():
        cfg.dump()
        _logger.info(cfg)

    # Setup global engines
    global _engines
    _engines = engines

    events = []

    eval_fn = global_leader_only(eval_fn)

    # Pre-loop command
    command = _non_blocking_input()
    if command in ["eval", "eval_quit"]:
        engines.eval()
        eval_fn(engines=engines)
        engines.train()
    if command in ["quit", "eval_quit"]:
        return

    # Training loop
    for batch_i, batch in enumerate(_make_infinite_epochs(train_dl)):  
        if engines.global_step >= cfg.max_iter:
            break

        batch = to_device(batch, torch.cuda.current_device())  
        # a batch consists of paths, speakers, texts, audio prompts, 8 levels of target discrete code and 1st level of target discrete code
        stats = engines.step(feeder=train_feeder, batch=batch)  
        elapsed_time = stats.get("elapsed_time", 0)
        logger(data=stats)
        if use_tb:
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', stats["model.loss"], step=engines.global_step)
                tf.summary.scalar('learning_rate', stats["model.lr"], step=engines.global_step)

        command = _non_blocking_input()

        if "@" in command:
            what, when = command.split("@")
            try:
                events.append((what, int(when)))
                _logger.info(f"Event {command} registered.")
            except Exception as e:
                _logger.error(e)
            command = ""

        # Commands are the current command plus the triggered (i.e. iteration >= trigger point) events
        events = [e for e in events if e[1] >= engines.global_step]
        commands = [command] + [e[0] for e in events if e[1] == engines.global_step]

        for command in commands:
            if command in ["event show", "event"]:
                msg = "Events:\n" + "\n".join(["@".join(map(str, e)) for e in events])
                _logger.info(msg)

            if command == "event clear":
                events.clear()

            if "time" in command:
                target_iter = cfg.max_iter
                if " to " in command:
                    try:
                        target_iter = int(command.split(" to ")[-1])
                    except Exception as e:
                        _logger.error(e)
                remaining_iters = target_iter - engines.global_step + 1
                remaining_time = int(remaining_iters * elapsed_time)
                _logger.info(humanize.precisedelta(remaining_time))

            save_ckpt_every = cfg.save_ckpt_every or cfg.eval_every

            saving_commands = ["save"]

            if cfg.save_on_quit:
                saving_commands.append("quit")

            if engines.global_step % save_ckpt_every == 0 or command in saving_commands:
                engines.save_checkpoint()

            if engines.global_step % cfg.eval_every == 0 or command in ["eval"]:
                engines.eval()
                eval_stats = eval_fn(engines=engines) 
                with train_summary_writer.as_default():
                    tf.summary.scalar('val_loss', eval_stats["loss.nll"], step=engines.global_step)
                engines.train()

            if command in ["quit"]:
                return
