import json
import logging
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np

from .config import cfg
from .data import create_train_val_dataloader
from .emb import qnt
from .utils import setup_logging, to_device, trainer
from .vall_e import get_model

_logger = logging.getLogger(__name__)


def load_engines():
    model = get_model(cfg.model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    _logger.info(f"Number of parameters of AR model: {params // 1e6}M.")

    engines = dict(
        model=trainer.Engine(
            model=model,
            config=cfg.ds_cfg,
        ),
    )

    return trainer.load_engines(engines, cfg)


def main():
    setup_logging(cfg.log_dir)

    train_dl, val_dl = create_train_val_dataloader()

    def train_feeder(engines, batch, name):
        model = engines["model"]

        if cfg.model.startswith("ar"):
            _ = model.forward(
                text_list=batch["text"],
                proms_list=batch["proms"],
                targ_list=batch["resp"],  
            )
        elif cfg.model.startswith("nar"):
            _ = model.forward(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resps_list=batch["resps"],
            )
        else:
            raise NotImplementedError(cfg.model)

        losses = model.gather_attribute("loss")

        loss = torch.stack([*losses.values()]).sum()

        stats = {}
        stats |= {k: v.item() for k, v in losses.items()}
        stats |= engines.gather_attribute("scalar")

        return loss, stats

    @torch.inference_mode()
    def run_eval(engines, name, dl):
        model = engines["model"]
        log_dir = cfg.log_dir / str(engines.global_step) / name
        stats = defaultdict(list)
        for batch in tqdm(dl):
            batch: dict = to_device(batch, cfg.device)

            if cfg.model.startswith("ar"):
                _ = model.forward(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    targ_list=batch["resp"], 
                )
            elif cfg.model.startswith("nar"):
                _ = model.forward(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    resps_list=batch["resps"],
                )
            else:
                raise NotImplementedError(cfg.model)

            losses = model.gather_attribute("loss")
            batch_stats = {k: v.item() for k, v in losses.items()}
            for k, v in batch_stats.items():
                stats[k].append(v)
            torch.cuda.empty_cache()

        ### generate audio output of validation set
        def generate():
            batch = next(iter(dl))   
            batch: dict = to_device(batch, cfg.device)
            if cfg.model.startswith("ar"):
                resp_list = model.forward(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    max_steps=cfg.max_val_ar_steps,
                    sampling_temperature=cfg.sampling_temperature,
                )
                resps_list = [r.unsqueeze(-1) for r in resp_list]
            elif cfg.model.startswith("nar"):
                resps_list = model.forward(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    resps_list=[r.unsqueeze(-1) for r in batch["resp"]],
                    sampling_temperature=cfg.sampling_temperature,
                )
            else:
                raise NotImplementedError(cfg.model)
            
            for path, ref, hyp in zip(batch["path"], batch["resps"], resps_list):
                relpath = path.relative_to(cfg.data_root)
                hyp_path = (log_dir / "hyp" / relpath).with_suffix(".wav")
                ref_path = (log_dir / "ref" / relpath).with_suffix(".wav")
                hyp_path.parent.mkdir(parents=True, exist_ok=True)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                qnt.decode_to_file(ref, ref_path)
                if len(hyp) > 0:
                    qnt.decode_to_file(hyp, hyp_path)
        generate()
        torch.cuda.empty_cache()

        qnt.unload_model()

        stats = {k: sum(v) / len(v) for k, v in stats.items()}
        stats["global_step"] = engines.global_step
        stats["name"] = name
        _logger.info(f"Eval: {stats}.")

        _logger.info(f"{json.dumps(stats)}.")
        return stats

    def eval_fn(engines):
        # run_eval(engines, "subtrain", subtrain_dl)
        return run_eval(engines, "val", val_dl)

    trainer.train(
        engines_loader=load_engines,
        train_dl=train_dl,
        train_feeder=train_feeder,
        eval_fn=eval_fn,
        use_tb=True
    )


if __name__ == "__main__":
    main()
