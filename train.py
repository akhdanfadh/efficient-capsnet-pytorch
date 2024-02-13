import argparse

import numpy as np
import torch

import model.losses as module_loss
import model.metrics as module_metric
import model.model as module_arch
import utils.data_loader as module_data
from utils.config import Config
from utils.trainer import MnistTrainer


def main(config):
    logger = config.get_logger("train")

    # set seed
    SEED = config["seed"]
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    logger.info(f"Seed set to {SEED}")

    # set device
    device = torch.device(
        "cuda" if config["cuda"] and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()
    logger.info("Data loader set up")

    # build model architecture
    model = config.init_obj("arch", module_arch)
    model = model.to(device)
    logger.info("Model set up")

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # get function handles of loss and metrics
    loss = config.init_obj("loss", module_loss)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    logger.info("Loss and metrics set up")

    trainer = MnistTrainer(
        config=config,
        device=device,
        train_data_loader=data_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        metric_fns=metrics,
        scheduler=lr_scheduler,
        valid_data_loader=valid_data_loader,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Efficient CapsNet Training")
    args.add_argument(
        "-c", "--config", type=str, required=True, help="path to config file"
    )
    config = Config.from_args(args)
    main(config)
