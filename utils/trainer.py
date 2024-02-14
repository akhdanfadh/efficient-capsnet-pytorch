from abc import abstractmethod
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .config import Config
from .logger import TensorboardWriter, get_logger
from .tools import MetricTracker


class BaseTrainer:
    """Custom base class for all trainers.

    This class provides basic utilities for training a model, including setting
    up the model architecture, initializing the optimizer and loss function, and
    providing logging and visualization utilities. The class also supports
    monitoring model performance and saving the best model.

    Attributes:
        config (Config): The configuration object.
        model (torch.nn.Module): The model architecture.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module | Callable): The loss function.
        metric_fns (list[torch.nn.Module | Callable]): A list of metric functions.
        n_epoch (int): The number of epochs to train the model.
        start_epoch (int): The starting epoch number, used for resuming training.
        early_stop (int): The number of epochs to wait before early stopping.
        logger (Logger): The logger instance.
        writer (TensorboardWriter): The visualization writer instance.
        log_step (int): The frequency of logging training information.
        save_period (int): The frequency of saving model checkpoints.
        checkpoint_dir (Path): The directory to save model checkpoints.
        monitor (str): The model performance monitoring mode.
        mnt_mode (str): The monitoring mode (min or max).
        mnt_best (float): The best monitored metric value.
    """

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module | Callable,
        metric_fns: list[torch.nn.Module | Callable],
    ):
        self.config = config
        cfg_trainer: dict = self.config["trainer"]

        # setup architecture
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.n_epoch: int = cfg_trainer["epochs"]
        self.start_epoch = 1

        # setup logger and visualization writer instance
        self.logger = get_logger(
            name="trainer", verbosity=config["trainer"]["verbosity"]
        )
        self.writer = TensorboardWriter(
            config.log_dir, self.logger, enabled=cfg_trainer["tensorboard"]
        )
        self.log_step: int = cfg_trainer["log_step"]
        self.save_period: int = cfg_trainer["save_period"]
        self.checkpoint_dir = config.save_dir

        # configuration to monitor model performance and save best
        self.monitor: str = cfg_trainer.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in [
                "min",
                "max",
            ], "Only support min and max monitor mode"
            self.mnt_best = float("inf") if self.mnt_mode == "min" else float("-inf")
            self.early_stop: int = cfg_trainer.get("early_stop", float("inf"))
            if self.early_stop < 0:
                self.early_stop = float("inf")

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """Abstract method for training the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing logged information for this epoch.

        Raises:
            NotImplementedError: This is an abstract method that should be
                implemented by subclasses.
        """
        raise NotImplementedError

    def train(self) -> None:
        """Full model training logic for a specified number of epochs.

        Raises:
            KeyError: If the specified metric for monitoring is not found in the log.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.n_epoch + 1):
            result: dict = self._train_epoch(epoch)

            # save logged information into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info("   {:15s}: {}".format(str(key), value))

            # monitor best performance and perform early stopping
            best = False
            if self.monitor != "off":
                try:  # check improvement on the specified monitor metric
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] < self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] > self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '%s' is not found. Model performance monitoring is disabled.",
                        self.mnt_metric,
                    )
                    self.monitor = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # early stopping
                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for %d epochs. Training stops.",
                        self.early_stop,
                    )
                    break

            # save model checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """Save the current model checkpoint.

        Args:
            epoch (int): The current epoch number.
            save_best (bool, optional): Whether to save this checkpoint as the
                best so far. Defaults to False.
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        fname = str(self.checkpoint_dir / f"ep{epoch}.pth")
        torch.save(state, fname)
        self.logger.info("Checkpoint saved: %s ...", fname)

        if save_best:  # save as the best yet
            best_fname = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_fname)
            self.logger.info("Best checkpoint saved: %s ...", best_fname)


class MnistTrainer(BaseTrainer):
    """Custom trainer for the MNIST dataset, validation included.

    Attributes:
        config (Config): Configuration object.
        device (torch.device): Device to run the model on.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module | Callable): Loss function.
        metric_fns (list[Any]): List of metric functions.
        train_data_loader (torch.utils.data.DataLoader): Training data loader.
        valid_data_loader (torch.utils.data.DataLoader, optional): Validation data
            loader. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional):
            Learning rate scheduler. Defaults to None.
        n_batch (int): Number of training steps (batches) in an epoch.
        train_metrics (MetricTracker): Training metric tracker.
        valid_metrics (MetricTracker): Validation metric tracker.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module | Callable,
        metric_fns: list[torch.nn.Module | Callable],
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        super().__init__(config, model, optimizer, criterion, metric_fns)
        self.config = config
        self.device = device
        self.lr_scheduler = lr_scheduler

        # data loader and metric configuration
        self.train_loader = train_data_loader
        self.valid_loader = valid_data_loader
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in metric_fns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in metric_fns], writer=self.writer
        )
        self.n_batch = len(self.train_loader)

    def _train_epoch(self, epoch):
        self.model.train()  # set the model to training mode
        self.train_metrics.reset()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # configure data and optimizer
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()  # zero the gradients

            # forward and backward pass
            out_images, out_labels = self.model(images, labels, mode="train")
            loss = self.criterion(images, labels, out_images, out_labels)
            loss.backward()
            self.optimizer.step()

            # get the index of the maximum value
            label = labels.argmax(dim=1)  # from one-hot
            out_label = out_labels.argmax(dim=1)  # from probability

            # update tracker
            self.writer.set_step((epoch - 1) * self.n_batch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for metric in self.metric_fns:
                self.train_metrics.update(metric.__name__, metric(label, out_label))

            # log training information
            if batch_idx % self.log_step == 0 or batch_idx == len(self.train_loader):
                self.logger.debug(self._progress(epoch, batch_idx, loss.item()))
                self.writer.add_image(
                    "input", make_grid(images.cpu(), nrow=8, normalize=True)
                )

        train_log = self.train_metrics.result()

        # validate the model, if provided
        if self.valid_loader is not None:
            val_log = self._valid_epoch(epoch)
            train_log.update(**{"val_" + k: v for k, v in val_log.items()})
        
        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return train_log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valid_loader):
                # configure data and optimizer
                images = images.to(self.device)
                labels = labels.to(self.device)

                # # forward and backward pass
                out_images, out_labels = self.model(images, labels, mode="eval")
                loss = self.criterion(images, labels, out_images, out_labels)

                # get the index of the maximum value
                label = labels.argmax(dim=1)  # from one-hot
                out_label = out_labels.argmax(dim=1)  # from probability

                # update tracker
                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                for metric in self.metric_fns:
                    self.valid_metrics.update(metric.__name__, metric(label, out_label))

        # add histogram of model parameters to the tensorboard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, epoch_idx, batch_idx, loss_value):
        if hasattr(self.train_loader, "n_samples"):
            current = batch_idx * self.train_loader.batch_size
            samples = self.train_loader.n_samples
        else:
            current = batch_idx
            samples = self.n_batch

        base = "Train Epoch: {:>{}}/{} [{:>{}}/{} ({:3.0f}%)], Loss: {:.6f}"
        return base.format(
            epoch_idx,
            len(str(self.n_epoch)),
            self.n_epoch,
            current,
            len(str(samples)),
            samples,
            100 * current / samples,
            loss_value,
        )
