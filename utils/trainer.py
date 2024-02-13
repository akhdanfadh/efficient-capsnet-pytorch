from abc import abstractmethod

import numpy as np
import torch
from torchvision.utils import make_grid

from .logger import TensorboardWriter
from .tools import MetricTracker, inf_loop


class BaseTrainer:
    def __init__(self, config, model, optimizer, loss_fn, metric_fns):
        self.config = config
        cfg_trainer = self.config["trainer"]

        # setup architecture
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns
        self.epochs = cfg_trainer["epochs"]
        self.start_epoch = 1

        # setup logger and visualization writer instance
        self.logger = config.get_logger(
            name="trainer", verbosity=config["trainer"]["verbosity"]
        )
        self.writer = TensorboardWriter(
            config.log_dir, self.logger, enabled=cfg_trainer["tensorboard"]
        )
        self.log_step = cfg_trainer["log_step"]
        self.save_period = cfg_trainer["save_period"]
        self.checkpoint_dir = config.save_dir

        # configuration to monitor model performance and save best
        self.monitor = cfg_trainer.get("monitor", "off")
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
            self.early_stop = cfg_trainer.get("early_stop", float("inf"))
            if self.early_stop < 0:
                self.early_stop = float("inf")

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor != "off":
                # check whether model performance improved or not, according to specified metric (mnt_metric)
                try:
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] < self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] > self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
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
                        "Validation performance didn't improve for {} epochs. Training stops.".format(
                            self.early_stop
                        )
                    )
                    break

            # save model checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        fname = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        torch.save(state, fname)

        self.logger.info("Checkpoint saved: {} ...".format(fname))
        if save_best:
            best_fname = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_fname)
            self.logger.info("Best checkpoint saved: {} ...".format(best_fname))


class MnistTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        device,
        train_data_loader,
        model,
        optimizer,
        loss_fn,
        metric_fns,
        scheduler=None,
        valid_data_loader=None,
        len_epoch=None,
    ):
        super().__init__(config, model, optimizer, loss_fn, metric_fns)
        self.config = config
        self.device = device
        self.scheduler = scheduler

        # data loader and metric configuration
        self.train_loader = train_data_loader
        self.valid_loader = valid_data_loader
        self.do_validation = self.valid_loader is not None
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in metric_fns]  # , writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in metric_fns]  # , writer=self.writer
        )

        # epoch configuration
        if len_epoch is None:  # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:  # iteration-based training
            self.train_loader = inf_loop(self.train_loader)
            self.len_epoch = len_epoch

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (input_images, targets) in enumerate(self.train_loader):
            input_images, targets = input_images.to(self.device), targets.to(
                self.device
            )
            self.optimizer.zero_grad()  # zero the gradients

            # forward and backward pass
            reconstructions, digit_caps_len = self.model(input_images, targets, mode='train')
            loss = self.loss_fn(targets, digit_caps_len, reconstructions, input_images)
            loss.backward()
            self.optimizer.step()

            y_pred = digit_caps_len.argmax(dim=1)  # get the index of the max probability
            y_true = targets.argmax(dim=1)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for metric in self.metric_fns:
                self.train_metrics.update(metric.__name__, metric(y_pred, y_true))
            
            if batch_idx % self.log_step == 0 or batch_idx == len(self.train_loader):
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                self.writer.add_image('input', make_grid(input_images.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.scheduler is not None:
            self.scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (input_images, targets) in enumerate(self.valid_loader):
                input_images, targets = input_images.to(self.device), targets.to(
                    self.device
                )
                reconstructions, digit_caps_len = self.model(input_images, targets, mode='valid')
                loss = self.loss_fn(
                    targets, digit_caps_len, reconstructions, input_images
                )

                y_pred = digit_caps_len.argmax(dim=1)  # get the index of the max probability
                y_true = targets.argmax(dim=1)
                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                for metric in self.metric_fns:
                    self.valid_metrics.update(metric.__name__, metric(y_pred, y_true))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_loader, "n_samples"):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
