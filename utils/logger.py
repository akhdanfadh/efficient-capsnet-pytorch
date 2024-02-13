import importlib
import logging
import logging.config
from datetime import datetime


class TensorboardWriter:
    """
    Handle visualization of training and validation statistics in Tensorboard.
    """

    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # retrieve vizualization writer
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, "
                    "but currently not installed on this machine. Please install "
                    "either TensorboardX with 'pip install tensorboardx', upgrade "
                    "PyTorch to version >= 1.1 to use 'torch.utils.tensorboard' "
                    "or turn off the option in the 'config.yaml' file."
                )
                logger.warning(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_fns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Set the step number for the visualization writer. If step is 0, the
        timer is reset. Otherwise, the time duration between the last step and
        the current step is recorded and added to the visualization.
        """
        self.mode = mode
        self.step = step

        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use, return add_data() methods of
        tensorboard with additional information (step, tag) added. Otherwise,
        return a blank function handle that does nothing.
        """
        if name in self.tb_writer_fns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = f"{self.mode}/{tag}"
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    f"Type object '{self.selected_module}' has no attribute '{name}'"
                )
            return attr


def setup_logging(cfg, log_dir):
    """
    Setup logging configuration
    """
    for _, handler in cfg["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(log_dir / handler["filename"])
    logging.config.dictConfig(cfg)
