import logging
from datetime import datetime
from pathlib import Path

from .logger import setup_logging
from .tools import read_yaml


class Config:
    def __init__(self, config, run_id=None):
        self._config = config

        # set experiment name and run id
        exp_name = config["name"]
        if not run_id:  # use timestamp as default
            run_id = datetime.now().strftime(r"%Y%m%d_%H%M%S")

        # set and create directory for saving log and model
        save_dir = Path(self.config["trainer"]["save_dir"])
        self._save_dir = save_dir / "models" / exp_name / run_id
        self._log_dir = save_dir / "log" / exp_name / run_id

        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # setup logging
        setup_logging(self.config['logger'], self.log_dir)
        self.log_levels = {
            0: logging.DEBUG,
            1: logging.INFO,
            2: logging.WARNING,
            3: logging.ERROR,
        }

    @classmethod
    def from_args(cls, args):
        """
        Initialize Config from command line arguments. Used in train and test.
        """
        if not isinstance(args, tuple):
            args = args.parse_args()

        cfg_fname = Path(args.config)
        assert cfg_fname.exists(), f"Config file not found at {cfg_fname}"
        config = read_yaml(cfg_fname)

        return cls(config)

    def init_obj(self, cfg_name, module, *args, **kwargs):
        """
        Initialize an object from module using the configuration.
        """
        config = self.config[cfg_name]
        module_name = config["type"]
        module_args = dict(config["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        """
        Access items like ordinary dict.
        """
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def save_dir(self):
        return self._save_dir

    def get_logger(self, name, verbosity=2):
        assert (
            verbosity in self.log_levels
        ), f"Verbosity option {verbosity} is out of range (0-3)"
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
