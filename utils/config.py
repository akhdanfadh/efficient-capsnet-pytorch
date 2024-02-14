import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Type

from .logger import global_logger_setup
from .tools import read_yaml


class Config:
    """Main class object to handle configuration and initialize logging.

    Attributes:
        _config (dict): The configuration dictionary loaded in from_args function.
        _save_dir (Path): The directory for saving models.
        _log_dir (Path): The directory for saving logs.
    """

    def __init__(self, config: dict, run_id: str = None):
        """Initialize the Config object.

        Args:
            config (dict): The configuration dictionary loaded in from_args function.
            run_id (str, optional): The run id for the experiment. If not
                provided, the current timestamp is used. Defaults to None.
        """
        self._config = config

        # set experiment name and run id
        exp_name = config["main"]["name"]
        if not run_id:  # use timestamp as default
            run_id = datetime.now().strftime(r"%Y%m%d_%H%M%S")

        # set and create directory for saving log and model
        save_dir = Path(self.config["trainer"]["save_dir"])
        self._save_dir: Path = save_dir / "models" / exp_name / run_id
        self._log_dir: Path = save_dir / "log" / exp_name / run_id

        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # setup logging
        global_logger_setup(self.config["logger"], self.log_dir)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Initialize Config from cli arguments. Used in train and test.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Config: An instance of Config initialized with the values from the
                config file.

        Raises:
            AssertionError: If the config file does not exist.
        """
        args = args.parse_args()
        cfg_fname = Path(args.config)
        assert cfg_fname.exists(), f"Config file not found at {cfg_fname}"

        config = read_yaml(cfg_fname)
        return cls(config, args.run_id)

    def init_obj(self, cfg_name: str, module: Type[Any], *args, **kwargs) -> Any:
        """Initialize an object from a module using the configuration.

        This method finds a function handle with the name given as 'type' in the
        configuration file, and returns the instance initialized with
        corresponding arguments given.

        `function = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `function = module."cfg['name']['type']"(a, b=1)`

        Args:
            cfg_name (str): The name of the configuration to use.
            module (Type[Any]): The module to initialize the object from.

        Returns:
            Any: The initialized object.

        Raises:
            AssertionError: Keyword arguments should not changed the specified
                configuration file.
        """
        config = self.config[cfg_name]
        module_name = config["type"]
        module_args = dict(config["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name: str) -> Any:
        """Access items like ordinary dict."""
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
