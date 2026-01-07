"""Top-level module for feisty"""
from . import offline_driver as offline_driver_mod, testcase
from .core import settings
from .core.interface import feisty_instance_type
from .offline_driver import (
    _offline_driver,
    config_and_run_from_dataset,
    config_and_run_from_netcdf,
    config_and_run_from_yaml,
    config_and_run_testcase,
)
