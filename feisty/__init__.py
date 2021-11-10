"""Top-level module for feisty"""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = '0.0.0'

from . import driver, settings
from .core import feisty_instance_type
