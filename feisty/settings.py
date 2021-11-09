import os

import yaml

path_to_here = os.path.dirname(os.path.realpath(__file__))


def get_defaults():
    """return default settings"""
    with open(f'{path_to_here}/default_settings.yml') as fid:
        return yaml.safe_load(fid)
