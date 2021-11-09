import os

import numpy as np

path_to_here = os.path.dirname(os.path.realpath(__file__))


def get_all_prey_and_preference(settings_dict):
    """inspect input and build list of all prey for each predator"""
    food_web_settings = settings_dict['food_web']
    all_prey = {}
    preference = {}
    for settings in food_web_settings:
        pred = settings['predator']
        if pred not in all_prey:
            all_prey[pred] = []
            preference[pred] = []
        all_prey[pred].append(settings['prey'])
        preference[pred].append(settings['encounter_parameters']['preference'])
    return all_prey, preference


def get_fish_func_type(settings_dict):
    """inspect input and generate dictionary of functional type of each group"""

    fish_func_type = {}
    for f in settings_dict['fish']:
        fish_func_type[f['name']] = f['functional_type']

    for b in settings_dict['benthic_prey']:
        fish_func_type[b['name']] = 'benthic_prey'

    for z in settings_dict['zooplankton']:
        fish_func_type[z['name']] = 'zooplankton'

    return fish_func_type
