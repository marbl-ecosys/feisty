import os
import numpy as np

import feisty
import feisty.settings as settings

path_to_here = os.path.dirname(os.path.realpath(__file__))

settings_dict_def = settings.get_defaults()
model_settings = settings_dict_def["model_settings"]
food_web_settings = settings_dict_def["food_web"]
zoo_settings = settings_dict_def["zooplankton"]
fish_settings = settings_dict_def["fish"]
benthic_prey_settings = settings_dict_def["benthic_prey"]
reproduction_routing = settings_dict_def["reproduction_routing"]

for i in range(len(settings_dict_def["food_web"])):
    settings_dict_def["food_web"][i]["encounter_parameters"]["preference"] = np.random.rand()


fish_ic_data = 1e-5
benthic_prey_ic_data = 1e-4

n_zoo = len(settings_dict_def["zooplankton"])
n_fish = len(settings_dict_def["fish"])
n_benthic_prey = 1

NX = 10
NX_2 = 5
domain_dict = {
    "NX": NX,
    "depth_of_seafloor": np.concatenate((np.ones(NX_2) * 1500.0, np.ones(NX_2) * 15.0)),
}

F = feisty.feisty_instance_type(
    settings_dict=settings_dict_def,
    domain_dict=domain_dict,
    fish_ic_data=fish_ic_data,
    benthic_prey_ic_data=benthic_prey_ic_data,
)

# inspect input and build list of all prey for each predator
all_prey = {}
preference = {}
for settings in food_web_settings:
    pred = settings["predator"]
    if pred not in all_prey:
        all_prey[pred] = []
        preference[pred] = []
    all_prey[pred].append(settings["prey"])
    preference[pred].append(settings["encounter_parameters"]["preference"])

# inspect input and generate dictionary of functional type of each group
fish_func_type = {}
for f in fish_settings:
    fish_func_type[f["name"]] = f["functional_type"]
for b in settings_dict_def["benthic_prey"]:
    fish_func_type[b["name"]] = "benthic_prey"
for z in settings_dict_def["zooplankton"]:
    fish_func_type[z["name"]] = "zooplankton"


zoo_names = [z["name"] for z in settings_dict_def["zooplankton"]]

zoo_predators = []
for i, link in enumerate(food_web_settings):
    if link["prey"] in zoo_names:
        zoo_predators.append(link["predator"])
