"""
utils.py
Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-
import json

def load_config(config_file_path):
    with open(config_file_path, mode="r") as j_object:
        config_data = json.load(j_object)
    return config_data