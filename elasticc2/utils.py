#!/usr/bin/env python3
# License: BSD-3-Clause

from pathlib import Path

import yaml


def load_config():
    current_dir = Path(__file__)
    config_dir = current_dir.parents[1]
    config_file = config_dir / "config.yaml"

    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    classes = config["classes"]

    classes_inv = {v: k for k, v in classes.items()}
    config["classes_inv"] = classes_inv

    config["all"] = [val for val in config["classes_inv"].values()]

    for key, val in config["classes_inv"].items():
        config[key] = [val]

    return config
