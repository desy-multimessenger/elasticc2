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

    config["classes_inv"]["all"] = [
        2222,
        2223,
        2224,
        2225,
        2226,
        2232,
        2233,
        2234,
        2235,
        2242,
        2243,
        2244,
        2245,
        2246,
        2322,
        2323,
        2324,
        2325,
        2332,
    ]

    return config
