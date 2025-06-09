import json


def save_config_to_file(config, plugins_config, path):
    data = {
        "config": config,
        "plugins_config": plugins_config,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("config", {}), data.get("plugins_config", {})

