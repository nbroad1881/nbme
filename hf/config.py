import re
import yaml
from pathlib import Path

def fix_e(cfg):

    def fix(value):
        pattern = r"\d+e\-\d+"
        if re.search(pattern, value):
            return eval(value)
        return value


    for k, v in cfg.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str):
                    cfg[k][kk] = fix(vv)
        elif isinstance(v, str):
            cfg[k] = fix(v)
    
    return cfg
    
    

def remove_defaults(cfg):
    to_remove = []
    args = cfg["training_arguments"]
    for key, value in args.items():
        if value == "<default>":
            to_remove.append(key)
    
    for key in to_remove:
        del args[key]

def get_configs(filename, filepath="./configs"):

    file = Path(filepath) / filename
    with open(file) as fp:
        cfg = yaml.safe_load(fp)

    
    remove_defaults(cfg)
    cfg = fix_e(cfg)

    # cfg["training_arguments"]["dataloader_num_workers"] = cfg["num_proc"]

    training_args = cfg.pop("training_arguments")
    return cfg, training_args