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
                cfg[k][kk] = fix(vv)
        else:
            cfg[k] = fix(v)
    
    return cfg
    
    

def remove_defaults(d):
    to_remove = []
    for key, value in d.items():
        if value == "<default>":
            to_remove.append(key)
    
    for key in to_remove:
        del d[key]
    return d

def get_configs(filename, filepath="./configs"):

    file = Path(filepath) / filename
    with open(file) as fp:
        cfg = yaml.safe_load(fp)

    
    cfg = remove_defaults(cfg)
    cfg = fix_e(cfg)

    # cfg["training_arguments"]["dataloader_num_workers"] = cfg["num_proc"]

    training_args = cfg.pop("training_arguments")
    return cfg, training_args