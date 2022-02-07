import numpy as np
from itertools import combinations
from utils import save_json, check_if_dir
import os

CONFIG_PATH = "configs/"

def save_config(config, filename):
    check_if_dir(CONFIG_PATH)
    save_json(os.path.join(CONFIG_PATH, filename), config)

if __name__ == '__main__':
        
    batch_config = {
        "img_dir" : '/home/carl/back-of-the-router/out/',
        "name": "",
        "collection": "Back-of-the-Router",
        "symbol": "BOTR",
        "website": "homonculi.org/art",
        "max_dim": 500,
        "shots": 10,
        "steps_per_octave": list(range(1, 30)),
        "step_size": list(np.linspace(0.1, 1e-5, num=100)),
        "octaves_range": list(combinations(list(range(-2,3)), 2)),
        "octaves_scale": [1.1, 1.2, 1.3],
        "model_layers":  [3,6,7,8,9,10,11,12,13,14,15,16],
        "save_dir" : 'out/',
        # "save_dir": "/content/drive/MyDrive/deep-dream-homunculi/SZT-flaimes/",
        "batch_tag" : "back of the router deep dream images"
    }
    save_config(batch_config, "botr_batch.json")

    single_config = {
        "img": '/home/carl/back-of-the-router/out/23.png',
        "name": "bionic",
        "collection": "back-of-the-router",
        "symbol": "BOTR",
        "website": "homonculi.org/art",
        "max_dim": 500,
        "steps_per_octave": 10,
        "step_size":0.01,
        "octaves_range":(-3,3),
        "octaves_scale":1.3,
        "model_layers": [16],
        "save_dir": "out/"
    }
    save_config(single_config, "botr_single.json")
