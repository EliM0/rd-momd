import yaml
import numpy as np

def read_config(config_file):
    with open(config_file, mode='r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        return cfg

def get_partial_distribution(game, info_state, distribution):
    size = int(np.sqrt(game.distribution_tensor_size()))

    x = info_state[:size].index(1)
    y = info_state[size:2*size].index(1)

    return [distribution[i*size + j] if i >= 0 and i < size and j >= 0 and j < size else 0 
            for i in range(x-1, x+2) for j in range(y-1, y+2)]
