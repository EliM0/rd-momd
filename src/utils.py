import yaml
import matplotlib.pyplot as plt
from datetime import datetime

def read_config(config_file):
    with open(config_file, mode='r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        return cfg
