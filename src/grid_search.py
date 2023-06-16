from baselines import ObsTRNNMunchausenDeepMirrorDescent
from sklearn.model_selection import ParameterGrid
import numpy as np
import yaml
import utils

def main():
    cfg = utils.read_config('./configs/crowd_config.yml')

    game_name = cfg['game_name']
    game_settings = cfg['game_settings']

    param_grid = {'epsilon_decay_duration': [300000, 400000, 500000, 700000], 'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5]}

    best_score = np.inf
    best_grid = param_grid

    i = 0
    for g in ParameterGrid(param_grid):
        logdir = 'grid_search/experiment{}'.format(i)
        cfg['trnn_momd']['epsilon_decay_duration'] = g['epsilon_decay_duration']
        cfg['trnn_momd']['epsilon_start'] = g['epsilon']
        cfg['trnn_momd']['epsilon_end'] = g['epsilon']

        alg = ObsTRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, observability=True, logdir=logdir)
        expls = alg.solve()
        
        if expls[-1] > best_score:
            best_score = expls[-1]
            best_grid = g

        i+=1

        with open('{}/parameters.yml'.format(logdir), 'w') as file:
            yaml.dump(g, file)
    
    with open('grid_search/best_parameters.yml'.format(logdir), 'w') as file:
            yaml.dump(best_grid, file)

if __name__ == "__main__":
   main()