import utils
from baselines import *
from absl import flags, app
from typing import Sequence

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')
flags.DEFINE_string('algorithm', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')

def main(argv: Sequence[str]):
    cfg = utils.read_config(FLAGS.config)

    game_name = cfg['game_name']
    game_settings = cfg['game_settings']

    alg = None

    if FLAGS.algorithm == 'dafp':
        alg = AverageFictitiousPlay(game_name, game_settings, cfg)
    elif FLAGS.algorithm == 'dmomd':
        alg = MunchausenDeepMirrorDescent(game_name, game_settings, cfg)
    elif FLAGS.algorithm == 'rnn-momd':
        alg = RNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg)

    elif FLAGS.algorithm == 'fo-dafp':
        alg = ObsAverageFictitiousPlay(game_name, game_settings, cfg)
    elif FLAGS.algorithm == 'fo-dmomd':
        alg = ObsMunchausenDeepMirrorDescent(game_name, game_settings, cfg)

    elif FLAGS.algorithm == 'po-dafp':
        alg = ObsAverageFictitiousPlay(game_name, game_settings, cfg, observability=True)
    elif FLAGS.algorithm == 'po-dmomd':
        alg = ObsMunchausenDeepMirrorDescent(game_name, game_settings, cfg, observability=True)
    elif FLAGS.algorithm == 'po-rnn-momd':
        raise NotImplementedError
    else:
        print("usage: evaluation.py --config=<config_dir> --algorithm=['afp', 'dmomd', 'rnn-momd', 'fo-dmomd', 'po-dmomd', 'po-rnn-momd']")
        exit()

    alg.solve()

if __name__ == "__main__":
   app.run(main)