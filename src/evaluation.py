import utils
from baselines import *
from absl import flags, app
from typing import Sequence

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')
flags.DEFINE_string('algorithm', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')
flags.DEFINE_string('logdir', None, 'Directory to save the logs')

def main(argv: Sequence[str]):
    cfg = utils.read_config(FLAGS.config)

    game_name = cfg['game_name']
    game_settings = cfg['game_settings']

    alg = None

    if FLAGS.algorithm == 'dafp':
        logdir = 'runs/D-AFP' if not FLAGS.logdir else FLAGS.logdir
        alg = AverageFictitiousPlay(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'dmomd':
        logdir = 'runs/D-MOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = MunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'rnn-momd':
        logdir = 'runs/RNN-MOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = RNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)

    elif FLAGS.algorithm == 'fo-dafp':
        logdir = 'runs/FO-DAFP' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsAverageFictitiousPlay(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'fo-dmomd':
        logdir = 'runs/FO-DMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'fo-rnn-momd':
        logdir = 'runs/FO-RNNMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'fo-trnn-momd':
        logdir = 'runs/FO-TRNNMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsTRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)

    elif FLAGS.algorithm == 'po-dafp':
        logdir = 'runs/PO-DAFP' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsAverageFictitiousPlay(game_name, game_settings, cfg, observability=True, logdir=logdir)
    elif FLAGS.algorithm == 'po-dmomd':
        logdir = 'runs/PO-DMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsMunchausenDeepMirrorDescent(game_name, game_settings, cfg, observability=True, logdir=logdir)
    elif FLAGS.algorithm == 'po-rnn-momd':
        logdir = 'runs/PO-RNNMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'po-trnn-momd':
        logdir = 'runs/PO-TRNNMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsTRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, observability=True, logdir=logdir)
    else:
        print("usage: evaluation.py --config=<config_dir> --algorithm=['afp', 'dmomd', 'rnn-momd', 'fo-dmomd', 'po-dmomd', 'po-rnn-momd']")
        exit()

    alg.solve()

if __name__ == "__main__":
   app.run(main)