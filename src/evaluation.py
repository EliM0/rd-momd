import utils
from baselines import *
from absl import flags, app
from typing import Sequence

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')
flags.DEFINE_string('algorithm', 'dmomd', 'Algorithm to execute.')
flags.DEFINE_string('logdir', None, 'Directory to save the logs')

def main(argv: Sequence[str]):
    cfg = utils.read_config(FLAGS.config)

    game_name = cfg['game_name']
    game_settings = cfg['game_settings']

    alg = None

    # Algorithms without population dependency
    if FLAGS.algorithm == 'dafp':
        logdir = 'runs/D-AFP' if not FLAGS.logdir else FLAGS.logdir
        alg = AverageFictitiousPlay(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'dmomd':
        logdir = 'runs/D-MOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = MunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'rnn-momd':
        logdir = 'runs/RD-MOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = RNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)

    # Algorithms with full observability
    elif FLAGS.algorithm == 'fo-dafp':
        logdir = 'runs/FO-DAFPP' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsAverageFictitiousPlay(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'fo-dmomd':
        logdir = 'runs/FO-DMOMDP' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'fo-1rd-momd':
        logdir = 'runs/FO-1RDMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'fo-trd-momd':
        logdir = 'runs/FO-TRDMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsTRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)

    # Algorithms with partial observability
    elif FLAGS.algorithm == 'po-dafp':
        logdir = 'runs/PO-DAFPP' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsAverageFictitiousPlay(game_name, game_settings, cfg, observability=True, logdir=logdir)
    elif FLAGS.algorithm == 'po-dmomd':
        logdir = 'runs/PO-DMOMDP' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsMunchausenDeepMirrorDescent(game_name, game_settings, cfg, observability=True, logdir=logdir)
    elif FLAGS.algorithm == 'po-1rd-momd':
        logdir = 'runs/PO-1RDMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, logdir=logdir)
    elif FLAGS.algorithm == 'po-trd-momd':
        logdir = 'runs/PO-TRDMOMD' if not FLAGS.logdir else FLAGS.logdir
        alg = ObsTRNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg, observability=True, logdir=logdir)
    else:
        print("usage: evaluation.py --config=<config_dir> --algorithm=['afp', 'dmomd', 'rnn-momd', 'fo-dmomd', 'po-dmomd', 'po-rnn-momd']")
        exit()

    alg.solve()

if __name__ == "__main__":
   app.run(main)