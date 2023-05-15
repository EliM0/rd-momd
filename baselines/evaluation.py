import utils
from baselines import FictitiousPlay, OnlineMirrorDescent, MunchausenDeepMirrorDescent
from absl import flags, app
from typing import Sequence

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './config.yml', 'Config file with the algorithms parameters.')

def main(argv: Sequence[str]):
    cfg = utils.read_config(FLAGS.config)

    # Crowd motion environment
    game_name = 'mfg_crowd_modelling_2d'

    fp = FictitiousPlay(game_name, cfg)
    omd = OnlineMirrorDescent(game_name, cfg)
    momd = MunchausenDeepMirrorDescent(game_name, cfg)

    exp_fp = fp.solve()
    exp_omd = omd.solve()
    exp_momd = momd.solve()

    utils.plot_explotability(exp_fp, exp_omd, exp_momd, None, 'Crowd Motion Environment')


if __name__ == "__main__":
   app.run(main)