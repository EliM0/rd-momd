import utils
from baselines import FictitiousPlay, OnlineMirrorDescent, MunchausenDeepMirrorDescent, RNNMunchausenDeepMirrorDescent
from absl import flags, app
from typing import Sequence

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')

def main(argv: Sequence[str]):
    cfg = utils.read_config(FLAGS.config)

    game_name = cfg['game_name']

    fp = FictitiousPlay(game_name, cfg)
    omd = OnlineMirrorDescent(game_name, cfg)
    momd = MunchausenDeepMirrorDescent(game_name, cfg)
    rnn = RNNMunchausenDeepMirrorDescent(game_name, cfg)

    exp_fp = fp.solve()
    exp_omd = omd.solve()
    exp_momd = momd.solve()
    exp_rnn = rnn.solve()

    utils.plot_explotability(exp_fp, exp_omd, exp_momd, exp_rnn, cfg['plot_title'])

if __name__ == "__main__":
   app.run(main)