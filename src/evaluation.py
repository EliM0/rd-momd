import utils
from baselines import OnlineMirrorDescent, MunchausenDeepMirrorDescent, RNNMunchausenDeepMirrorDescent, AverageFictitiousPlay
from absl import flags, app
from typing import Sequence

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './configs/crowd_config.yml', 'Config file with the algorithms parameters.')

def main(argv: Sequence[str]):
    cfg = utils.read_config(FLAGS.config)

    game_name = cfg['game_name']
    game_settings = cfg['game_settings']

    afp = AverageFictitiousPlay(game_name, game_settings, cfg)
    # omd = OnlineMirrorDescent(game_name, game_settings, cfg)
    momd = MunchausenDeepMirrorDescent(game_name, game_settings, cfg)
    rnn = RNNMunchausenDeepMirrorDescent(game_name, game_settings, cfg)

    exp_afp = afp.solve()
    # exp_omd = omd.solve()
    exp_momd = momd.solve()
    exp_rnn = rnn.solve()

    # utils.plot_explotability(exp_afp, exp_omd, exp_momd, exp_rnn, cfg['plot_title'], cfg['results_dir'])
    utils.plot_explotability(exp_afp, None, exp_momd, exp_rnn, cfg['plot_title'], cfg['results_dir'])

if __name__ == "__main__":
   app.run(main)