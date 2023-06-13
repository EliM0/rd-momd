from algorithms import obs_dmomd, obs_dqn, rnn_momd, obs_afp
from rl import rl_obs_environment

from open_spiel.python.mfg.algorithms import munchausen_deep_mirror_descent, average_network_fictitious_play
from open_spiel.python.mfg.algorithms import nash_conv, distribution
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import training
from open_spiel.python.jax import dqn
from open_spiel.python import rl_environment
from open_spiel.python import policy
from torch.utils.tensorboard import SummaryWriter

class MFGAlgorithm:
    def __init__(self, game_name, game_setting, cfg):
        self.num_iterations = cfg['iterations']
        self.cfg = cfg

        self.game = factory.create_game_with_setting(game_name, game_setting)
        self.num_players = self.game.num_players()

        uniform_policy = policy.UniformRandomPolicy(self.game)
        self.uniform_dist = distribution.DistributionPolicy(self.game, uniform_policy)

    def solve(self):
        expls = []
        for it in range(1, self.num_iterations + 1):
            self.md.iteration()

            exploitability = nash_conv.NashConv(self.game, self.md.policy).nash_conv()
            expls.append(exploitability)
                        
            print("Iteration", it, "Exploitability: ", exploitability)
            
            self.writer.add_scalar('Exploitability', exploitability, it)

        return expls


class MunchausenDeepMirrorDescent(MFGAlgorithm):
    def __init__(self, game_name, game_setting, cfg):
        super().__init__(game_name, game_setting, cfg)

        self.writer = SummaryWriter('runs/D-MOMD')

        envs = [rl_environment.Environment(self.game, mfg_distribution=self.uniform_dist, mfg_population=p,
                observation_type=rl_environment.ObservationType.OBSERVATION) for p in range(self.num_players)]

        env = envs[0]
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        kwargs = {
            "alpha": cfg['munchausen_omd']['alpha'],
            "batch_size": cfg['munchausen_omd']['batch_size'],
            "discount_factor": cfg['munchausen_omd']['discount_factor'],
            "epsilon_decay_duration": cfg['munchausen_omd']['epsilon_decay_duration'],
            "epsilon_end": cfg['munchausen_omd']['epsilon_end'],
            "epsilon_power": cfg['munchausen_omd']['epsilon_power'],
            "epsilon_start": cfg['munchausen_omd']['epsilon_start'],
            "gradient_clipping": cfg['munchausen_omd']['gradient_clipping'],
            "hidden_layers_sizes": [int(l) for l in cfg['munchausen_omd']['hidden_layers_sizes']],
            "huber_loss_parameter": cfg['munchausen_omd']['huber_loss_parameter'],
            "learn_every": cfg['munchausen_omd']['learn_every'],
            "learning_rate": cfg['munchausen_omd']['learning_rate'],
            "loss": cfg['munchausen_omd']['loss'],
            "min_buffer_size_to_learn": cfg['munchausen_omd']['min_buffer_size_to_learn'],
            "optimizer": cfg['munchausen_omd']['optimizer'],
            "replay_buffer_capacity": cfg['munchausen_omd']['replay_buffer_capacity'],
            "reset_replay_buffer_on_update": cfg['munchausen_omd']['reset_replay_buffer_on_update'],
            "seed": cfg['munchausen_omd']['seed'],
            "tau": cfg['munchausen_omd']['tau'],
            "update_target_network_every": cfg['munchausen_omd']['update_target_network_every'],
            "with_munchausen": cfg['munchausen_omd']['with_munchausen']
        }
        agents = [munchausen_deep_mirror_descent.MunchausenDQN(p, info_state_size, num_actions, **kwargs)
                for p in range(self.num_players)]

        num_episodes_per_iteration = cfg['munchausen_omd']['num_episodes_per_iteration']
        self.md = munchausen_deep_mirror_descent.DeepOnlineMirrorDescent(self.game, envs, agents, eval_every=cfg['munchausen_omd']['eval_every'],
                                                                        num_episodes_per_iteration=num_episodes_per_iteration)

class ObsMunchausenDeepMirrorDescent(MFGAlgorithm):
    def __init__(self, game_name, game_setting, cfg, observability=False):
        super().__init__(game_name, game_setting, cfg)

        logdir = 'runs/PO-DMOMD' if observability else 'runs/FO-DMOMD'
        self.writer = SummaryWriter(logdir)

        envs = [rl_obs_environment.ObsEnvironment(self.game, mfg_distribution=self.uniform_dist, mfg_population=p,
                observation_type=rl_environment.ObservationType.OBSERVATION, partial_obs=observability) for p in range(self.num_players)]

        env = envs[0]
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]
        distribution_size = env.observation_spec()["distribution"][0]

        kwargs = {
            "alpha": cfg['munchausen_omd']['alpha'],
            "batch_size": cfg['munchausen_omd']['batch_size'],
            "discount_factor": cfg['munchausen_omd']['discount_factor'],
            "epsilon_decay_duration": cfg['munchausen_omd']['epsilon_decay_duration'],
            "epsilon_end": cfg['munchausen_omd']['epsilon_end'],
            "epsilon_power": cfg['munchausen_omd']['epsilon_power'],
            "epsilon_start": cfg['munchausen_omd']['epsilon_start'],
            "gradient_clipping": cfg['munchausen_omd']['gradient_clipping'],
            "hidden_layers_sizes": [int(l) for l in cfg['munchausen_omd']['hidden_layers_sizes']],
            "huber_loss_parameter": cfg['munchausen_omd']['huber_loss_parameter'],
            "learn_every": cfg['munchausen_omd']['learn_every'],
            "learning_rate": cfg['munchausen_omd']['learning_rate'],
            "loss": cfg['munchausen_omd']['loss'],
            "min_buffer_size_to_learn": cfg['munchausen_omd']['min_buffer_size_to_learn'],
            "optimizer": cfg['munchausen_omd']['optimizer'],
            "replay_buffer_capacity": cfg['munchausen_omd']['replay_buffer_capacity'],
            "reset_replay_buffer_on_update": cfg['munchausen_omd']['reset_replay_buffer_on_update'],
            "seed": cfg['munchausen_omd']['seed'],
            "tau": cfg['munchausen_omd']['tau'],
            "update_target_network_every": cfg['munchausen_omd']['update_target_network_every'],
            "with_munchausen": cfg['munchausen_omd']['with_munchausen'],
            "partial_obs": observability
        }
        agents = [obs_dmomd.ObsMunchausenDQN(p, info_state_size + distribution_size, num_actions, **kwargs)
                for p in range(self.num_players)]

        num_episodes_per_iteration = cfg['munchausen_omd']['num_episodes_per_iteration']
        self.md = obs_dmomd.ObsDeepOnlineMirrorDescent(self.game, envs, agents, eval_every=cfg['munchausen_omd']['eval_every'],
                                                                        num_episodes_per_iteration=num_episodes_per_iteration)


class AverageFictitiousPlay(MFGAlgorithm):
    def __init__(self, game_name, game_setting, cfg):
        super().__init__(game_name, game_setting, cfg)

        self.writer = SummaryWriter('runs/D-AFP')

        self.envs = [rl_environment.Environment(self.game, mfg_distribution=self.uniform_dist, mfg_population=p) for p in range(self.num_players)]

        env = self.envs[0]
        info_state_size = env.observation_spec()['info_state'][0]
        num_actions = env.action_spec()['num_actions']

        kwargs_dqn = {
            'batch_size': cfg['average_fp']['batch_size'],
            'discount_factor': cfg['average_fp']['discount_factor'],
            'epsilon_decay_duration': cfg['average_fp']['epsilon_decay_duration'],
            'epsilon_end': cfg['average_fp']['epsilon_end'],
            'epsilon_start': cfg['average_fp']['epsilon_start'],
            'gradient_clipping': cfg['average_fp']['gradient_clipping'],
            'hidden_layers_sizes': [int(l) for l in cfg['average_fp']['hidden_layers_sizes']],
            'huber_loss_parameter': cfg['average_fp']['huber_loss_parameter'],
            'learn_every': cfg['average_fp']['learn_every'],
            'learning_rate': cfg['average_fp']['learning_rate'],
            'loss_str': cfg['average_fp']['loss'],
            'min_buffer_size_to_learn': cfg['average_fp']['min_buffer_size_to_learn'],
            'optimizer_str': cfg['average_fp']['optimizer'],
            'replay_buffer_capacity': cfg['average_fp']['replay_buffer_capacity'],
            'seed': cfg['average_fp']['seed'],
            'update_target_network_every': cfg['average_fp']['update_target_network_every']
        }
        self.br_rl_agents = [dqn.DQN(p, info_state_size, num_actions, **kwargs_dqn) for p in range(self.num_players)]

        num_training_steps_per_iteration = (cfg['average_fp']['avg_pol_num_training_steps_per_iteration'])

        kwargs_avg = {
            'batch_size': cfg['average_fp']['avg_pol_batch_size'],
            'hidden_layers_sizes': [
                int(l) for l in cfg['average_fp']['avg_pol_hidden_layers_sizes']
            ],
            'reservoir_buffer_capacity': cfg['average_fp']['avg_pol_reservoir_buffer_capacity'],
            'learning_rate': cfg['average_fp']['avg_pol_learning_rate'],
            'min_buffer_size_to_learn':cfg['average_fp']['avg_pol_min_buffer_size_to_learn'],
            'optimizer_str': cfg['average_fp']['avg_pol_optimizer'],
            'gradient_clipping': cfg['average_fp']['avg_gradient_clipping'],
            'seed': cfg['average_fp']['seed'],
            'tau': cfg['average_fp']['avg_pol_tau']
        }
        self.fp = average_network_fictitious_play.AverageNetworkFictitiousPlay(self.game, self.envs, self.br_rl_agents, 
            cfg['average_fp']['avg_pol_num_episodes_per_iteration'], num_training_steps_per_iteration, eval_every=cfg['average_fp']['eval_every'], **kwargs_avg)

    def solve(self):
        expls = []
        
        for it in range(1, self.num_iterations+1):
            training.run_episodes(self.envs, self.br_rl_agents, num_episodes=self.cfg['average_fp']['num_dqn_episodes_per_iteration'], is_evaluation=False)
            self.fp.iteration()

            exploitability = nash_conv.NashConv(self.game, self.fp.policy).nash_conv()
            expls.append(exploitability)
            
            print("Iteration", it, "Exploitability: ", exploitability)

            self.writer.add_scalar('Exploitability', exploitability, it)

        return expls

class ObsAverageFictitiousPlay(MFGAlgorithm):
    def __init__(self, game_name, game_setting, cfg, observability=False):
        super().__init__(game_name, game_setting, cfg)

        logdir = 'runs/PO-DAFP' if observability else 'runs/FO-DAFP'
        self.writer = SummaryWriter(logdir)

        self.envs = [rl_obs_environment.ObsEnvironment(self.game, mfg_distribution=self.uniform_dist, mfg_population=p, partial_obs=observability) for p in range(self.num_players)]

        env = self.envs[0]
        info_state_size = env.observation_spec()['info_state'][0]
        num_actions = env.action_spec()['num_actions']
        distribution_size = env.observation_spec()['distribution'][0]

        kwargs_dqn = {
            'batch_size': cfg['average_fp']['batch_size'],
            'discount_factor': cfg['average_fp']['discount_factor'],
            'epsilon_decay_duration': cfg['average_fp']['epsilon_decay_duration'],
            'epsilon_end': cfg['average_fp']['epsilon_end'],
            'epsilon_start': cfg['average_fp']['epsilon_start'],
            'gradient_clipping': cfg['average_fp']['gradient_clipping'],
            'hidden_layers_sizes': [int(l) for l in cfg['average_fp']['hidden_layers_sizes']],
            'huber_loss_parameter': cfg['average_fp']['huber_loss_parameter'],
            'learn_every': cfg['average_fp']['learn_every'],
            'learning_rate': cfg['average_fp']['learning_rate'],
            'loss_str': cfg['average_fp']['loss'],
            'min_buffer_size_to_learn': cfg['average_fp']['min_buffer_size_to_learn'],
            'optimizer_str': cfg['average_fp']['optimizer'],
            'replay_buffer_capacity': cfg['average_fp']['replay_buffer_capacity'],
            'seed': cfg['average_fp']['seed'],
            'update_target_network_every': cfg['average_fp']['update_target_network_every'],
            'partial_obs': observability
        }
        self.br_rl_agents = [obs_dqn.ObsDQN(p, info_state_size+distribution_size, num_actions, **kwargs_dqn) for p in range(self.num_players)]

        num_training_steps_per_iteration = (cfg['average_fp']['avg_pol_num_training_steps_per_iteration'])

        kwargs_avg = {
            'batch_size': cfg['average_fp']['avg_pol_batch_size'],
            'hidden_layers_sizes': [
                int(l) for l in cfg['average_fp']['avg_pol_hidden_layers_sizes']
            ],
            'reservoir_buffer_capacity': cfg['average_fp']['avg_pol_reservoir_buffer_capacity'],
            'learning_rate': cfg['average_fp']['avg_pol_learning_rate'],
            'min_buffer_size_to_learn':cfg['average_fp']['avg_pol_min_buffer_size_to_learn'],
            'optimizer_str': cfg['average_fp']['avg_pol_optimizer'],
            'gradient_clipping': cfg['average_fp']['avg_gradient_clipping'],
            'seed': cfg['average_fp']['seed'],
            'tau': cfg['average_fp']['avg_pol_tau']
        }
        self.fp = obs_afp.ObsAverageNetworkFictitiousPlay(self.game, self.envs, self.br_rl_agents, 
            cfg['average_fp']['avg_pol_num_episodes_per_iteration'], num_training_steps_per_iteration, eval_every=cfg['average_fp']['eval_every'], **kwargs_avg)

    def solve(self):
        expls = []
        
        for it in range(1, self.num_iterations+1):
            training.run_episodes(self.envs, self.br_rl_agents, num_episodes=self.cfg['average_fp']['num_dqn_episodes_per_iteration'], is_evaluation=False)
            self.fp.iteration()

            exploitability = nash_conv.NashConv(self.game, self.fp.policy).nash_conv()
            expls.append(exploitability)
            
            print("Iteration", it, "Exploitability: ", exploitability)

            self.writer.add_scalar('Exploitability', exploitability, it)

        return expls


class RNNMunchausenDeepMirrorDescent(MFGAlgorithm):
    def __init__(self, game_name, game_setting, cfg):
        super().__init__(game_name, game_setting, cfg)

        self.writer = SummaryWriter('runs/RNN-MOMD')

        envs = [rl_environment.Environment(self.game, mfg_distribution=self.uniform_dist, mfg_population=p,
                observation_type=rl_environment.ObservationType.OBSERVATION) for p in range(self.num_players)]

        env = envs[0]
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        kwargs = {
            "alpha": cfg['rnn_munchausen_omd']['alpha'],
            "batch_size": cfg['rnn_munchausen_omd']['batch_size'],
            "discount_factor": cfg['rnn_munchausen_omd']['discount_factor'],
            "epsilon_decay_duration": cfg['rnn_munchausen_omd']['epsilon_decay_duration'],
            "epsilon_end": cfg['rnn_munchausen_omd']['epsilon_end'],
            "epsilon_power": cfg['rnn_munchausen_omd']['epsilon_power'],
            "epsilon_start": cfg['rnn_munchausen_omd']['epsilon_start'],
            "gradient_clipping": cfg['rnn_munchausen_omd']['gradient_clipping'],
            "hidden_layers_sizes": [int(l) for l in cfg['rnn_munchausen_omd']['hidden_layers_sizes']],
            "huber_loss_parameter": cfg['rnn_munchausen_omd']['huber_loss_parameter'],
            "learn_every": cfg['rnn_munchausen_omd']['learn_every'],
            "learning_rate": cfg['rnn_munchausen_omd']['learning_rate'],
            "loss": cfg['rnn_munchausen_omd']['loss'],
            "min_buffer_size_to_learn": cfg['rnn_munchausen_omd']['min_buffer_size_to_learn'],
            "optimizer": cfg['rnn_munchausen_omd']['optimizer'],
            "replay_buffer_capacity": cfg['rnn_munchausen_omd']['replay_buffer_capacity'],
            "reset_replay_buffer_on_update": cfg['rnn_munchausen_omd']['reset_replay_buffer_on_update'],
            "seed": cfg['rnn_munchausen_omd']['seed'],
            "tau": cfg['rnn_munchausen_omd']['tau'],
            "update_target_network_every": cfg['rnn_munchausen_omd']['update_target_network_every'],
            "with_munchausen": cfg['rnn_munchausen_omd']['with_munchausen']
        }
        agents = [rnn_momd.RNNMunchausenDQN(p, info_state_size, num_actions, **kwargs)
                for p in range(self.num_players)]

        num_episodes_per_iteration = cfg['rnn_munchausen_omd']['num_episodes_per_iteration']
        self.md = rnn_momd.RNNDeepOnlineMirrorDescent(self.game, envs, agents, eval_every=cfg['rnn_munchausen_omd']['eval_every'],
                                                                        num_episodes_per_iteration=num_episodes_per_iteration)
