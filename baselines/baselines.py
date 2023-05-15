from open_spiel.python.mfg.algorithms import fictitious_play, mirror_descent, munchausen_deep_mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv, distribution
from open_spiel.python.mfg.games import factory
from open_spiel.python import rl_environment
from open_spiel.python import policy

class FictitiousPlay:
    def __init__(self, game_name, cfg):
        self.game = factory.create_game_with_setting(game_name, None)

        self.num_iterations = cfg['iterations']
        self.learning_rate = cfg['fictitious_play']['learning_rate'] 
        if self.learning_rate is None: self.learning_rate=  1 / cfg['iterations']
        
        self.fp = fictitious_play.FictitiousPlay(self.game)

    def solve(self):
        print("\n--------------------\nFictitious play\n--------------------")

        expls = []
        for it in range(self.num_iterations):
            self.fp.iteration(learning_rate=self.learning_rate)
            fp_policy = self.fp.get_policy()
            
            exploitability = nash_conv.NashConv(self.game, fp_policy).nash_conv()
            expls.append(exploitability)
        
            print("Iteration", it, "Exploitability: ", exploitability)

        return expls

class OnlineMirrorDescent:
    def __init__(self, game_name, cfg):
        self.game = factory.create_game_with_setting(game_name, None)
        
        self.num_iterations = cfg['iterations']
        self.learning_rate = cfg['online_mirror_descent']['learning_rate']

        self.md = mirror_descent.MirrorDescent(self.game, lr=self.learning_rate)

    def solve(self):
        print("\n--------------------\nOnline Mirror Descent\n--------------------")
        expls  = []
        for it in range(self.num_iterations):
            self.md.iteration()
            md_policy = self.md.get_policy()

            exploitability = nash_conv.NashConv(self.game, md_policy).nash_conv()
            expls.append(exploitability)

            print("Iteration", it, "Exploitability: ", exploitability)

        return expls

class MunchausenDeepMirrorDescent:
    def __init__(self, game_name, cfg):
        self.game = factory.create_game_with_setting(game_name, None)
        num_players = self.game.num_players()

        self.num_iterations = cfg['iterations']

        uniform_policy = policy.UniformRandomPolicy(self.game)
        uniform_dist = distribution.DistributionPolicy(self.game, uniform_policy)

        envs = [rl_environment.Environment(self.game, mfg_distribution=uniform_dist, mfg_population=p,
                observation_type=rl_environment.ObservationType.OBSERVATION) for p in range(num_players)]

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
                for p in range(num_players)]

        num_episodes_per_iteration = cfg['munchausen_omd']['num_episodes_per_iteration']
        self.md = munchausen_deep_mirror_descent.DeepOnlineMirrorDescent(self.game, envs, agents, eval_every=cfg['munchausen_omd']['eval_every'],
                                                                        num_episodes_per_iteration=num_episodes_per_iteration)

    def solve(self):
        print("\n--------------------\nMunchausen Deep Online Mirror Descent\n--------------------")

        expls = []
        for it in range(1, self.num_iterations + 1):
            self.md.iteration()

            exploitability = nash_conv.NashConv(self.game, self.md.policy).nash_conv()
            expls.append(exploitability)
                        
            print("Iteration", it, "Exploitability: ", exploitability)

        return expls
