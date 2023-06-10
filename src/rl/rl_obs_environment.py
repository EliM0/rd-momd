from open_spiel.python import rl_environment
from utils import get_partial_distribution

import pyspiel

SIMULTANEOUS_PLAYER_ID = pyspiel.PlayerId.SIMULTANEOUS

class ObsEnvironment(rl_environment.Environment):
  """Open Spiel reinforcement learning environment class."""

  def __init__(self,
               game,
               discount=1.0,
               chance_event_sampler=None,
               observation_type=None,
               include_full_state=False,
               mfg_distribution=None,
               mfg_population=None,
               enable_legality_check=False,
               partial_obs=False,
               **kwargs):

    super().__init__(game, discount, chance_event_sampler, observation_type, include_full_state, mfg_distribution, 
                     mfg_population, enable_legality_check, **kwargs)

    self.partial_obs = partial_obs

  def get_time_step(self):
    """Returns a `TimeStep` without updating the environment.

    Returns:
      A `TimeStep` namedtuple containing:
        observation: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        reward: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discount: list of discounts in the range [0, 1], or None if step_type is
          `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": [],
        "serialized_state": [],
        "distribution": []
    }

    rewards = []
    step_type = rl_environment.StepType.LAST if self._state.is_terminal() else rl_environment.StepType.MID
    self._should_reset = step_type == rl_environment.StepType.LAST

    cur_rewards = self._state.rewards()
    for player_id in range(self.num_players):
      rewards.append(cur_rewards[player_id])
      observations["info_state"].append(
          self._state.observation_tensor(player_id) if self._use_observation
          else self._state.information_state_tensor(player_id))
      
      if self.game.get_type().short_name == 'mfg_crowd_modelling_2d':
        distribution = self._state.distribution_tensor(player_id)
        dist = get_partial_distribution(self._game, observations["info_state"][-1], distribution) if self.partial_obs else distribution
        observations["distribution"].append(dist)

      observations["legal_actions"].append(self._state.legal_actions(player_id))
    observations["current_player"] = self._state.current_player()
    discounts = self._discounts
    if step_type == rl_environment.StepType.LAST:
      # When the game is in a terminal state set the discount to 0.
      discounts = [0. for _ in discounts]

    if self._include_full_state:
      observations["serialized_state"] = pyspiel.serialize_game_and_state(
          self._game, self._state)

    # For gym environments
    if hasattr(self._state, "last_info"):
      observations["info"] = self._state.last_info

    return rl_environment.TimeStep(
        observations=observations,
        rewards=rewards,
        discounts=discounts,
        step_type=step_type)

  def reset(self):
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
      A `TimeStep` namedtuple containing:
        observations: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        rewards: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discounts: list of discounts in the range [0, 1], or None if step_type
          is `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    self._should_reset = False
    if self._game.get_type(
    ).dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD and self._num_players > 1:
      self._state = self._game.new_initial_state_for_population(
          self._mfg_population)
    else:
      self._state = self._game.new_initial_state()
    self._sample_external_events()

    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": [],
        "serialized_state": [],
        "distribution": []
    }

    for player_id in range(self.num_players):
      observations["info_state"].append(
          self._state.observation_tensor(player_id) if self._use_observation
          else self._state.information_state_tensor(player_id))
      
      if self.game.get_type().short_name == 'mfg_crowd_modelling_2d':
        distribution = self._state.distribution_tensor(player_id)
        dist = get_partial_distribution(self._game, observations["info_state"][-1], distribution) if self.partial_obs else distribution
        observations["distribution"].append(dist)
        
      observations["legal_actions"].append(self._state.legal_actions(player_id))
    observations["current_player"] = self._state.current_player()

    if self._include_full_state:
      observations["serialized_state"] = pyspiel.serialize_game_and_state(
          self._game, self._state)

    return rl_environment.TimeStep(
        observations=observations,
        rewards=None,
        discounts=None,
        step_type=rl_environment.StepType.FIRST)

  def observation_spec(self):
    """Defines the observation per player provided by the environment.

    Each dict member will contain its expected structure and shape. E.g.: for
    Kuhn Poker {"info_state": (6,), "legal_actions": (2,), "current_player": (),
                "serialized_state": ()}

    Returns:
      A specification dict describing the observation fields and shapes.
    """
    dist_size = 9 if self.partial_obs else self._game.distribution_tensor_size()
    
    return dict(
        info_state=tuple([
            self._game.observation_tensor_size() if self._use_observation else
            self._game.information_state_tensor_size()
        ]),
        legal_actions=(self._game.num_distinct_actions(),),
        current_player=(),
        serialized_state=(),
        distribution=tuple([dist_size])
    )
