import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # The GPU id to use, usually either "0" or "1"

import numpy as np

from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
from absl import flags

BASE_PATH = './tmp/dope_run'
GAME = 'Asterix'
LOG_PATH = os.path.join(BASE_PATH, 'basic_agent', GAME)

class BasicAgent(object):
    """This agent randomly selects an action and sticks to it. It will change
      actions with probability switch_prob."""
    def __init__(self, sess, num_actions, switch_prob=0.1):
        self._sess = sess
        self._num_actions = num_actions
        self._switch_prob = switch_prob
        self._last_action = np.random.randint(num_actions)
        self.eval_mode = False
        pass

    # How select an action?
    # we define our policy here
    def _choose_action(self):
        if np.random.random() <= self._switch_prob:
            self._last_action = np.random.randint(self._num_actions)
        return self._last_action

    # when it checkpoints during training, anything we should do?
    def bundle_and_checkpoint(self):
        pass

    # loading from checkpoint
    def unbundle(self, unused_checkpoint_dir, unused_checkpoint_version, unused_data):
        pass

    # first action to take
    def begin_episode(self, unused_observation):
        return self._choose_action()

    # cleanup
    def end_episode(self, unused_reward):
        pass

    # we can update our policy here
    # using the reward and observation
    # dynamic programming, Q learning, monte carlo methods, etc.
    def step(self, reward, observation):
        return self._choose_action()

def create_basic_agent(sess, environment):
    """The Runner class will expect a function of this type to create an agent."""
    return BasicAgent(sess=sess, num_actions=environment.action_space.n, switch_prob=0.2)

# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework. We also explicitly terminate after 110 iterations (instead
# of the standard 200) to demonstrate the plotting of partial runs.
basic_runner = run_experiment.Runner(LOG_PATH,
                                     create_basic_agent,
                                     game_name=GAME,
                                     num_iterations=200,
                                     training_steps=10,
                                     evaluation_steps=10,
                                     max_steps_per_episode=100)

print('Will train basic agent, please be patient, may be a while...')
basic_runner.run_experiment()
print('Done training!')