import gym
from gym import error, spaces, utils
from gym.utils import seeding

import chess
import chess.svg
from IPython.display import SVG


class ChessExtraHardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass
