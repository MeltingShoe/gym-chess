import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

try:
    import chess
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}.  (HINT: see README for python-chess installation instructions".format(e))


class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = chess.Board()
        self.alt_moves = 0
        self.reward_lookup = {
            'check': 0.05,
            'mate': 100.0,
            'stalemate': 0.0,
            'p': 0.1,
            'n': 0.3,
            'b': 0.3,
            'r': 0.5,
            'q': 0.9,
            '1': 0.1,  # Promotion to pawn
            '2': 0.1,  # Promotion to knight
            '3': 0.1,  # Promotion to bishop
            '4': 0.1,  # Promotion to rook
            '5': 0.1   # Promotion to queen
        }

    def _step(self, action):
        """
        input: action in UCI format (i.e. 'a2a4')

        :return:
            state: numpy array:  [[board with all pieces represented as integers], [list of legal moves]]
            reward: Float value
            is_terminated: if game has ended in checkmate, stalemate, insufficient material, seventyfive-move rule,
            info: dictionary containing any debugging information
                           fivefold repetition, or a variant end condition.
        """
        self.alt_reset()

        reward = self._generate_reward(action)

        self.env.push_uci(action)

        state = self._get_array_state()
        reward = self._update_reward(reward)
        is_terminated = self.env.is_game_over()
        info = {}
        return state, reward, is_terminated, info

    def alt_step(self, action):
        """
        Similar to step().  However, calling step() will remove all moves made by alt_step(), and
        revert to the last move used in step().

        :param action: action in UCI format (i.e. 'a2a4')
        :return:
            state: numpy array:  [[board with all pieces represented as integers], [list of legal moves]]
            is_terminated: if game has ended in checkmate, stalemate, insufficient material, seventyfive-move rule,
                           fivefold repetition, or a variant end condition.
        """
        self.alt_moves += 1
        self.env.push_uci(action)

        state = self._get_array_state()
        is_terminated = self.env.is_game_over()
        info = {}
        return state, is_terminated, info

    def alt_pop(self):
        """Pops a single move performed by alt_step()"""
        if self.alt_moves > 0:
            self.env.pop()
            self.alt_moves -= 1
        return self._get_array_state()

    def alt_reset(self):
        """Pops all moves performed by alt_step()"""
        while self.alt_moves > 0:
            self.env.pop()
            self.alt_moves -= 1
        return self._get_array_state()

    def _reset(self):
        """
        :return: current state as numpy array
        """
        self.env.reset()
        self.alt_moves = 0
        state = self._get_array_state()
        return state

    def _render(self, mode='human', close=False):
        print(self.env)

    def _get_legal_move_list(self):
        a = list(enumerate(self.env.legal_moves))
        b = [x[1] for x in a]
        c = []
        i = 0
        for item in b:
            c.append(str(b[i]))
            i += 1
        return c

    def _get_array_state(self):
        """
        input: String from chess.Board.board_fen().  Ex.: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

        Each lower-case character is black piece, and upper case is white piece.

        :return: 8x8 numpy array.  Current player's pieces are positive integers, enemy pieces are negative.
        """
        state = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]])

        split_board = self.env.board_fen().split("/")
        row = 0
        for rank in split_board:
            col = 0
            for file in rank:
                if file.isdigit():
                    col += int(file)
                else:
                    piece_enum = chess.Piece.from_symbol(file).piece_type
                    if (self.env.turn and file.islower()) or (file.isupper() and not self.env.turn):
                        piece_enum *= -1

                    state[row][col] = piece_enum
                    col += 1
            row += 1
        legal_moves = self._get_legal_move_list()
        return [state, legal_moves]

    def _update_reward(self, current_reward):
        reward = current_reward

        if self.env.is_check():
            reward += self.reward_lookup['check']

        end_game_result = self.env.result()
        if '1-0' in end_game_result or '0-1' in end_game_result:
            reward = self.reward_lookup['mate']
        elif '1/2-1/2' in end_game_result:
            reward = self.reward_lookup['stalemate']

        return reward

    def _generate_reward(self, action):
        """Assign rewards to moves, captures, queening, checks, and winning"""
        reward = 0.0
        piece_map = self.env.piece_map()

        to_square = chess.Move.from_uci(action).to_square
        if to_square in piece_map.keys():
            captured_piece = piece_map[to_square].symbol()
            reward = self.reward_lookup[captured_piece.lower()]

        promotion = chess.Move.from_uci(action).promotion
        if promotion is not None:
            reward += self.reward_lookup[str(promotion)]
        return reward

    def get_rewards_dict(self):
        '''returns dict of rewards'''
        return self.reward_lookup

    def set_reward(self, index, value):
        '''sets value of reward given it's index'''
        if index in self.reward_lookup:
            self.reward_lookup[index] = value
        else:
            return 'index not found'
        return self.reward_lookup
