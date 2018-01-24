import unittest
from gym_chess.envs.chess_env import ChessEnv
import numpy as np


def get_reset_state():
    e_map = np.array([[-4, -2, -3, -5, -6, -3, -2, -4],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [0,  0,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  0,  0],
                      [1,  1,  1,  1,  1,  1,  1,  1],
                      [4,  2,  3,  5,  6,  3,  2,  4]])
    e_legal_moves = ['g1h3', 'g1f3', 'b1c3', 'b1a3', 'h2h3', 'g2g3', 'f2f3', 'e2e3', 'd2d3', 'c2c3',
                     'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'e2e4', 'd2d4', 'c2c4',  'b2b4', 'a2a4']
    return e_map, e_legal_moves


def get_single_step_state():
    e_map = np.array([[ 4,  2,  3,  5,  6,  3,  2,  4],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                      [-1,  0,  0,  0,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                      [ 0, -1, -1, -1, -1, -1, -1, -1],
                      [-4, -2, -3, -5, -6, -3, -2, -4]])
    e_legal_moves = ['g8h6', 'g8f6', 'b8c6', 'b8a6', 'h7h6', 'g7g6', 'f7f6', 'e7e6', 'd7d6', 'c7c6',
                     'b7b6', 'a7a6', 'h7h5', 'g7g5', 'f7f5', 'e7e5', 'd7d5', 'c7c5', 'b7b5', 'a7a5']
    return e_map, e_legal_moves


def get_two_step_state():
    e_map = np.array([[-4, -2, -3, -5, -6, -3, -2, -4],
                      [-1,  0, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                      [ 0, -1,  0,  0,  0,  0,  0,  0],
                      [ 1,  0,  0,  0,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  0,  0,  0,  0],
                      [ 0,  1,  1,  1,  1,  1,  1,  1],
                      [ 4,  2,  3,  5,  6,  3,  2,  4]])
    e_legal_moves = ['g1h3', 'g1f3', 'b1c3', 'b1a3', 'a1a3', 'a1a2', 'a4b5', 'a4a5', 'h2h3', 'g2g3', 'f2f3',
                     'e2e3', 'd2d3', 'c2c3', 'b2b3', 'h2h4', 'g2g4', 'f2f4', 'e2e4', 'd2d4', 'c2c4', 'b2b4']
    return e_map, e_legal_moves


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv()

    def test_reset(self):
        e_map, e_legal_moves = get_reset_state()

        actual = self.env.reset()

        self.assertTrue(np.array_equal(e_map, actual[0]))
        self.assertTrue(np.array_equal(e_legal_moves, actual[1]))

    def test_single_step_happy_path(self):
        e_map, e_legal_moves = get_single_step_state()
        e_reward = 0.0

        self.env.reset()
        a_state, a_reward, a_terminated, _ = self.env.step("a2a4")

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))
        self.assertEqual(e_reward, a_reward)
        self.assertFalse(a_terminated)

    def test_two_step_happy_path(self):
        e_map, e_legal_moves = get_two_step_state()
        e_reward = 0.0

        self.env.reset()
        self.env.step("a2a4")
        a_state, a_reward, a_terminated, _ = self.env.step("b7b5")

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))
        self.assertEqual(e_reward, a_reward)
        self.assertFalse(a_terminated)

    def test_alt_step_happy_path(self):
        e_map, e_legal_moves = get_single_step_state()

        self.env.reset()
        a_state, a_terminated, _ = self.env.alt_step("a2a4")

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))
        self.assertFalse(a_terminated)

    def test_alt_step_step(self):
        e_map, e_legal_moves = get_single_step_state()

        self.env.reset()
        self.env.alt_step("d2d4")
        a_state, a_reward, a_terminated, _ = self.env.step("a2a4")

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))
        self.assertFalse(a_terminated)

    def test_alt_step_reset(self):
        e_map, e_legal_moves = get_reset_state()

        self.env.reset()
        self.env.alt_step("d2d4")
        a_state = self.env.reset()

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))

    def test_alt_step_reset_step(self):
        e_map, e_legal_moves = get_single_step_state()

        self.env.reset()
        self.env.alt_step("d2d4")
        self.env.reset()
        a_state, a_reward, a_terminated, _ = self.env.step("a2a4")

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))
        self.assertFalse(a_terminated)

    def test_alt_step_alt_reset(self):
        e_map, e_legal_moves = get_single_step_state()

        self.env.reset()
        self.env.step("a2a4")
        self.env.alt_step("b7b5")
        self.env.alt_step("a4a5")
        a_state = self.env.alt_reset()

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))

    def test_alt_step_alt_pop(self):
        e_map, e_legal_moves = get_two_step_state()

        self.env.reset()
        self.env.step("a2a4")
        self.env.alt_step("b7b5")
        self.env.alt_step("a4a5")
        a_state = self.env.alt_pop()

        self.assertTrue(np.array_equal(e_map, a_state[0]))
        self.assertTrue(np.array_equal(e_legal_moves, a_state[1]))


if __name__ == '__main__':
    unittest.main()
