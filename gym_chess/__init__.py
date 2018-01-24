from gym.envs.registration import register

register(
    id='chess-v0',
    entry_point='gym_chess.envs:ChessEnv',
)
register(
    id='chess-extrahard-v0',
    entry_point='gym_chess.envs:ChessExtraHardEnv',
)
