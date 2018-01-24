# gym-chess
This documentation is slightly out of date and will be updated soon.
For now what you need to know is that calling env.action_space.n returns a list of legal moves

## Pre-requisites:
Below are a list of packages that are required to get the game to work, along with links for installation instructions.

### python-chess
For usage and installation instructions, see: https://pypi.python.org/pypi/python-chess

### OpenAI Gym
For installation instructions, see: https://www.youtube.com/watch?v=Io1wOuHEyW0&feature=youtu.be

## gym_chess Documentation
This environment is still actively being developed. Always check the docs for the most up to date information.

### gym_chess Installation
While in the root directory you can run the following code:
```
pip install -e .
```

### Running chess as a gym environment
```
import gym
import gym_chess

env = gym.make('chess-v0')
env.render()
```

## Return Values

### state
The environment state consists of 2 parts: [[an 8x8 array of the game board with pieces represented as integers],[A list of all legal moves]]
Pieces are assigned numerical values as such:
```
1: Pawn
2: Knight
3: Bishop
4: Rook
5: Queen
6: King
```
Values are always positive **for the side who's turn it is.**

### reward
Float value defined in the REWARD_LOOKUP dict.

Reward values are currently set as such:
```
Check: 0.05
Checkmate: 100
Stalemate: 0
Pawn promotion: 0.1
**Taking a piece:**
Pawn: 0.1
Knight: 0.3
Bishop: 0.3
Rook: 0.5
Queen: 0.9
```

### is_terminated
Boolean value representing if the game has ended in checkmate, stalemate, insufficient material, seventyfive-move rule, fivefold repetition, or a variant end condition.

### info
Dictionary containing any debugging information

## Params

### action
The move to be played

Must be represented in UCI format (i.e. 'a2a4' or 'b7b8q')

## Methods

### step(action)
Plays a move

Returns state, reward, is_terminated, info

### reset()
Resets the board to the starting position

Returns state

### render()
Prints the board to the console with pieces represented as letters

### alt_step(action)
Similar to step(). However, calling step() will remove all moves made by alt_step(), and revert to the last move used in step().

This can be used to allow the agent to look ahead several moves.

Returns state, is_terminated, info

### alt_pop()
Pops a single move performed by alt_step()

This can be used to build trees of possible future board positions.

Returns state

### alt_reset()
Pops all moves performed by alt_step() 

Returns state
