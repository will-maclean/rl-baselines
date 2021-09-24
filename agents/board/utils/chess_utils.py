import gym
import gym_chess
import chess
import chess.pgn
import numpy as np

from utils.replay import StandardReplayBuffer

S = (119, 8, 8)
A = 4672


def result(winner):
    if winner == "1-0":
        return 1
    elif winner == "0-1":
        return -1
    else:
        return 0


def load_games(env, path, memory: StandardReplayBuffer):

    with open(path) as f:
        while True:

            game = chess.pgn.read_game(f)

            if game is None or memory.is_full():
                # no games left, or replay buffer is full
                break

            s = env.reset()

            winner = result(game.headers["Result"])

            for move in game.mainline_moves():
                encoded_move = env.encode(move)

                pi = np.zeros(A)
                pi[encoded_move] = 1

                memory.append(s, winner, pi)

                winner *= -1

                s, _, _, _ = env.step(encoded_move)

    return memory


if __name__ == "__main__":
    mem = StandardReplayBuffer(max_mem_states=10_000)
    load_games("lichess_db_standard_rated_2013-06.pgn", mem)