import glob
import itertools
import os
import pickle
from copy import copy, deepcopy
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import time

import psutil

from mctspy.games.examples.crew_node_determinization import CooperativeGameNodeDet
from mctspy.games.examples.crew_solveable import MCTSCrewSolver
from mctspy.games.examples.generate_crew_games import SUITS, DECK, DECK_SIZE, CrewStatePublic, create_board_state, \
    features_from_game_state


def game_from_beginning(seed, players, num_goals, model):
    np.random.seed(seed)
    deck = copy(DECK)
    np.random.shuffle(deck)
    initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
    # true_hands = deepcopy(initial_hands)
    captain = [DECK[-1] in hand for hand in initial_hands].index(True)
    # unknown_hands = deepcopy(true_hands)
    # unknown_hands[captain].remove(DECK[-1])
    goal_deck = copy(DECK[0:-4])
    np.random.shuffle(goal_deck)
    initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goal_cards=goal_deck[0:num_goals],
                                          captain=captain)
    initial_board_state.known_hands = initial_hands
    initial_board_state.num_unknown = [0 for _ in range(players)]
    initial_board_state.possible_cards = pd.DataFrame()
    root = CooperativeGameNodeDet(initial_board_state, root=True)
    board_state = deepcopy(initial_board_state)
    c_param = 1.4
    simulations = 1000
    while board_state.game_result is None:
        mcts = MCTSCrewSolver(root, model=model)
        mcts.create_tree(simulations_number=simulations, c_param=c_param)
        best_node = mcts.best_action()
        board_state = best_node.state
        best_node.root = True
        root = best_node
    result = board_state.game_result
    return result


if __name__ == '__main__':
    startTime = time.time()

    players = 3  # int(sys.argv[1])
    seed_start = 0  # int(sys.argv[3])
    seed_end = 1000  # int(sys.argv[4])
    interval = 100
    # max_rows_per_export = 100000

    parent_path = r'G:\Users\DavidK\analyses_not_project_specific\20220831_simulation_results\20220918_results_take2'

    with open(parent_path + r'/model_{}pl_round1.pkl'.format(players), 'rb') as f:
        model = pickle.load(f)

    seeds = range(seed_start, seed_end)
    for i in range(10):
        num_goals = i+1
        subseeds = seeds[i*interval:(i+1)*interval]
        with Pool(60) as p:
            models = [model for _ in range(interval)]
            player_list = [players for _ in range(interval)]
            num_goal_list = [num_goals for _ in range(interval)]
            results = p.starmap(game_from_beginning, zip(subseeds, player_list, num_goal_list, models))
        # results = []
        # for seed in subseeds:
        #     results.append(game_from_beginning(seed, players, num_goals, model))

        print(num_goals)
        print(sum(results))

        print()
        executionTime = (time.time() - startTime) / 60
        print('Execution time in minutes: ' + str(executionTime))
        print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
