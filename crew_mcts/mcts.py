import glob
import itertools
import os
import pickle
import time
from collections import defaultdict
from copy import copy, deepcopy
from multiprocessing import Pool

import numpy as np
from datetime import datetime
import pandas as pd
import psutil
from sklearn.tree import DecisionTreeClassifier

from mctspy.games.examples.generate_crew_games import SUITS, DECK, DECK_SIZE, CrewStatePublic, create_board_state, \
    features_from_game_state
# from mctspy.games.examples.crew_game_state import SUITS, DECK, DECK_SIZE, CrewStatePublic
# from mctspy.games.examples.crew_node import FEATURE_DICT_BASE, card_list_to_series, \
#     card_series_to_list
from mctspy.games.examples.crew_node_determinization import CooperativeGameNodeDet
# from mctspy.games.examples.permutation3 import swap



# implement communication signal

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001




class MCTSCrewSolver():
    def __init__(self, node, model):
        self.root = node
        self.model = model


    def create_tree(self, c_param, simulations_number=None):
        # create the starting deck as a series with card indices and player index as values
        # reward = 0
        for _ in range(0, simulations_number):
            v = self._tree_policy(c_param=c_param)
            reward = v.rollout(self.model)
            v.backpropagate(reward)
        # actions = []
        # while v.parent:
        #     actions.append(v.parent_action)
        #     v = v.parent
        # return reward//1

    def best_action(self):
        return self.root.best_child(c_param=0.0)


    def _tree_policy(self, c_param=1.4):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                new_node = current_node.expand()
                return new_node
            else:
                current_node = current_node.select(c_param=c_param)
        return current_node


def run_game(seed, round, players, model, c_param=1.4):
    np.random.seed(seed)
    initial_board_state = create_board_state(round=round, players=players)
    feature_list = []
    root = CooperativeGameNodeDet(initial_board_state, root=True)
    board_state = deepcopy(initial_board_state)
    while board_state.game_result is None:
        mcts = MCTSCrewSolver(root, model=model)
        if len(root.state.trick) == 0:
            feature_list.append(features_from_game_state(root.state))
        simulations = 1000
        mcts.create_tree(simulations_number=simulations, c_param=c_param)
        best_node = mcts.best_action()
        board_state = best_node.state
        best_node.root = True
        root = best_node
    result = board_state.game_result
    return [features + [result] for features in feature_list]

if __name__ == '__main__':
    startTime = time.time()

    feature_cols = ['leading_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + [
        'leading_total_cards'] + \
                   ['leading_goal_{}'.format(c) for c in DECK] + ['leading_total_goals'] + \
                   ['pl1_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + [
                       'pl1_total_cards'] + \
                   ['pl1_goal_{}'.format(c) for c in DECK] + ['pl1_total_goals'] + \
                   ['pl2_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + [
                       'pl2_total_cards'] + \
                   ['pl2_goal_{}'.format(c) for c in DECK] + ['pl2_total_goals'] + \
                   ['{}_total'.format(s) for s in SUITS] + ['total_goals', 'round', 'result']

    players = 3  # int(sys.argv[1])
    for round in reversed(range(1, 14)):
        seed_start = 0  # int(sys.argv[3])
        seed_end = 1000  # int(sys.argv[4])
        max_rows_per_export = 100000

        parent_path = r'G:\Users\DavidK\analyses_not_project_specific\20220831_simulation_results\20220918_results_take2'

        model = None
        if round < 13:
            with open(parent_path + r'/model_{}pl_round{}.pkl'.format(players, round+1), 'rb') as f:
                model = pickle.load(f)

        os.makedirs(parent_path, exist_ok=True)

        for idx in range(int(np.ceil((seed_end - seed_start) / max_rows_per_export))):
            seeds = range(idx * max_rows_per_export, min(seed_end, (idx + 1) * max_rows_per_export))

            with Pool(60) as p:
                models = [model for _ in range(len(seeds))]
                player_list = [players for _ in range(len(seeds))]
                round_list = [round for _ in range(len(seeds))]
                list_of_rows = p.starmap(run_game, zip(seeds, round_list, player_list, models))
            rows = list(itertools.chain.from_iterable(list_of_rows))
            # rows = []
            # for seed in seeds:
            #     rows += run_game(seed, round, players, model)

            df = pd.DataFrame(data=rows, columns=feature_cols)

            export_file = parent_path + '/pl{}_round{}_{}_{}_{}.csv'.format(players, round, seeds[0], seeds[-1],
                                                                            datetime.today().strftime('%Y%m%d'))
            print(export_file)
            df.to_csv(export_file)
            print('{} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
            del df
            del rows

        all_files = glob.glob(parent_path + '/*.csv')
        li = []

        print('reading data')

        for filename in all_files:
            df = pd.read_csv(filename, index_col='Unnamed: 0')
            li.append(df)
            print('.', end='')

        df = pd.concat(li)
        del li

        print()
        executionTime = (time.time() - startTime) / 60
        print('Execution time in minutes: ' + str(executionTime))
        print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

        y_train = np.array(df['result'])
        x_train = df.drop('result', axis=1).values

        print('fitting model')

        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(x_train, y_train)

        print('writing results')

        with open(parent_path + '/model_{}pl_round{}.pkl'.format(players, round), 'wb') as f:
            pickle.dump(model, f)

        executionTime = (time.time() - startTime) / 60
        print('Execution time in minutes: ' + str(executionTime))
        print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))


