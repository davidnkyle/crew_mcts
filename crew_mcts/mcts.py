import time
from collections import defaultdict
from copy import copy, deepcopy
from multiprocessing import Pool

import numpy as np

import pandas as pd

from mctspy.games.examples.crew_game_state import SUITS, DECK, DECK_SIZE, CrewStatePublic
from mctspy.games.examples.crew_node import FEATURE_DICT_BASE, card_list_to_series, \
    card_series_to_list
from mctspy.games.examples.crew_node_determinization import CooperativeGameNodeDet
from mctspy.games.examples.permutation3 import swap



# implement communication signal

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001




class MCTSCrewSolver():
    def __init__(self, node):
        self.root = node


    def create_tree(self, c_param, simulations_number=None):
        # create the starting deck as a series with card indices and player index as values
        reward = 0
        for _ in range(0, simulations_number):
            v = self._tree_policy(c_param=c_param)
            reward = v.rollout()
            if reward == 1:
                break
            v.backpropagate(reward)
        actions = []
        while v.parent:
            actions.append(v.parent_action)
            v = v.parent
        print(list(reversed(actions)))
        print(_)
        return reward//1

    def best_action(self, hand):
        return self.root.best_child(hand=hand, c_param=0.0, expand_if_necessary=True)


    def _tree_policy(self, c_param=1.4):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                new_node = current_node.expand()
                return new_node
            else:
                current_node = current_node.select(c_param=c_param)
        return current_node


def run_game(seed, c_param=0.1):
    np.random.seed(seed)
    players = 3
    deck = copy(DECK)
    np.random.shuffle(deck)
    initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
    true_hands = deepcopy(initial_hands)
    captain = [DECK[-1] in hand for hand in initial_hands].index(True)
    num_goals = 3
    goal_deck = copy(DECK[0:-4])
    np.random.shuffle(goal_deck)
    initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goal_cards=goal_deck[0:num_goals],
                                          captain=captain)
    initial_board_state.known_hands = true_hands
    initial_board_state.possible_cards = pd.DataFrame()
    board_state = initial_board_state
    print()
    print()
    print('NEW GAME')
    print('----------------')
    print()
    print('goal cards: {}'.format(board_state.goal_cards))
    print(true_hands)
    print()
    print('leading: {}'.format(board_state.leading))
    first_seed = []
    deck_idx = 0
    for pl in range(initial_board_state.players):
        first_seed.append(DECK[deck_idx:deck_idx + initial_board_state.num_unknown[pl]])
        deck_idx += initial_board_state.num_unknown[pl]
    root = CooperativeGameNodeDet(initial_board_state, root=True)
    mcts = MCTSCrewSolver(root)
    simulations = 1000
    result = mcts.create_tree(simulations_number=simulations, c_param=c_param)
    return result

if __name__ == '__main__':
    startTime = time.time()
    seeds = list(range(100, 110))
    # results = []
    # for seed in seeds:
    #     results.append(run_game(seed))
    with Pool(5) as p:
        results = p.map(run_game, seeds)
    # results = [run_game(100)]
    print(sum(results))
    executionTime = (time.time() - startTime)/60
    print('Execution time in minutes: ' + str(executionTime))


