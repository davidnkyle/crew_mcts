from collections import defaultdict

import numpy as np
import pandas as pd
from copy import copy, deepcopy






from mctspy.games.examples.crew_game_state import SUITS, DECK, CrewStatePublic
from mctspy.games.examples.integer_programming import initial_sample



class CooperativeGameNodeDet():

    def __init__(self, state, parent=None, parent_action=None, root=False):
        self.state = state
        self.parent = parent
        self.children = []
        # if unknown_hands is None:
        #     unknown_hands = [[] for _ in range(self.state.players)]
        # self.unknown_hands = deepcopy(unknown_hands)
        self.parent_action = parent_action  # add a parameter for parent action
        # self._n = pd.DataFrame(index=FEATURES, columns=range(self.state.players), data=0)
        # self._n = np.zeros((state.players, DECK_SIZE*2 + 5))
        # self._N = np.zeros((state.players, DECK_SIZE * 2 + 5))
        self._number_of_visits_total = 0
        # self._number_of_wins = pd.DataFrame(index=FEATURES, columns=range(self.state.players), data=0)
        self._number_of_wins_total = 0
        self._results = defaultdict(int)
        self._all_untried_actions = None
        # if feature_weights is None:
        #     feature_weights = FEATURE_DICT_BASE
        # self.feature_weights = feature_weights
        self.root = root

    # @property
    def untried_actions(self):
        if self._all_untried_actions is None:
            self._all_untried_actions = self.state.get_all_actions()
        legal = self.state.get_legal_actions([])
        return [a for a in self._all_untried_actions if a in legal]

    @property
    def hand(self):
        turn = self.state.turn
        return self.state.known_hands[turn]

    @property
    def q(self):
        return self._number_of_wins_total

    @property
    def n(self):
        return self._number_of_visits_total

    def is_fully_expanded(self):
        return len(self.untried_actions()) == 0


    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]


    def select(self, c_param):
        if not self.is_fully_expanded():
            raise ValueError('Only select a node when it is already fully expanded')
        node = self.best_child(c_param=c_param)
        return node

    def expand(self):
        action = self.untried_actions()[0]
        self._all_untried_actions.remove(action)
        next_state = self.state.move(action)
        child_node = CooperativeGameNodeDet(
            next_state, parent=self, parent_action=action
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        # game_states = []
        # unknown_hand_states = []
        actions = []
        game_result = current_rollout_state.game_result
        while not current_rollout_state.is_game_over():
            # game_states.append(current_rollout_state)
            # unknown_hand_states.append(unknown_hands)
            possible_moves = current_rollout_state.get_legal_actions([])
            game_result = 0
            while (len(possible_moves) > 0) and (game_result == 0):
                action = possible_moves.pop(np.random.randint(len(possible_moves)))
                new_state = current_rollout_state.move(action)
                game_result = new_state.game_result
                if game_result == 1:
                    break
            actions.append(action)
            current_rollout_state = new_state
        if game_result == 0 and current_rollout_state.rounds_left != 0:
            goal_completion = 1-(current_rollout_state.goals_remaining/self.state.num_goals)
            inv_game_completion = self.state.total_rounds/current_rollout_state.rounds_left
            game_result = goal_completion #**inv_game_completion
        if game_result == 1:
            print(actions)
        return game_result

    def backpropagate(self, result):
        self._number_of_visits_total += 1
        self._number_of_wins_total += result
        if self.parent and not self.root:
            self.parent.backpropagate(result)

    def rollout_policy(self, possible_moves):
        next_move = np.random.randint(len(possible_moves))
        # print(next_move)
        return possible_moves[next_move%len(possible_moves)]


if __name__ == '__main__':

    game = CrewStatePublic(players=3, goal_cards=['g3', 'p1', 'b4'], captain=1)
    game.possible_cards = pd.DataFrame()
    game.known_hands = [['b1', 'g2', 'g3', 'p4'], ['g1', 'g4', 'p2', 'p3', 'z2'], ['b2', 'b3', 'b4', 'p1', 'z1']]
    root = CooperativeGameNodeDet(game, root=True)
    node = root.expand()
    result = node.rollout()
    node.backpropagate(result)
    bc = root.best_child()

