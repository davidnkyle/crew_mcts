import time
from copy import copy, deepcopy
import pickle
import itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
from datetime import datetime
import os, psutil
import sys

# first argument is number of players
# second argument is which round this is modelling
# third argument seed range start
# fourth argument seed range end

non_trump = 'bgpy'
trump = 'z'
SUITS = non_trump + trump
DECK = ['{}{}'.format(color, number) for color in non_trump for number in range(1, 10)] + \
    ['{}{}'.format(trump, number) for number in range(1, 5)]
COMMS = ['{}{}{}'.format(color, number, modifier) for color in non_trump for number in range(1, 10) for modifier in 'hol']
DECK_ARRAY = np.array(DECK)
DECK_SIZE = len(DECK)


def evaluate_trick(trick):
    if len(trick) == 0:
        raise ValueError('No trick to evaluate')
    suit = trick[0][0]
    cards = [c for c in trick if c[0] in (suit, 'z')]
    cards.sort(reverse=True)
    return trick.index(cards[0])

def map_known_cards(known_cards):
    map = {}
    for suit in SUITS:
        max_value = 9
        if suit == 'z':
            max_value = 4
        cards_in_suit = [c for c in known_cards if c[0] == suit]
        length = len(cards_in_suit)
        if length > 0:
            cards_in_suit.sort()
            cards_in_suit.reverse()
            for i in range(length):
                map[cards_in_suit[i]] = '{}{}'.format(suit, max_value - i)
            if length > 1:
                map[cards_in_suit[-1]] = '{}1'.format(suit)
    return map


class CrewStatePublic():
    def __init__(self, players=3, num_goals=3, goal_cards=None, captain=None):
        if (players < 3) or (players > 5):
            raise ValueError('Only allow between 3 and 5 players')
        self.players = players
        self.discard = []
        self.num_unknown = [len(DECK[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // self.players]) for i in range(self.players)]
        self.known_hands = [[] for _ in range(self.players)]
        self.captain = captain
        self.possible_cards = pd.DataFrame(index=DECK, columns=range(self.players), data=1)
        if self.captain is not None:
            self.player_has(self.captain, DECK[-1])
        self.leading = captain
        self.turn = captain
        self.num_goals = num_goals
        self.select_goals_phase = True
        self.communication_phase = False
        self.coms = [None for _ in range(self.players)]
        if goal_cards is None:
            goal_cards = []
        self.goal_cards = goal_cards
        self.goals = [[] for _ in range(self.players)]
        self.total_rounds = DECK_SIZE//self.players
        self.rounds_left = DECK_SIZE//self.players
        self.trick = []

    def player_has(self, player, card, flesh_out=True):
        if card not in self.known_hands[player]:
            self.known_hands[player].append(card)
            self.num_unknown[player] -= 1
            if self.num_unknown[player] < 0:
                raise ValueError('Impossible situation')
            self.possible_cards.drop(card, inplace=True)
        if flesh_out:
            self.flesh_out_possibilities()

    @property
    def known_cards(self):
        known_hands = list(itertools.chain.from_iterable(self.known_hands))
        return known_hands + self.trick + self.discard

    def unknown_hand(self, hand):
        kc = self.known_cards
        return [c for c in hand if c not in kc]

    @property
    def goals_remaining(self):
        return sum([len(goal) for goal in self.goals])

    @property
    def game_result(self):
        if not self.select_goals_phase:
            players_with_goals_left = 0
            for pl in self.goals:
                if len(pl) > 0:
                    players_with_goals_left += 1
                    for c in pl:
                        if c in self.discard:
                            return 0 # if the goal is still active and in the discard pile, there is no way to win
            if players_with_goals_left == 0:
                return 1
            if players_with_goals_left > self.rounds_left:
                return 0
        return None

    def is_game_over(self):
        return self.game_result is not None

    def flesh_out_possibilities(self):
        rows_good = True
        columns_good = True
        if (self.possible_cards.sum(axis=1) == 0).any():
            raise ValueError('Impossible matrix')
        for card, row in self.possible_cards.loc[self.possible_cards.sum(axis=1) == 1].iterrows():
            rows_good = False
            pl = row[row == 1].index[0]
            self.player_has(pl, card, flesh_out=False)
        players = []
        for pl in self.possible_cards.columns:
            if self.possible_cards[pl].sum() < self.num_unknown[pl]:
                raise ValueError('impossible matrix')
            elif self.possible_cards[pl].sum() == self.num_unknown[pl]:
                columns_good = False
                for card in self.possible_cards.loc[self.possible_cards[pl] == 1].index:
                    self.player_has(pl, card, flesh_out=False)
                if self.num_unknown[pl] != 0:
                    raise ValueError('impossible matrix')
                players.append(pl)
        self.possible_cards.drop(players, axis=1, inplace=True)
        if (not rows_good) or (not columns_good):
            self.flesh_out_possibilities()

    def move(self, move):
        new = deepcopy(self)
        if new.select_goals_phase:
            new.goals[new.turn].append(move)
            new.goal_cards.remove(move)
            new.turn = (new.turn + 1) % self.players
            if len(new.goal_cards) == 0:
                new.select_goals_phase = False
                # new.communication_phase = True
                new.turn = new.captain
            return new
        if new.communication_phase:
            if move is not None:
                if new.coms[new.turn] is not None:
                    raise ValueError('can only communicate once')
                new.coms[new.turn] = move
                card = move[:2]
                new.player_has(new.turn, card)
                num = int(move[1])
                # add structural zeros
                if move[2] in ['o', 'h']:
                    if num < 9:
                        upper_index = new.possible_cards.filter(regex='{}[{}-9]'.format(move[0], num+1), axis=0).index
                        if not upper_index.empty and new.turn in new.possible_cards.columns:
                            new.possible_cards.loc[upper_index, new.turn] = 0
                        # new.possible_cards[new.turn, idx+1:(idx//9 + 1)*9] = 0
                if move[2] in ['o', 'l']:
                    if num > 1:
                        lower_index = new.possible_cards.filter(regex='{}[1-{}]'.format(move[0], num-1), axis=0).index
                        if not lower_index.empty and new.turn in new.possible_cards.columns:
                            new.possible_cards.loc[lower_index, new.turn] = 0
                        # new.possible_cards[new.turn, (idx // 9) * 9:idx] = 0
                new.flesh_out_possibilities()
            while new.communication_phase:
                new.turn = (new.turn + 1) % new.players
                if new.turn == new.leading:
                    new.communication_phase = False
                if new.coms[new.turn] is None:
                    break
            return new
        new.player_has(new.turn, move)
        new.trick.append(move)
        new.known_hands[new.turn].remove(move)
        if len(new.trick) > 1:
            # if player did not follow suit they dont have any of that suit
            leading_suit = new.trick[0][0]
            if move[0] != leading_suit:
                suit_index = new.possible_cards.filter(regex=leading_suit, axis=0).index
                if not suit_index.empty and new.turn in new.possible_cards.columns:
                    new.possible_cards.loc[suit_index, new.turn] = 0
                new.flesh_out_possibilities()
                # start = SUITS.index(leading_suit)
                # new.possible_cards[self.turn, start*9:(start+1)*9] = 0
        if len(new.trick) < new.players:
            new.turn = (new.turn + 1) % self.players
            return new
        winner = (evaluate_trick(new.trick) + new.leading) % new.players
        new.goals[winner] = [g for g in new.goals[winner] if g not in new.trick]
        # new.goals[winner] = list(set(new.goals[winner]).difference(new.trick)) # remove any goals in the trick
        new.discard += new.trick # add trick to discard
        new.trick = []
        new.rounds_left -= 1
        new.leading = winner
        new.turn = winner
        # new.communication_phase = True
        # if new.coms[new.turn] is not None:
        #     while new.communication_phase:
        #         new.turn = (new.turn + 1) % new.players
        #         if new.turn == new.leading:
        #             new.communication_phase = False
        #         if new.coms[new.turn] is None:
        #             break
        return new

    def full_hand(self, unknown_hand):
        return self.known_hands[self.turn] + unknown_hand

    def is_move_legal(self, move, unknown_hand):
        if self.select_goals_phase:
            return move in self.goal_cards
        full_hand = self.full_hand(unknown_hand)
        if self.communication_phase:
            if move is None:
                return True
            if move[0] not in 'bgpy':
                return False
            in_suit = [c for c in full_hand if c[0]==move[0]]
            in_suit.sort()
            if len(in_suit) == 1:
                if move[2] == 'o':
                    return in_suit[0] == move
            elif len(in_suit) > 1:
                if move[2] == 'h':
                    return in_suit[-1] == move
                elif move[2] == 'l':
                    return in_suit[0] == move
            return False
        if not move in full_hand: # you dont have this card in your hand
            return False
        if len(self.trick) > 0:
            leading_suit = self.trick[0][0] # you must follow suit if you can
            if leading_suit in [c[0] for c in full_hand]:
                return move[0] == leading_suit
        return True

    def get_legal_actions(self, unknown_hand):
        full_hand = self.full_hand(unknown_hand)
        if self.select_goals_phase:
            return copy(self.goal_cards)
        if self.communication_phase:
            allowable = []
            if self.coms[self.turn] is None:
                sort_hand = copy(full_hand)
                sort_hand.sort()
                for suit in 'bgpy':
                    in_suit = [c for c in sort_hand if c[0] == suit]
                    if len(in_suit) == 1:
                        allowable.append(in_suit[0] + 'o')
                    elif len(in_suit) > 1:
                        allowable.append(in_suit[0] + 'l')
                        allowable.append(in_suit[-1] + 'h')
            return allowable
        return [c for c in full_hand if self.is_move_legal(c, full_hand)]

    def get_all_actions(self):
        if self.select_goals_phase:
            return copy(self.goal_cards)
        if self.communication_phase:
            if self.coms[self.turn] is None:
                return copy(COMMS) + [None]
            return [None]
        return copy(DECK)

    def to_feature_form(self):
        new = deepcopy(self)
        if not new.possible_cards.empty:
            raise ValueError('this feature is not implemented to handle possible_cards')
        if len(new.trick) > 0:
            raise ValueError('this method is not implemented for mid-trick moves')
        pl_shift = (-self.leading) % new.players
        new.leading = 0
        new.discard = []
        map = map_known_cards(new.known_cards)
        new.known_hands = [[map[c] for c in self.known_hands[(idx + self.leading) % new.players]] for idx in range(new.players)]
        new.captain = (self.captain + pl_shift) % new.players
        new.turn = (self.turn + pl_shift) % new.players
        # new.coms = [self.coms[(idx + pl_shift) % new.players] for idx in range(new.players)]
        new.goals = [[map[c] for c in self.goals[(idx + self.leading) % new.players]] for idx in range(new.players)]
        return new


def hand_to_vector(hand):
    values = []
    for card in DECK:
        if card in hand:
            values.append(1)
        else:
            values.append(0)
    return values



def features_from_game_state(game_state):
    features = []
    suit_sums = [0, 0, 0, 0, 0]
    total_goals = 0
    for pl in range(3):
        vec = hand_to_vector(game_state.known_hands[pl])
        features += vec
        for i in range(5):
            total = sum(vec[9 * i: 9 * (i + 1)])
            suit_sums[i] += total
            features.append(total)
        features.append(sum(vec))
        goal_vec = hand_to_vector(game_state.goals[pl])
        features += goal_vec
        player_goals = sum(goal_vec)
        features.append(player_goals)
        total_goals += player_goals
    features += suit_sums
    features.append(total_goals)
    features.append(game_state.rounds_left)
    return features


def check_for_viability(turns, game_state, model):
    if turns == 0:
        gr = game_state.game_result
        if gr is None:
            new = game_state.to_feature_form()
            return model.predict(np.array([features_from_game_state(new)]))[0]
        if gr == 1:
            gr = np.inf
        return gr
    for action in game_state.get_legal_actions([]):
        state = game_state.move(action)
        r = check_for_viability(turns-1, state, model)
        if r > 0:
            return r
    return 0


def create_board_state(round=13, players=3):
    deck = copy(DECK)
    some_non_trump_cards = False
    max_tries = 100
    while not some_non_trump_cards and max_tries > 0:
        np.random.shuffle(deck)
        initial_board_state = CrewStatePublic(players=players, goal_cards=[],
                                              captain=0)

        initial_board_state.leading = 0
        initial_board_state.num_unknown = [0, 0, 0]
        rounds_left = len(deck)//players - round + 1
        initial_board_state.rounds_left = rounds_left
        initial_board_state.select_goals_phase = False
        extra_card = 0
        if players==3:
            extra_card = 1
        all_cards_raw = deck[0:rounds_left*players + extra_card]
        map = map_known_cards(all_cards_raw)
        all_cards = [map[c] for c in all_cards_raw]
        non_trump_cards = [c for c in all_cards if 'z' not in c]
        some_non_trump_cards = (len(non_trump_cards) > 0)
        max_tries -= 1
    num_goals = np.random.randint(min(len(non_trump_cards), 10)) + 1
    goals = non_trump_cards[:num_goals]
    max_num_players_with_goals = min(rounds_left, 3)
    players_with_goals = np.random.choice(range(3), max_num_players_with_goals, replace=False)
    for goal in goals:
        pl_idx = np.random.randint(max_num_players_with_goals)
        initial_board_state.goals[players_with_goals[pl_idx]].append(goal)
    np.random.shuffle(all_cards)
    initial_hands = [all_cards[j*rounds_left:(j+1)*rounds_left] for j in range(players)]
    if players==3:
        player_with_extra_card = np.random.randint(players)
        initial_hands[player_with_extra_card].append(all_cards[-1])
    initial_board_state.known_hands = initial_hands
    initial_board_state.possible_cards = pd.DataFrame()
    return initial_board_state


def generate_crew_games(seed, model, players, round):

    np.random.seed(seed)
    # inputs = []
    # results = []

    initial_board_state = create_board_state(round, players)

    features = features_from_game_state(initial_board_state)
    # inputs.append(features)
    result = check_for_viability(3, initial_board_state, model)
    # results.append(result)

    return features + [result]

    # df = pd.DataFrame(data=inputs, columns=feature_cols)
    # df['result'] = results
    # return df

if __name__ == '__main__':
    startTime = time.time()

    feature_cols = ['leading_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + ['leading_total_cards'] + \
                   ['leading_goal_{}'.format(c) for c in DECK] + ['leading_total_goals'] + \
                   ['pl1_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + ['pl1_total_cards'] + \
                   ['pl1_goal_{}'.format(c) for c in DECK] + ['pl1_total_goals'] + \
                   ['pl2_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + ['pl2_total_cards'] + \
                   ['pl2_goal_{}'.format(c) for c in DECK] + ['pl2_total_goals'] + \
                   ['{}_total'.format(s) for s in SUITS] + ['total_goals', 'result']

    players = 3 #int(sys.argv[1])
    round = 12 #int(sys.argv[2])
    seed_start = 0 #int(sys.argv[3])
    seed_end = 30000000 #int(sys.argv[4])
    max_rows_per_export = 1000000

    if round == len(DECK)//players:
        model = None
    else:
        with open(r'model_{}pl_round{}.pkl'.format(players, round+1), 'rb') as f:
            model = pickle.load(f)

    parent_path = r'G:\Users\DavidK\analyses_not_project_specific\20220831_simulation_results\results'
    os.makedirs(parent_path, exist_ok=True)

    for idx in range(int(np.ceil((seed_end-seed_start)/max_rows_per_export))):
        seeds = range(idx*max_rows_per_export, min(seed_end, (idx+1)*max_rows_per_export))

        with Pool(60) as p:
            models = [model for _ in range(len(seeds))]
            player_list = [players for _ in range(len(seeds))]
            round_list = [round for _ in range(len(seeds))]
            rows = p.starmap(generate_crew_games, zip(seeds, models, player_list, round_list))
        # rows = []
        # for seed in seeds:
        #     rows.append(generate_crew_games(seed, model, players, round))

        df = pd.DataFrame(data=rows, columns=feature_cols)

        export_file = parent_path + '/pl{}_round{}_{}_{}_{}.csv'.format(players, round, seeds[0], seeds[-1], datetime.today().strftime('%Y%m%d'))
        print(export_file)
        df.to_csv(export_file)
        print('{} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
        del df
        del rows

    executionTime = (time.time() - startTime) / 60
    print('Execution time in minutes: ' + str(executionTime))
    print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
