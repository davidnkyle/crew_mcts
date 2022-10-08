import itertools

import numpy as np
import pandas as pd
from copy import copy, deepcopy

non_trump = 'bgpy'
trump = 'z'
SUITS = non_trump + trump
DECK = ['{}{}'.format(color, number) for color in non_trump for number in range(1, 10)] + \
    ['{}{}'.format(trump, number) for number in range(1, 5)]
COMMS = ['{}{}{}'.format(color, number, modifier) for color in non_trump for number in range(1, 10) for modifier in 'hol']
# weights = [1, 1, 1, 1, 1, 1, 2, 3, 5]
# deck_weights = weights*4 + [5, 5, 7, 10]
# DECK_WEIGHTS = np.array(deck_weights + deck_weights + [5, 5, 5, 5, 5])
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

        # self.possible_cards = np.ones((self.players, DECK_SIZE))
        # self.possible_cards[:, DECK.index('z4')] = 0
        # self.possible_cards[self.captain, DECK.index('z4')] = 1
        # self.weights = copy(DECK_WEIGHTS)
        # for goal in self.goal_cards:
        #     self.weights[DECK.index(goal)] += 10


    def player_has(self, player, card, flesh_out=True):
        if card not in self.known_hands[player]:
            self.known_hands[player].append(card)
            self.num_unknown[player] -= 1
            if self.num_unknown[player] < 0:
                raise ValueError('Impossible situation')
            self.possible_cards.drop(card, inplace=True)
        if flesh_out:
            self.flesh_out_possibilities()

    # def _deal_goals(self):
    #     cards = random.sample(DECK, self.num_goals)
    #     self.goals = [[] for _ in range(self.players)]
    #     i = self.captain
    #     for c in cards:
    #         self.goals[i].append(c)
    #         i = (i+1)%self.players

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

    # def game_reward(self, l=0.1):
    #     if not self.select_goals_phase:
    #         lose = False
    #         goals_left = 0
    #         for pl in self.goals:
    #             for c in pl:
    #                 if c in self.discard:
    #                     lose = True # if the goal is still active and in the discard pile, there is no way to win
    #                 goals_left += 1
    #         goal_completion = 1 - goals_left/self.num_goals
    #         game_completion = 1 - self.rounds_left/self.total_rounds
    #         pity_prize = l*goal_completion*game_completion
    #         if lose:
    #             return pity_prize
    #         if goals_left == 0:
    #             return 1
    #         if self.rounds_left == 0:
    #             return pity_prize
    #     return None

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

    #
    # def get_feature_idx(self, hand):
    #     idxs = []
    #     for i in range(DECK_SIZE):
    #         if DECK[i] in hand:
    #             idxs.append(i)
    #         else:
    #             idxs.append(i + DECK_SIZE)
    #     str_hand = ''.join(hand)
    #     for j in range(len(SUITS)):
    #         if SUITS[j] not in str_hand:
    #             idxs.append(j + 2*DECK_SIZE)
    #     return idxs
    #
    # def get_feature_vector(self, hand):
    #     v = np.zeros(DECK_SIZE*2 + 5)
    #     v[self.get_feature_idx(hand)] = 1
    #     return v

    # def flesh_out(self):
    #     if np.less(self.possible_cards.sum(axis=1), self.num_cards_per_player).any():
    #         raise ValueError('Impossible matrix')
    #     if np.less(self.possible_cards.sum(axis=0), 1).any():
    #         raise ValueError('Impossible matrix')
    #

if __name__ == '__main__':
    game = CrewStatePublic(players=3, goal_cards=['g3', 'p1', 'b4'], captain=1)
    game = game.move('p1')
    game = game.move('b4')
    game = game.move('g3')
    # game = game.move('p2h')
    # game = game.move('b4o')
    # game = game.move('g3h')
    game = game.move('z2')
    game = game.move('p1')
    game = game.move('z1')
    game = game.move('b2')
    game = game.move('b4')
    game = game.move('b1')
    game = game.move('p3')
    game = game.move('g1')
    game = game.move('p2')
    game = game.move('g2')
    game = game.move('g3')
    game = game.move('g4')
    print(game.game_result)
