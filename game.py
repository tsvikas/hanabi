import random
from pprint import pprint
from collections import namedtuple
from enum import Enum
from typing import List, NamedTuple, Tuple


class EndMode(Enum):
    official = 1
    endless = 2
    fair = 3


class IllegalMove(Exception):
    pass


class Suit(int):
    @classmethod
    def from_str(cls, s):
        if len(s) == 1:
            for c in ['a', 'A']:
                d = ord(s) - ord(c)
                if 0 <= d <= 10:
                    return cls(d)
        return cls(int(s))
    def __str__(self):
        return chr(ord("A") + self)
    __repr__ = __str__


class Rank(int):
    @classmethod
    def from_str(cls, s):
        if len(s) == 1:
            for c in ['1']:
                d = ord(s) - ord(c)
                if 0 <= d <= 10:
                    return cls(d)
        return cls(int(s) - 1)
    def __str__(self):
        return chr(ord("1") + self)
    __repr__ = __str__


class KnownCard(namedtuple('KnownCard', 'suit rank')):
    def __repr__(self):
        return f':{self.suit}{self.rank}'


class Card(namedtuple('Card', 'id data')):
    def __repr__(self):
        return f'#{self.id:02}{self.data or ""}'
    def hidden(self):
        return self._replace(data=None)


class Tokens(namedtuple('Tokens', 'clues lives')):
    pass


class Rules(namedtuple('Rules', 'max_tokens suits ranks cards_per_player')):
    pass


class Move():
    @property
    def move_prefix(self):
        return f'{self.move} '

    @property
    def resolved_prefix(self):
        return f'{self.cur_player}>{self.move} '

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(cls.identifier, *args, **kwargs)


class Clue(namedtuple('Clue', 'move player type param'), Move):
    identifier = 'c'

    def __repr__(self):
        return self.move_prefix + f'P={self.player} {self.type}={self.param}'


class Play(namedtuple('Play', 'move card_id'), Move):
    identifier = 'p'

    def __repr__(self):
        return self.move_prefix + f'{self.card_id}'


class Discard(namedtuple('Discard', 'move card_id'), Move):
    identifier = 'd'

    def __repr__(self):
        return self.move_prefix + f'{self.card_id}'

IDENTIFIER_TO_MOVE = {cls.identifier: cls for cls in [Clue, Play, Discard]}


def tuple_to_move(tup: Tuple) -> NamedTuple:
    return IDENTIFIER_TO_MOVE[tup[0]]._make(tup)
    
    
class ResolvedClue(namedtuple('ResolvedClue', 'move cur_player player type param cards cards_neg'), Clue):
    def __repr__(self):
        return self.resolved_prefix + f'player={self.player} {self.type}={self.param} {self.cards} {self.cards_neg}'


class ResolvedPlay(namedtuple('ResolvedPlay', 'move cur_player card new_card is_success'), Play):
    @property
    def card_id(self):  # overwrite the parent property
        return self.card.id

    def __repr__(self):
        return self.resolved_prefix + f'{self.card}{"+" if self.is_success else "-"} {self.new_card}'


class ResolvedDiscard(namedtuple('ResolvedDiscard', 'move cur_player card new_card'), Discard):
    @property
    def card_id(self):  # overwrite the parent property
        return self.card.id

    def __repr__(self):
        return self.resolved_prefix + f'{self.card}  {self.new_card}'


class ResolvedDraw(namedtuple('ResolvedDraw___', 'move cur_player cards'), Move):
    identifier = 'n'

    def __repr__(self):
        return self.resolved_prefix + f'{self.cards}'

DEFAULT_RULES = Rules(max_tokens=Tokens(8, 4), cards_per_player=None, suits=5, ranks=[3, 2, 2, 2, 1])


class Hanabi:
    def __init__(self, players, rules=DEFAULT_RULES, deck=None, allow_cheats=False, end_mode=EndMode.official):
        if rules.cards_per_player is None:
            rules = rules._replace(cards_per_player=5 if len(players) <= 3 else 4)
        self.players = players
        self.rules = rules
        self.deck = deck
        self.allow_cheats = allow_cheats
        if not isinstance(end_mode, EndMode):
            end_mode = EndMode[end_mode]
        self.end_mode = end_mode

        self.current_player = None

        if deck is None:
            self.deck = self.new_shuffled_deck(self.rules.suits, self.rules.ranks)
        self.deck_start = self.deck.copy()

        self.hands = tuple([[] for _ in players])
        self.hands_start = None
        self.tokens = self.rules.max_tokens
        self.log = []
        self.notes = []
        self.player_states = [None] * len(players)

        self.final_player = None
        self.slots = [0] * self.rules.suits
        self.discard_pile = tuple([[0] * len(self.rules.ranks) for _ in range(self.rules.suits)])

    def run(self) -> int:
        if self.log:
            raise RuntimeError("run already done")
        self.deal_cards()
        while True:
            for i in self.iterate_players():
                if isinstance(self.final_player, tuple):
                    self.final_player = self.final_player[1]
                hands = list(self.hands)
                if not self.allow_cheats:
                    hands[i] = [card.hidden() for card in hands[i]]
                player = self.players[self.current_player]
                self.player_states[i], move, note = player(self.player_states[i], self.log, hands, self.rules, self.tokens, self.slots, self.discard_pile)
                self.notes.append(note)
                self.resolve(tuple_to_move(move))
                if self.is_game_over():
                    return self.score

    @staticmethod
    def new_shuffled_deck(suits: int, ranks: List[int]) -> List[Card]:
        cards = []
        for suit in range(suits):
            for rank, count in enumerate(ranks):
                cards += [KnownCard(Suit(suit), Rank(rank))] * count
        random.shuffle(cards)
        return list(reversed([Card(card_id, card) for card_id, card in enumerate(cards)]))

    def deal_cards(self) -> None:
        for player in self.iterate_players():
            cards = [self.take_card_from_deck_to_hand() for _i in range(self.rules.cards_per_player)]
            self.log.append(ResolvedDraw.create(self.current_player, cards))
            self.notes.append('deal')
            if any(card is None for card in cards):
                raise RuntimeError("not enough cards to deal")
        self.hands_start = tuple([hand.copy() for hand in self.hands])
        
    def iterate_players(self):
        for i in range(len(self.players)):
            self.current_player = i
            yield i

    def take_card_from_deck_to_hand(self) -> Card:
        if not self.deck:
            if self.end_mode == EndMode.fair and self.final_player is None:
                self.final_player = (self.current_player - 1) % len(self.hands)
            return None
        card = self.deck.pop()
        if self.end_mode == EndMode.official and not self.deck and self.final_player is None:
            self.final_player = (None, self.current_player)
        self.hands[self.current_player].append(card)
        return card.hidden()

    def is_game_over(self) -> bool:
        if self.end_mode == EndMode.endless and not [i for i in self.hands[(self.current_player + 1) % len(self.hands)] if i]:
            return True
        return (self.lives == 0 or self.final_player == self.current_player or
                all(slot == len(self.rules.ranks) for slot in self.slots))

    @classmethod
    def resolve_clue(_cls, cur_player, hands, move: NamedTuple):
            if move.player == cur_player:
                raise IllegalMove("can't clue yourself")
            cards = [card for card in hands[move.player] if getattr(card.data, move.type) == move.param]
            if not cards:
                raise IllegalMove("no empty clues")
            cards_pos = [card.hidden() for card in cards]
            cards_neg = [card.hidden() for card in hands[move.player] if card not in cards]
            return ResolvedClue.create(cur_player, move.player, move.type, move.param, cards_pos, cards_neg)
        
    def resolve(self, move: NamedTuple) -> None:
        if isinstance(move, Clue):
            if self.clues <= 0:
                raise IllegalMove("no clues to give")
            clue = self.resolve_clue(self.current_player, self.hands, move)
            self.clues -= 1
            self.log.append(clue)
        elif isinstance(move, Play):
            card = self.take_card_from_current_hand(move.card_id)
            is_success = self.slots[card.data.suit] == card.data.rank
            if is_success:
                self.slots[card.data.suit] += 1
                if self.slots[card.data.suit] == len(self.rules.ranks):
                    if self.clues < self.rules.max_tokens.clues:
                        self.clues += 1
            else:
                self.discard_pile[card.data.suit][card.data.rank] += 1
                self.lives -= 1
                assert self.lives >= 0
            drawn = self.take_card_from_deck_to_hand()
            self.log.append(ResolvedPlay.create(self.current_player, card, drawn, is_success))
        elif isinstance(move, Discard):
            card = self.take_card_from_current_hand(move.card_id)
            self.discard_pile[card.data.suit][card.data.rank] += 1
            self.clues += 1
            drawn = self.take_card_from_deck_to_hand()
            self.log.append(ResolvedDiscard.create(self.current_player, card, drawn))
        else:
            raise IllegalMove('no such move')

    @property
    def clues(self):
        return self.tokens.clues

    @clues.setter
    def clues(self, value: int):
        self.tokens = self.tokens._replace(clues=value)

    @property
    def score(self):
        return sum(self.slots) if self.lives > 0 else 0

    @property
    def lives(self):
        return self.tokens.lives

    @lives.setter
    def lives(self, value: int):
        self.tokens = self.tokens._replace(lives=value)

    def take_card_from_current_hand(self, card_id: int) -> Card:
        return self.take_card_from_hand(self.hands[self.current_player], card_id)
    
    @staticmethod
    def take_card_from_hand(hand: List[int], card_id: int) -> Card:
        for i, card in enumerate(hand):
            if card.id == card_id:
                del hand[i]
                return card
        raise IllegalMove(f'no such card in hand: {card_id}')
    
    def id_to_orig_card(self, card_id):
        return self.deck_start[-1 - card_id]

    def log_with_spoilers(self):
        log = []
        for move in self.log:
            if isinstance(move, (Play, Discard)):
                log.append(move._replace(new_card=move.new_card and self.id_to_orig_card(move.new_card.id)))
            elif isinstance(move, Clue):
                log.append(move._replace(cards=[self.id_to_orig_card(card.id) for card in move.cards],
                                         cards_neg=[self.id_to_orig_card(card.id) for card in move.cards_neg]))
            elif isinstance(move, ResolvedDraw):
                log.append(move._replace(cards=[self.id_to_orig_card(card.id) for card in move.cards]))
        return log

    def hands_history(self):
        hands_history = []
        hands = [[] for _i in range(len(self.hands))]
        for move in self.log:
            if isinstance(move, (ResolvedPlay, ResolvedDiscard)):
                self.take_card_from_hand(hands[move.cur_player], move.card.id)
                if move.new_card:
                    hands[move.cur_player] += [self.id_to_orig_card(move.new_card.id)]
            elif isinstance(move, ResolvedClue):
                pass
            elif isinstance(move, ResolvedDraw):
                hands[move.cur_player] += [self.id_to_orig_card(card.id) for card in move.cards]
            else:
                assert False
            hands_history.append(hands[move.cur_player].copy())
        return hands_history

    def slots_history(self):
        slots_history = []
        slots = [0] * self.rules.suits
        for move in self.log:
            if isinstance(move, ResolvedPlay):
                if move.is_success:
                    slots[move.card.data.suit] += 1
            elif isinstance(move, (ResolvedDraw, ResolvedDiscard, ResolvedClue)):
                pass
            else:
                assert False
            slots_history.append(slots.copy())
        return slots_history
    
    def tokens_history(self):
        clues_history = []
        lives_history = []
        clues, lives = self.rules.max_tokens
        for move in self.log:
            if isinstance(move, ResolvedPlay):
                if move.is_success:
                    if move.card.data.rank == len(self.rules.ranks)-1 and clues < self.rules.max_tokens.clues:
                        clues += 1
                else:
                    lives -= 1
            elif isinstance(move, ResolvedDiscard):
                assert clues < self.rules.max_tokens.clues
                clues += 1
            elif isinstance(move, ResolvedClue):
                clues -= 1
            elif isinstance(move, ResolvedDraw):
                pass
            else:
                assert False
            clues_history.append(clues)
            lives_history.append(lives)
        return zip(clues_history, lives_history)

    def max_rank_history(self):
        discard = [[0 for _r in self.rules.ranks] for _s in range(self.rules.suits)]
        max_rank = [len(self.rules.ranks)] * self.rules.suits
        max_rank_history = []
        for move in self.log:
            if isinstance(move, (ResolvedPlay, ResolvedDiscard)):
                if isinstance(move, ResolvedPlay) and move.is_success:
                    pass
                else:
                    discard[move.card.data.suit][move.card.data.rank] += 1
                    if discard[move.card.data.suit][move.card.data.rank] == self.rules.ranks[move.card.data.rank]:
                        max_rank[move.card.data.suit] = min(max_rank[move.card.data.suit], int(move.card.data.rank))
            elif isinstance(move, (ResolvedDraw, ResolvedClue)):
                pass
            else:
                assert False
            max_rank_history.append(max_rank.copy())
        return max_rank_history

    def print_history(self, last='.', thin=False):
        suits = [Suit(s) for s in range(self.rules.suits)]
        ranks = self.rules.ranks
        f_str = '{:<50}{:<42}{:<17}{:<17}{:>2}{:>2}{:>4}  {}'
        titles1 = ['', str(self.end_mode), str(suits), str(ranks), '', '', '', '']
        titles2 = ['Move', 'Hand', 'Slots', 'max_rank', 'c', 'l', 's', 'note']
        if thin:
            f_str = '{:<50}'
        print(f_str.format(*titles1))
        print(f_str.format(*titles2))
        last_args = None
        for move, hand, slots, max_rank, (clues, lives), note in zip(
                self.log, self.hands_history(), self.slots_history(), self.max_rank_history(), 
                self.tokens_history(), self.notes):
            this_args = slots, max_rank, clues, lives, sum(slots)
            if last_args is None:
                last_args = this_args
            print(f_str.format(
                    move, hand, 
                    *[last if last and pr==cu else str(cu) for (pr, cu) in zip(last_args, this_args)], 
                    note))
            last_args = this_args
        if thin:
            self.describe()
        else:
            print("\nDiscard pile:")
            pprint(self.discard_pile)

    def describe(self):
        # deck_start
        d0 = 5
        deck_strs = list(map(str, self.deck_start[::-1]))[sum(map(len, self.hands_start)):]
        print("deck_start:")
        print('[' + '\n '.join([', '.join(deck_strs[i:i+d0]) for i in range(0, len(deck_strs), d0)]) + ']')
        # rest
        for attr in 'hands_start rules end_mode  log  hands slots tokens discard_pile score'.split():
            print(f'{attr}:')
            pprint(getattr(self, attr))
