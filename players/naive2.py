from game import Clue, Play, Discard, ResolvedClue

from game import Card, Tokens, Rules
from typing import NamedTuple, List, Tuple


def naive2_player(state: None, log: List[NamedTuple], hands: List[List[Card]],
                 rules: Rules, tokens: Tokens, slots: List[int],
                 discard_pile: List[List[int]]) -> Tuple[None, NamedTuple]:
    """
    Zvika and Ofer's naive player
    """
    my_id = len(log) % len(hands)
    my_card_ids = [card.id for card in hands[my_id]]

    hinted_cards = set()
    for move in log[-len(hands):]:
        if isinstance(move, ResolvedClue):
            if move.player == my_id:
                hinted_cards = hinted_cards.union(card.id for card in move.cards)

    # Its better to play than hint
    if hinted_cards:
        play_card = max(hinted_cards)
        return state, Play.create(play_card), ''

    # Its better to hint than discard
    if tokens.clues > 0:
        for i in range(len(hands) - 1):
            player = (my_id + i + 1) % len(hands)
            player_suits = set([card.data.suit for card in hands[player]])
            player_ranks = set([card.data.rank for card in hands[player]])
            playable_suits = set()
            playable_ranks = set()
            for suit in player_suits:
                card = max([card for card in hands[player] if card.data.suit == suit])
                if slots[card.data.suit] == card.data.rank:
                    playable_suits.add(card.data.suit)
            for rank in player_ranks:
                card = max([card for card in hands[player] if card.data.rank == rank])
                if slots[card.data.suit] == card.data.rank:
                    playable_ranks.add(card.data.rank)

            # its better to go up then sideways
            clue = None
            clue_rank = 0
            if playable_ranks:
                clue = Clue.create(player, 'rank', max(playable_ranks))
                clue_rank = max(player_ranks)
            for suit in playable_suits:
                if slots[suit] > clue_rank:
                    clue = Clue.create(player, 'suit', suit)
                    clue_rank = slots[suit]
            if clue:
                return state, clue, ''

    # Its better to discard then playing like an idiot
    if tokens.clues < rules.max_tokens.clues:
        return state, Discard.create(min(my_card_ids)), ''

    # If all else fails, play like an idiot
    return state, Play.create(max(my_card_ids)), 'idiot'
