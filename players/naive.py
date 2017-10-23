from game import Clue, Play, Discard, ResolvedClue

from game import Card, Tokens, Rules
from typing import NamedTuple, List, Tuple


def naive_player(state: None, log: List[NamedTuple], hands: List[List[Card]],
                 rules: Rules, tokens: Tokens, slots: List[int],
                 discard_pile: List[List[int]]) -> Tuple[None, NamedTuple]:
    """
    Zvika and Ofer's naive player
    """
    my_id = len(log) % len(hands)
    my_card_ids = [card.id for card in hands[my_id]]

    hinted_cards = set()
    for move in log[-len(hands):]:
        if isinstance(move, ResolvedClue) and move.player == my_id:
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
            for card in hands[player]:
                if slots[card.data.suit] != card.data.rank:
                    player_suits -= set([card.data.suit])
                    player_ranks -= set([card.data.rank])
            if player_ranks:
                # its better to go up then sideways
                return state, Clue.create(player, 'rank', max(player_ranks)), ''
            if player_suits:
                return state, Clue.create(player, 'suit', player_suits.pop()), ''

    # Its better to discard then playing like an idiot
    if tokens.clues < rules.max_tokens.clues:
        return state, Discard.create(min(my_card_ids)), ''

    # If all else fails, play like an idiot
    return state, Play.create(max(my_card_ids)), 'idiot'
