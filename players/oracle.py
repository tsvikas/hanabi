from game import Play, Discard, Clue
from game import Card, Tokens, Rules
from typing import NamedTuple, List, Tuple


def oracle_player(state: None, log: List[NamedTuple], hands: List[List[Card]],
                  rules: Rules, tokens: Tokens, slots: List[int],
                  discard_pile: List[List[int]]) -> Tuple[None, NamedTuple]:
    """
    Zvika and Ofer's oracle player
    """
    my_id = len(log) % len(hands)
    my_hand = hands[my_id]
    if my_hand[0].data is None:
        raise RuntimeError("I need to be omniscient")

    # play something playable
    playable_card = None
    for card in my_hand:
        if slots[card.data.suit] == card.data.rank:
            if playable_card is None or playable_card.data.rank < card.data.rank:
                playable_card = card
    if playable_card is not None:
        return state, Play.create(playable_card.id)

    def get_card_to_discard():
        # discard already played
        for card in my_hand:
            if slots[card.data.suit] > card.data.rank:
                return card.id
        # discard unreachable
        for suit in range(rules.suits):
            max_rank_in_suit = None
            for rank in range(len(rules.ranks)):
                left_in_rank = rules.ranks[rank] - discard_pile[suit][rank]
                if rank >= slots[suit] and left_in_rank == 0:
                    max_rank_in_suit = rank
                    break
            if max_rank_in_suit:
                for card in my_hand:
                    if card.data.suit == suit and card.data.rank > max_rank_in_suit:
                        return card.id
        # discard duplicates in own hand
        knowns = [card.data for card in my_hand]
        if len(set(knowns)) < len(knowns):
            for i, known in enumerate(knowns):
                for known2 in knowns[i+1:]:
                    if known == known2:
                        return my_hand[i].id
        # discard duplicates with others
        knowns = [card.data for card in my_hand]
        for hand in hands[:my_id]+hands[my_id+1:]:
            knowns2 = [card.data for card in hand]
            if len(set(knowns+knowns2)) < len(knowns)+len(set(knowns2)):
                for i, known in enumerate(knowns):
                    for known2 in knowns2:
                        if known == known2:
                            return my_hand[i].id

    # discard something discardable
    if tokens.clues < rules.max_tokens.clues:
        card = get_card_to_discard()
        if card is not None:
            return state, Discard.create(card)

    # nothing useful to do
    # try to pass with useless clue
    if tokens.clues > 0:
        player = (my_id + 1) % len(hands)
        if hands[player]:
            return state, Clue.create(player, 'suit', hands[player][0].data.suit)

    # try to pass with false play
    if tokens.lives > 1:
        card = get_card_to_discard()
        if card is not None:
            return state, Play.create(card)

    # you have to throw something useful. try the farthest from the suit
    # look for an expandable card
    diff = None
    throw = None
    for card in my_hand:
        card_diff = card.data.rank - slots[card.data.suit]
        if diff is None or card_diff > diff:
            if rules.ranks[card.data.rank] - discard_pile[card.data.suit][card.data.rank] > 1:
                diff = card_diff
                throw = card
    # look for a non expandable card, if you must (BOO!)
    if diff is None:
        for card in my_hand:
            card_diff = card.data.rank - slots[card.data.suit]
            if diff is None or card_diff > diff:
                diff = card_diff
                throw = card

    # throw by discard
    if tokens.clues < rules.max_tokens.clues:
        return state, Discard.create(throw.id)

    # throw by false play
    return state, Play.create(throw.id)
