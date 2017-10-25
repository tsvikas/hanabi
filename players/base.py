import random
from pprint import pprint
from game import Clue, Play, Discard, Move

from game import Card, Tokens, Rules, Suit, Rank
from typing import NamedTuple, List, Tuple, Callable


def make_io_player(name: str) -> Callable:
    """
    Aur's basic IO player generator

    usage:
        h = Hanabi([make_io_player('Aur'), make_io_player('Ofer')])
    """
    def io_player(state: None, log: List[NamedTuple], hands: List[List[Card]],
                  rules: Rules, tokens: Tokens, slots: List[int],
                  discard_pile: List[List[int]]) -> Tuple[None, Move, str]:
        print(f"{name}'s turn")
        pprint(log[-len(hands):])
        pprint(hands)
        pprint(tokens)
        pprint(slots)
        pprint(discard_pile)
        print('What will it be?')
        print('[c]lue <player><type><n>\t\t[p]lay <card>\t\t[d]iscard <card>')
        move = input().split()
        try:
            if move[0] == 'c':
                _, (player, clue_type, param) = move
                return state, Clue.create(
                    int(player),
                    {'s': 'suit', 'n': 'rank'}[clue_type],
                    {'s': Suit, 'n': Rank}[clue_type].from_str(param)), ''
            elif move[0] == 'p':
                _, card_id = move
                return state, Play.create(int(card_id)), ''
            elif move[0] == 'd':
                _, card_id = move
                return state, Discard.create(int(card_id)), ''
            else:
                raise ValueError("not a valid move type")
        except (ValueError, KeyError):
            print('illegal move')
            return io_player(state, log, hands, rules, tokens, slots, discard_pile)
    return io_player


def random_player(state: None, log: List[NamedTuple], hands: List[List[Card]],
                  rules: Rules, tokens: Tokens, slots: List[int],
                  discard_pile: List[List[int]]) -> Tuple[None, Move, str]:
    """
    Tsvika and Ofer's random player

    Usage:
        h = Hanabi([random_player] * 3)
    """
    my_id = len(log) % len(hands)

    possible_actions = []
    if tokens.lives > 1:
        possible_actions.append(Play)

    if tokens.clues > 0:
        possible_actions.append(Clue)

    if tokens.clues < rules.max_tokens.clues:
        possible_actions.append(Discard)

    action = random.choice(possible_actions)

    if action == Play:
        return state, Play.create(random.choice(hands[my_id]).id), ''
    if action == Discard:
        return state, Discard.create(random.choice(hands[my_id]).id), ''
    if action == Clue:
        player = random.choice([i for i in range(len(hands)) if i != my_id])
        clue_type = random.choice(['suit', 'rank'])
        return state, Clue.create(
            player, clue_type,
            getattr(random.choice(hands[player]).data, clue_type)), ''
