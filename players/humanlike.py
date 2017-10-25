from collections import namedtuple
from game import Clue, Play, Discard, ResolvedClue

CardInfo = namedtuple('CardInfo', 'positive negative')
Info = namedtuple('Info', 'suit rank')
PossibleClue = namedtuple('PossibleClue', 'player card type')


def humanlike_player(state, log, hands, rules, tokens, slots, discard_pile):
    """
    Ofer's humanlike player
    """

    def add_card_to_state(given_state, card_id):
        if card_id not in given_state:
            given_state[card_id] = CardInfo(
                Info(None, None),
                Info([True for _ in range(rules.suits)], [True for _ in rules.ranks])
            )

    def update_cards(cards, player_id=None, clue=None):
        if player_id is None:
            player_id = my_id

        hinted_cards = set()
        _log = log[-len(hands):]
        if clue is not None:
            _log.append(clue)

        for move in _log:
            if isinstance(move, ResolvedClue):
                if move.player == player_id:
                    for card in move.cards:
                        hinted_cards.add(card.id)

                card_ids_in_hint = set()
                for card in move.cards:
                    add_card_to_state(cards, card.id)
                    card_ids_in_hint.add(card.id)
                    cards[card.id] = cards[card.id]._replace(
                        positive=cards[card.id].positive._replace(**{move.type: move.param}))
                for card in hands[player_id]:
                    if card.id not in card_ids_in_hint:
                        add_card_to_state(cards, card.id)
                        new_negative = getattr(cards[card.id].negative, move.type)
                        new_negative[move.param] = False
                        cards[card.id] = cards[card.id]._replace(
                            negative=cards[card.id].negative._replace(**{move.type: new_negative}))

        # Consolidate negatives in hand
        for card_id in hinted_cards:
            if cards[card_id].negative.suit.count(True) == 1:
                cards[card.id]._replace(positive=cards[card.id].positive._replace(
                    suit=[i for i, v in enumerate(cards[card_id].negative.suit) if v]))
            if cards[card_id].negative.rank.count(True) == 1:
                cards[card.id]._replace(positive=cards[card.id].positive._replace(
                    rank=[i for i, v in enumerate(cards[card_id].negative.rank) if v]))

        return cards, hinted_cards

    def get_max_rank_in_suit(suit, _slots, _discard_pile):
        max_rank_in_suit = None
        for rank in range(len(rules.ranks)):
            left_in_rank = rules.ranks[rank] - _discard_pile[suit][rank]
            if rank >= _slots[suit] and left_in_rank == 0:
                max_rank_in_suit = rank
                break

        return max_rank_in_suit

    def is_playable_suit(suit, _slots, _discard_pile):
        if _slots[suit] > len(rules.ranks):
            return False

        max_rank_in_suit = get_max_rank_in_suit(suit, _slots, _discard_pile)
        if max_rank_in_suit is not None and max_rank_in_suit < _slots[suit]:
            return False

        return True

    def should_play_card(cards, cards_in_hand, hinted_cards, _slots=None, _discard_pile=None):
        if _slots is None:
            _slots = slots
        if _discard_pile is None:
            _discard_pile = discard_pile

        hinted_cards = hinted_cards.intersection(cards_in_hand)
        definate_cards_to_play = set()
        cards_to_play = set()
        for card_id in cards_in_hand:
            add_card_to_state(cards, card_id)
            if cards[card_id].positive.suit is not None and cards[card_id].positive.rank is not None:
                if is_play_legal(cards[card_id].positive.suit, cards[card_id].positive.rank, _slots):
                    definate_cards_to_play.add(card_id)

            if card_id in sorted(hinted_cards):
                if cards[card_id].positive.rank is not None and cards[card_id].positive.suit is None and any(
                        [is_playable_suit(suit, _slots, _discard_pile) and cards[card_id].positive.rank == _slots[suit]
                         for suit in range(len(_slots))]):
                    cards_to_play.add(card_id)
                if cards[card_id].positive.suit is not None and cards[card_id].positive.rank is None \
                        and is_playable_suit(cards[card_id].positive.suit, _slots, _discard_pile):
                    cards_to_play.add(card_id)

        if definate_cards_to_play:  # its better to go up than go sideways!
            highest_rank = 0
            cards_in_highest_rank = set()
            for card_id in definate_cards_to_play:
                if cards[card_id].positive.rank > highest_rank:
                    highest_rank = cards[card_id].positive.rank
                    cards_in_highest_rank = set()
                if cards[card_id].positive.rank == highest_rank:
                    cards_in_highest_rank.add(card_id)
            return sorted(cards_in_highest_rank)[-1]  # play newest card

        if cards_to_play:
            return sorted(cards_to_play)[-1]  # play newest card

        return None

    def what_will_player_play(cards, hand, player_id, clue, _slots, _discard_pile):
        cards, _hinted_cards = update_cards(cards, player_id, clue)
        card_id = should_play_card(cards, [card.id for card in hand], _hinted_cards, _slots, _discard_pile)
        if card_id is not None:
            card = [card for card in hand if card.id == card_id][0]
            legal = is_play_legal(card.data.suit, card.data.rank, _slots)
            return cards, legal, card_id
        else:
            return cards, None, None

    def is_play_legal(suit, rank, _slots):
        return _slots[suit] == rank

    def create_clue(my_id, _player, clue_type, param):
        cards = [card for card in hands[_player] if getattr(card.data, clue_type) == param]
        cards_neg = [card for card in hands[_player] if getattr(card.data, clue_type) != param]
        return ResolvedClue.create(my_id, _player, clue_type, param, cards, cards_neg)

    # Start

    if state is None:
        state = {}

    my_id = len(log) % len(hands)
    state, state_actions = update_cards(state)

    my_card_ids = [card.id for card in hands[my_id]]

    card_to_play = should_play_card(state, my_card_ids, state_actions, slots, discard_pile)

    if card_to_play is not None:  # Its better to play than hint
        return state, Play.create(card_to_play), 'Played card'

    if tokens.clues > 0:  # Its better to hint than discard
        foreseen_slots = list(slots)
        foreseen_state = dict(state)

        for i in range(len(hands) - 1):
            player = (my_id + i + 1) % len(hands)
            foreseen_state, is_legal, play = what_will_player_play(
                foreseen_state, hands[player], player, None, foreseen_slots, discard_pile)
            player_state, player_hinted = update_cards(foreseen_state, player)
            player_play = should_play_card(state, [card.id for card in hands[player]], player_hinted)
            if player_play is not None:
                card = [card for card in hands[player] if card.id == player_play][0]

                if is_play_legal(card.data.suit, card.data.rank, slots):
                    foreseen_slots[card.data.suit] = card.data.rank
                    continue
                else:  # try and rectify stupidity
                    for card in hands[player]:
                        suit_clue = create_clue(my_id, player, 'suit', card.data.suit)
                        _, is_legal, play = what_will_player_play(
                            dict(foreseen_state), hands[player], player, suit_clue, foreseen_slots, discard_pile)
                        if is_legal or play is None:
                            return state, Clue.create(player, 'suit', card.data.suit), 'Gave hint against stupid play'

                        rank_clue = create_clue(my_id, player, 'rank', card.data.rank)
                        _, is_legal, play = what_will_player_play(
                            dict(foreseen_state), hands[player], player, rank_clue, foreseen_slots, discard_pile)
                        if is_legal or play is None:
                            return state, Clue.create(player, 'rank', card.data.rank), 'Gave hint against stupid play'

            good_clues = set()
            for card in hands[player]:
                if slots[card.data.suit] == card.data.rank:
                    suit_clue = create_clue(my_id, player, 'suit', card.data.suit)
                    _, is_legal, play = what_will_player_play(
                        dict(foreseen_state), hands[player], player, suit_clue, foreseen_slots, discard_pile)
                    if is_legal and play == card.id:
                        good_clues.add(PossibleClue(player=player, card=card, type='suit'))

                    rank_clue = create_clue(my_id, player, 'rank', card.data.rank)
                    _, is_legal, play = what_will_player_play(
                        dict(foreseen_state), hands[player], player, rank_clue, foreseen_slots, discard_pile)
                    if is_legal and play == card.id:
                        good_clues.add(PossibleClue(player=player, card=card, type='rank'))

            if good_clues:
                # make sure highest card possible is played
                highest_rank = 0
                given_clue = None
                for clue in good_clues:
                    if given_clue is None:
                        given_clue = clue
                    if clue.card.data.rank > highest_rank:
                        highest_rank = clue.card.data.rank
                        given_clue = clue
                return state, Clue.create(given_clue.player, given_clue.type,
                                          getattr(given_clue.card.data, given_clue.type)), 'Gave actionable clue'

    if tokens.clues < rules.max_tokens.clues:  # Its better to discard then playing like an idiot
        protected_cards = set()
        for card_id in my_card_ids:
            # Throw away useless cards
            if state[card_id].positive.suit is not None and not is_playable_suit(state[card_id].positive.suit, slots,
                                                                                 discard_pile):
                return state, Discard.create(card_id), 'Discarded unplayable suit'

            if state[card_id].positive.rank is not None and all(
                    [slot < state[card_id].positive.rank for slot in slots]):
                return state, Discard.create(card_id), 'Discarded Unplayable rank'

            if state[card_id].positive.suit is not None and state[card_id].positive.rank is not None:
                if slots[state[card_id].positive.suit] < state[card_id].positive.rank:
                    return state, Discard.create(card_id), 'Discarded unplayable known card'

                # Don't throw away lone copies
                available_copies = rules.ranks[state[card_id].positive.rank]
                discarded_copies = discard_pile[state[card_id].positive.suit][state[card_id].positive.rank]
                if available_copies - discarded_copies == 1:
                    protected_cards.add(card_id)

            # Don't throw away 5s
            if state[card_id].positive.rank is not None:
                available_copies = rules.ranks[state[card_id].positive.rank]
                if available_copies == 1:
                    protected_cards.add(card_id)

        throwaways = set(my_card_ids) - protected_cards
        if throwaways:
            return state, Discard.create(min(throwaways)), 'Discarded unprotected card'

        return state, Discard.create(min(my_card_ids)), 'Discarded oldest card'

    if tokens.clues > 0:
        # give random clue to the player playing before you so the other players may fix it
        player = (my_id - 1) % len(hands)
        if hands[player]:
            highest_rank_in_hand = sorted([card.data.rank for card in hands[player]])[-1]
            return state, Clue.create(player, 'rank', highest_rank_in_hand), 'Gave random clue'

    # If all else fails, play like an idiot
    return state, Play.create(max(my_card_ids)), 'Played random card'
