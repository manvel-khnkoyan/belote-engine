from typing import List
from .state import State
from .actions import Action, ActionPass, ActionPlayCard
from src.models.card import Card
from src.models.trump import Trump, TrumpMode
from src.utility.cards import Cards

class Rules:        
    def __init__(self):
        pass
    
    def actions(self, state: State) -> List[Action]:
        """Get list of valid actions the current player can take"""
        valid_actions = []

        # Get valid play cards
        valid_play_cards = self._valid_play_cards(state)
        if len(valid_play_cards) > 0:
            valid_actions.extend(ActionPlayCard(card) for card in valid_play_cards)

        # Other action types (declarations etc) can be added here

        # If no valid actions found, allow passing
        if not valid_actions:
            return [ActionPass()]

        return valid_actions

    def _valid_play_cards(self, state: State) -> List[Card]:
        """Get list of valid cards the current player can play"""
        cards = state.cards
        table = state.table
        trump = state.trump

        # Check if the current player has any cards
        if not cards:
            return []

        # When the player is the first to play, they can play any card
        if len(table) == 0:
            return cards
        
        # Get lead suit (suit of the first card played in this trick)
        table_lead_suit = table[0].suit
        
        # Check if player has cards of the lead suit
        player_table_lead_suit_cards = [card for card in cards if card.suit == table_lead_suit]

        # If player has cards of the lead suit, they MUST play one of them
        if player_table_lead_suit_cards:
            return player_table_lead_suit_cards

        # Player has no cards of the lead suit - must cut (play trump) if possible
        player_trump_cards = [card for card in cards if card.is_trump(trump)]
        
        # Get highest trump card on the table
        table_winner_card, _ = Cards.winner(table, trump)
        
        # If table winner is a trump, must play a trump higher than it
        if table_winner_card.is_trump(trump):
            player_winner_cards = [card for card in player_trump_cards if card.beats(trump, table_winner_card)]
            if player_winner_cards:
                return player_winner_cards
        
        # If player has any trump cards (and can't beat or winner isn't trump), play any trump
        if player_trump_cards:
            return player_trump_cards
        
        # No valid cards of lead suit or trump - play anything
        return cards
