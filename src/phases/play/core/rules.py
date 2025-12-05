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

        # Previously there was an early-return that allowed 'Pass' too eagerly
        # which caused the simulator to loop without any card plays.
        # We'll compute valid play cards first and return `Pass` only when
        # there are no other valid actions.

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
        
        # Get highest trump card on the table
        # Note: Cards.winner returns (card, index), we need the card
        table_winner_card, _ = Cards.winner(table, trump)
        
        # Get player's trump cards higher than the table's highest trump card
        # We need to check if table_winner_card is actually a trump or if it's just the current winner
        # Logic: if table_winner_card is trump, we need a higher trump.
        # If table_winner_card is NOT trump (e.g. lead suit), but we have to play trump (cut),
        # we need to play a trump higher than any existing trump on table (if any).
        
        # Let's refine "highest trump card on the table".
        # If no trumps on table, this is None (conceptually).
        # But table.winner(trump) returns the current winning card.
        
        # Filter hand for trumps that are strictly better than the current table winner
        # ONLY IF the current table winner is a trump card.
        
        # Filter for cards that beat the current winner
        player_winner_cards = [card for card in cards if card.beats(trump, table_winner_card)]

        # Check if player has cards of the lead suit
        player_table_lead_suit_cards = [card for card in cards if card.suit == table_lead_suit]

        # When player has cards of the lead suit
        if player_table_lead_suit_cards:
            # When leading suit is trump
            if (table_winner_card.is_trump(trump)):
                # If player has trump cards that are higher than the table's highest trump card
                # They MUST play a higher trump if possible.
                if player_winner_cards:
                    return player_winner_cards
                    
            # Play any card of the lead suit (if lead is not trump)
            return player_table_lead_suit_cards

        # Player has no cards of the lead suit
        
        # If partner is winning, we can play anything (no need to cut)
        # But wait, State is personal, do we know if partner is winning?
        # State.table is just a list of cards. We need to know who played what.
        # The standard Belote rule: "If your partner is master, you are not forced to cut."
        # However, the provided reference implementation didn't seem to check for partner.
        # It just said:
        # "Player has no cards of the lead suit and player has winner cards -> return player_winner_cards"
        # This implies forced cutting if you have a trump that can beat the current winner.
        
        # Let's stick to the provided logic for now:
        # "Player has no cards of the lead suit and player has winner cards"
        
        # We need to recalculate player_winner_cards for the "cutting" scenario.
        # If we are cutting, we must play a trump that beats the current winner (if current winner is trump).
        # If current winner is NOT trump, any trump beats it.
        
        # If we can beat the current winner with a trump, we must do so.
        if player_winner_cards:
             return player_winner_cards
                
        # Player has no trump cards (that can beat?), they can play any card
        return cards
