from typing import List
from .state import State
from .actions import Action, ActionPlayCard
from src.models.card import Card
from src.models.trump import Trump, TrumpMode
from src.models.collection import Collection

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
        
        return valid_actions

    def _valid_play_cards(self, state: State) -> List[Card]:
        """Get list of valid cards the current player can play"""
        cards = state.cards
        table = Collection(state.table)
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
        # Note: Collection.winner returns (card, index), we need the card
        table_winner_card, _ = table.winner(trump)
        
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
        
        current_winner_is_trump = (
            trump.mode == TrumpMode.Regular and 
            table_winner_card.suit == trump.suit
        ) or (trump.mode == TrumpMode.AllTrump)

        player_winner_cards = []
        if current_winner_is_trump:
             player_winner_cards = [
                card for card in cards 
                if (trump.mode == TrumpMode.Regular and card.suit == trump.suit) or (trump.mode == TrumpMode.AllTrump)
                and card.beats(trump, table_winner_card)
            ]
        else:
             # If winner is not trump, any trump is a "winner card" in context of cutting
             # But we only need this list if we are required to play trump.
             player_winner_cards = [
                card for card in cards
                if (trump.mode == TrumpMode.Regular and card.suit == trump.suit) or (trump.mode == TrumpMode.AllTrump)
             ]

        # Check if player has cards of the lead suit
        player_table_lead_suit_cards = [card for card in cards if card.suit == table_lead_suit]

        # When player has cards of the lead suit
        if player_table_lead_suit_cards:
            # When leading suit is trump
            if (trump.mode == TrumpMode.Regular and table_lead_suit == trump.suit) or (trump.mode == TrumpMode.AllTrump):
                # If player has trump cards that are higher than the table's highest trump card
                # They MUST play a higher trump if possible.
                if player_winner_cards:
                    return player_winner_cards
                
                # If they can't beat it, they still must play trump (lead suit)
                return player_table_lead_suit_cards
                    
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
        
        # Re-evaluating player_winner_cards logic from reference:
        # player_winner_cards = [card for card in hand if card.higher_than(table_winner_card, self.trump)]
        # This implies strictly beating the current table winner.
        
        player_can_beat_current_winner = [card for card in cards if card.beats(trump, table_winner_card)]
        
        # But we only care about this if we are cutting (playing trump).
        # So we filter for trumps.
        player_trumps_that_beat_winner = [
            c for c in player_can_beat_current_winner 
            if (trump.mode == TrumpMode.Regular and c.suit == trump.suit) or (trump.mode == TrumpMode.AllTrump)
        ]
        
        if player_trumps_that_beat_winner:
             return player_trumps_that_beat_winner
             
        # If we can't beat the winner with a trump (or don't have higher trumps),
        # but we still have trumps, do we have to play them?
        # Reference: "Player has no trump cards, they can play any card"
        # This implies if they HAVE trumps (even lower ones), they might be forced?
        # The reference logic:
        # if player_winner_cards: return player_winner_cards
        # return self.deck[self.next_player]
        
        # This suggests:
        # 1. Follow suit (handled above).
        # 2. If can't follow suit, and can beat current winner (with trump), do it.
        # 3. Else play anything.
        
        # Wait, standard Belote rules say you MUST cut if you can't follow suit, unless partner is winning.
        # If you can't over-cut, you still must play trump if opponent is winning?
        # The reference code is a bit ambiguous: "Player has no trump cards, they can play any card".
        # It checks `if player_winner_cards: return player_winner_cards`.
        # If `player_winner_cards` is empty (cannot beat), it returns ALL cards.
        # This implies you don't have to undertrump?
        
        # Let's strictly follow the reference implementation provided in the prompt.
        
        # Reference:
        # player_winner_cards = [card for card in hand if card.higher_than(table_winner_card, self.trump)]
        # Note: higher_than likely means beats().
        
        player_winner_cards_ref = [c for c in cards if c.beats(trump, table_winner_card)]
        
        if player_winner_cards_ref:
             # But wait, we only cut with TRUMP.
             # If I have an Ace of another suit, it doesn't "beat" the table winner in the context of the game rules for valid moves,
             # unless it's the lead suit (which we already handled).
             # So `player_winner_cards` in the reference MUST imply trumps (or cards that become master).
             # Since we are in the "can't follow suit" branch, the only way to beat the current winner is with a Trump.
             
             # So we filter for trumps.
             player_trumps_that_beat = [
                 c for c in player_winner_cards_ref
                 if (trump.mode == TrumpMode.Regular and c.suit == trump.suit) or (trump.mode == TrumpMode.AllTrump)
             ]
             
             if player_trumps_that_beat:
                 return player_trumps_that_beat
                
        # Player has no trump cards (that can beat?), they can play any card
        return cards
