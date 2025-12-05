import pytest
from deck import Deck
from src.models.card import Card

class TestDeck:
    def test_create_returns_four_hands(self):
        """Test that create returns 4 hands"""
        hands = Deck.create()
        assert len(hands) == 4
    
    def test_create_each_hand_has_eight_cards(self):
        """Test that each hand has exactly 8 cards"""
        hands = Deck.create()
        for hand in hands:
            assert len(hand) == 8
    
    def test_create_total_cards_is_32(self):
        """Test that total number of cards is 32"""
        hands = Deck.create()
        total_cards = sum(len(hand) for hand in hands)
        assert total_cards == 32
    
    def test_create_all_cards_are_unique(self):
        """Test that all cards are unique (no duplicates)"""
        hands = Deck.create()
        all_cards = [card for hand in hands for card in hand]
        
        # Check uniqueness by converting to set of (suit, rank) tuples
        card_tuples = [(card.suit, card.rank) for card in all_cards]
        assert len(card_tuples) == len(set(card_tuples))
    
    def test_create_has_all_suits(self):
        """Test that deck contains all 4 suits"""
        hands = Deck.create()
        all_cards = [card for hand in hands for card in hand]
        suits = set(card.suit for card in all_cards)
        assert suits == {0, 1, 2, 3}
    
    def test_create_has_all_ranks(self):
        """Test that deck contains all 8 ranks"""
        hands = Deck.create()
        all_cards = [card for hand in hands for card in hand]
        ranks = set(card.rank for card in all_cards)
        assert ranks == {0, 1, 2, 3, 4, 5, 6, 7}
    
    def test_create_each_suit_has_eight_cards(self):
        """Test that each suit appears exactly 8 times"""
        hands = Deck.create()
        all_cards = [card for hand in hands for card in hand]
        
        suit_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for card in all_cards:
            suit_counts[card.suit] += 1
        
        for count in suit_counts.values():
            assert count == 8
    
    def test_create_each_rank_appears_four_times(self):
        """Test that each rank appears exactly 4 times (once per suit)"""
        hands = Deck.create()
        all_cards = [card for hand in hands for card in hand]
        
        rank_counts = {i: 0 for i in range(8)}
        for card in all_cards:
            rank_counts[card.rank] += 1
        
        for count in rank_counts.values():
            assert count == 4
    
    def test_create_with_shuffle_true_is_random(self):
        """Test that shuffled decks are different"""
        hands1 = Deck.create(shuffle=True)
        hands2 = Deck.create(shuffle=True)
        
        # Flatten hands
        cards1 = [card for hand in hands1 for card in hand]
        cards2 = [card for hand in hands2 for card in hand]
        
        # Compare if orders are different
        # (Very unlikely to be the same if shuffled)
        same_order = all(c1.suit == c2.suit and c1.rank == c2.rank 
                        for c1, c2 in zip(cards1, cards2))
        
        # They should be different (with very high probability)
        # Note: This could theoretically fail, but probability is 1/32! which is astronomically small
        assert not same_order
    
    def test_create_with_shuffle_false_is_ordered(self):
        """Test that unshuffled decks are in order"""
        hands1 = Deck.create(shuffle=False)
        hands2 = Deck.create(shuffle=False)
        
        # Flatten hands
        cards1 = [card for hand in hands1 for card in hand]
        cards2 = [card for hand in hands2 for card in hand]
        
        # Compare if orders are the same
        same_order = all(c1.suit == c2.suit and c1.rank == c2.rank 
                        for c1, c2 in zip(cards1, cards2))
        
        assert same_order
    
    def test_create_unshuffled_order(self):
        """Test that unshuffled deck follows expected order (suits then ranks)"""
        hands = Deck.create(shuffle=False)
        all_cards = [card for hand in hands for card in hand]
        
        # Expected order: suit 0 ranks 0-7, suit 1 ranks 0-7, etc.
        expected_index = 0
        for suit in range(4):
            for rank in range(8):
                card = all_cards[expected_index]
                assert card.suit == suit
                assert card.rank == rank
                expected_index += 1
    
    def test_create_hands_are_consecutive(self):
        """Test that hands contain consecutive cards from the deck"""
        hands = Deck.create(shuffle=False)
        
        # Hand 0 should be cards 0-7
        # Hand 1 should be cards 8-15
        # Hand 2 should be cards 16-23
        # Hand 3 should be cards 24-31
        
        all_cards = [card for hand in hands for card in hand]
        
        for i, hand in enumerate(hands):
            start_idx = i * 8
            for j, card in enumerate(hand):
                assert card == all_cards[start_idx + j]
