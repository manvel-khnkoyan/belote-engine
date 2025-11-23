import pytest
from card import Card
from trump import Trump, TrumpMode
from set import Set

class TestSet:
    def test_set_values_sequences(self):
        trump = Trump(TrumpMode.Regular, 0)
        
        # Tierce
        tierce = Set(Set.Tierce, [Card(0, 0), Card(0, 1), Card(0, 2)])
        assert tierce.value(trump) == 20
        
        # Quarte
        quarte = Set(Set.Quarte, [Card(0, 0), Card(0, 1), Card(0, 2), Card(0, 3)])
        assert quarte.value(trump) == 50
        
        # Quinte
        quinte = Set(Set.Quinte, [Card(0, 0), Card(0, 1), Card(0, 2), Card(0, 3), Card(0, 4)])
        assert quinte.value(trump) == 100

    def test_set_values_quartets_regular(self):
        trump = Trump(TrumpMode.Regular, 0)
        
        # 4 Jacks -> 200
        jacks = Set(Set.Quartet, [Card(0, 4), Card(1, 4), Card(2, 4), Card(3, 4)])
        assert jacks.value(trump) == 200
        
        # 4 Nines -> 140
        nines = Set(Set.Quartet, [Card(0, 2), Card(1, 2), Card(2, 2), Card(3, 2)])
        assert nines.value(trump) == 140
        
        # 4 Aces -> 100
        aces = Set(Set.Quartet, [Card(0, 7), Card(1, 7), Card(2, 7), Card(3, 7)])
        assert aces.value(trump) == 100
        
        # 4 Kings -> 100
        kings = Set(Set.Quartet, [Card(0, 6), Card(1, 6), Card(2, 6), Card(3, 6)])
        assert kings.value(trump) == 100
        
        # 4 Sevens -> 0
        sevens = Set(Set.Quartet, [Card(0, 0), Card(1, 0), Card(2, 0), Card(3, 0)])
        assert sevens.value(trump) == 0

    def test_set_values_quartets_no_trump(self):
        trump = Trump(TrumpMode.NoTrump, None)
        
        # 4 Jacks -> 100
        jacks = Set(Set.Quartet, [Card(0, 4), Card(1, 4), Card(2, 4), Card(3, 4)])
        assert jacks.value(trump) == 100
        
        # 4 Nines -> 0
        nines = Set(Set.Quartet, [Card(0, 2), Card(1, 2), Card(2, 2), Card(3, 2)])
        assert nines.value(trump) == 0
        
        # 4 Aces -> 190
        aces = Set(Set.Quartet, [Card(0, 7), Card(1, 7), Card(2, 7), Card(3, 7)])
        assert aces.value(trump) == 190

    def test_beats_quartet_vs_sequence(self):
        trump = Trump(TrumpMode.Regular, 0)
        
        quartet = Set(Set.Quartet, [Card(0, 6)]*4) # 4 Kings
        tierce = Set(Set.Tierce, [Card(0, 0), Card(0, 1), Card(0, 2)])
        
        # Quartet beats sequence
        assert quartet.beats(trump, tierce) == True
        assert tierce.beats(trump, quartet) == False

    def test_beats_quartet_vs_quartet(self):
        trump = Trump(TrumpMode.Regular, 0)
        
        jacks = Set(Set.Quartet, [Card(0, 4)]*4) # 200
        aces = Set(Set.Quartet, [Card(0, 7)]*4) # 100
        
        assert jacks.beats(trump, aces) == True
        assert aces.beats(trump, jacks) == False

    def test_beats_sequence_vs_sequence_trump(self):
        # Spades trump
        trump = Trump(TrumpMode.Regular, 0)
        
        # Tierce in Spades (Trump)
        tierce_trump = Set(Set.Tierce, [Card(0, 0), Card(0, 1), Card(0, 2)])
        
        # Tierce in Hearts (Non-Trump)
        tierce_hearts = Set(Set.Tierce, [Card(1, 0), Card(1, 1), Card(1, 2)])
        
        # Trump sequence beats non-trump sequence
        assert tierce_trump.beats(trump, tierce_hearts) == True
        assert tierce_hearts.beats(trump, tierce_trump) == False

    def test_beats_sequence_vs_sequence_rank(self):
        trump = Trump(TrumpMode.Regular, 0)
        
        # Tierce to Ace (Spades)
        tierce_high = Set(Set.Tierce, [Card(0, 5), Card(0, 6), Card(0, 7)])
        
        # Tierce to King (Spades)
        tierce_low = Set(Set.Tierce, [Card(0, 4), Card(0, 5), Card(0, 6)])
        
        assert tierce_high.beats(trump, tierce_low) == True
        assert tierce_low.beats(trump, tierce_high) == False

    def test_beats_sequence_vs_sequence_length(self):
        trump = Trump(TrumpMode.Regular, 0)
        
        quarte = Set(Set.Quarte, [Card(0, 0), Card(0, 1), Card(0, 2), Card(0, 3)])
        tierce = Set(Set.Tierce, [Card(0, 5), Card(0, 6), Card(0, 7)]) # Higher rank but shorter
        
        assert quarte.beats(trump, tierce) == True
        assert tierce.beats(trump, quarte) == False

    def test_extract(self):
        # 7, 8, 9 of Spades (Tierce)
        # 4 Jacks (Quartet)
        cards = [
            Card(0, 0), Card(0, 1), Card(0, 2), # Spades 7, 8, 9
            Card(1, 4), Card(2, 4), Card(3, 4), Card(0, 4) # 4 Jacks
        ]
        
        sets = Set.extract(cards)
        
        # Should find 1 Tierce and 1 Quartet
        assert len(sets) == 2
        
        types = sorted([s.type for s in sets])
        assert types == [Set.Tierce, Set.Quartet]
