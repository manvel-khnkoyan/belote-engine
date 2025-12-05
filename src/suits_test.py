import pytest
from suits import Suits

class TestSuitManager:
    
    def test_default_suits(self):
        """Test that default suits are returned without transformation"""
        assert Suits[0] == '♠'
        assert Suits[1] == '♥'
        assert Suits[2] == '♦'
        assert Suits[3] == '♣'
    
    def test_transform_with_simple_map(self):
        """Test transformation with a simple mapping function"""
        # Rotate suits by 1: 0->1, 1->2, 2->3, 3->0
        Suits.transform(lambda i: (i + 1) % 4)
        
        assert Suits[0] == '♥'  # Was ♠
        assert Suits[1] == '♦'  # Was ♥
        assert Suits[2] == '♣'  # Was ♦
        assert Suits[3] == '♠'  # Was ♣
    
    def test_transform_with_swap(self):
        """Test transformation that swaps suits"""
        # Swap: 0<->2, 1<->3
        swap_map = {0: 2, 1: 3, 2: 0, 3: 1}
        Suits.transform(lambda i: swap_map[i])
        
        assert Suits[0] == '♦'  # Was ♠
        assert Suits[1] == '♣'  # Was ♥
        assert Suits[2] == '♠'  # Was ♦
        assert Suits[3] == '♥'  # Was ♣
    
    def test_transform_with_reverse(self):
        """Test transformation that reverses suit order"""
        Suits.transform(lambda i: 3 - i)
        
        assert Suits[0] == '♣'  # Was ♠
        assert Suits[1] == '♦'  # Was ♥
        assert Suits[2] == '♥'  # Was ♦
        assert Suits[3] == '♠'  # Was ♣
    
    
    def test_multiple_transforms(self):
        """Test that setting a new transform replaces the old one"""
        Suits.transform(lambda i: (i + 1) % 4)
        assert Suits[0] == '♥'
        
        Suits.transform(lambda i: (i + 2) % 4)
        assert Suits[0] == '♦'  # New transform applied
    
    def test_identity_transform(self):
        """Test transform that doesn't change anything"""
        Suits.transform(lambda i: i)
        
        assert Suits[0] == '♠'
        assert Suits[1] == '♥'
        assert Suits[2] == '♦'
        assert Suits[3] == '♣'
