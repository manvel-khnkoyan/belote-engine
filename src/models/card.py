from trump import Trump, TrumpMode
from const import Suits, Ranks

class Card:
    def __init__(self, suit: int, rank: int):
        assert 0 <= suit <= 3 and 0 <= rank <= 7
        self.suit, self.rank = suit, rank
    
    def __iter__(self):
        return iter((self.suit, self.rank))
    
    def __repr__(self):
        return f"{Ranks[self.rank]}{Suits[self.suit]}"

    def beats(self, trump: Trump, next: "Card") -> bool:
        if self.suit != next.suit:
            return trump.mode == TrumpMode.Regular and self.suit == trump.suit
        
        diff = self.value(trump) - next.value(trump)
        return diff > 0 if diff != 0 else self.rank > next.rank
    
    def value(self, trump: Trump) -> int:
        # No Trump mode
        if trump.mode == TrumpMode.NoTrump:
            match self.rank:
                case 7:  # ACE
                    return 19
                case 3:  # TEN
                    return 10
                case 6:  # KING
                    return 4
                case 5:  # QUEEN
                    return 3
                case 4:  # JACK
                    return 2
                case 2:  # NINE
                    return 0
                case 1:  # EIGHT
                    return 0
                case 0:  # SEVEN
                    return 0
        
        # All Trump mode
        elif trump.mode == TrumpMode.AllTrump:
            match self.rank:
                case 4:  # JACK
                    return 20
                case 2:  # NINE
                    return 14
                case 7:  # ACE
                    return 11
                case 3:  # TEN
                    return 10
                case 6:  # KING
                    return 4
                case 5:  # QUEEN
                    return 3
                case 1:  # EIGHT
                    return 0
                case 0:  # SEVEN
                    return 0
        
        # Regular (Trump suit) mode
        elif trump.mode == TrumpMode.Regular:
            if self.suit == trump.suit:
                match self.rank:
                    case 4:  # JACK (trump)
                        return 20
                    case 2:  # NINE (trump)
                        return 14
                    case 7:  # ACE
                        return 11
                    case 3:  # TEN
                        return 10
                    case 6:  # KING
                        return 4
                    case 5:  # QUEEN
                        return 3
                    case 1:  # EIGHT
                        return 0
                    case 0:  # SEVEN
                        return 0
            else:
                match self.rank:
                    case 7:  # ACE
                        return 11
                    case 3:  # TEN
                        return 10
                    case 6:  # KING
                        return 4
                    case 5:  # QUEEN
                        return 3
                    case 4:  # JACK
                        return 2
                    case 2:  # NINE
                        return 0
                    case 1:  # EIGHT
                        return 0
                    case 0:  # SEVEN
                        return 0
        
        # Default fallback
        raise ValueError(f"Unknown trump mode: {trump.mode}")


