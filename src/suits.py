class _SuitManager:
    def __init__(self):
        self._suits = ['♠', '♥', '♦', '♣']
        self._transform_map = None
    
    def transform(self, map):
        self._transform_map = map
    
    def __getitem__(self, index: int) -> str:
        return self._suits[self._transform_map[index] if self._transform_map else index]

Suits = _SuitManager()