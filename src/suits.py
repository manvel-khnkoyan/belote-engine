class _SuitManager:
    def __init__(self):
        self._suits = ['♠', '♥', '♦', '♣']
        self._transform_map = None
    
    def transform(self, map):
        self._transform_map = map
    
    def __getitem__(self, index: int) -> str:
        if self._transform_map:
            # Handle both dict and callable (function) transform maps
            if callable(self._transform_map):
                transformed_index = self._transform_map(index)
            else:
                transformed_index = self._transform_map[index]
            return self._suits[transformed_index]
        return self._suits[index]

Suits = _SuitManager()