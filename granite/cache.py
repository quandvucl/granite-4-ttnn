"""Position tracker for the Granite hybrid model decode loop."""

from typing import List, Optional


class MambaCacheManager:
    """Tracks decode position."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int = 1,
        attention_layer_indices: Optional[List[int]] = None,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.attention_layer_indices = attention_layer_indices or [5, 15, 25, 35]
        self.current_position = 0

    # Update the current decode position.
    def increment_position(self, num_tokens: int = 1):
        self.current_position += num_tokens

    # Get the current decode position.
    def get_position(self) -> int:
        return self.current_position

    # Reset the current decode position to 0.
    def reset(self):
        self.current_position = 0
