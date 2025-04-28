""""""
from dataclasses import dataclass, field

import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional


@dataclass
class Edge:
    from_node_id: int
    to_node_id: int
    id: int = field(init=False)

    _next_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        # Set id
        self.id = Edge._next_id
        Edge._next_id += 1

    def update(self, from_node_id, to_node_id) -> None:
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id

    def add_attribute(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
