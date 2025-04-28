""""""
from dataclasses import dataclass, field

import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional, Any


@dataclass
class Node:
    id: int = field(init=False)
    neighbours: list = field(init=False)

    ID = 0

    def __post_init__(self):
        self.id = Node.ID
        self.neighbours = []
        Node.ID += 1

    def __init__(self, **kwargs):
        """Override dataclass' constructor"""
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __repr__(self):
        attributes = []
        for key, value in self.__dict__.items():
            attributes.append(f"{key}={value!r}")
        return f"Node({', '.join(attributes)})"

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def add_neighbours(self, *nodes: "Node"):
        self.neighbours.extend([node.id for node in nodes])
