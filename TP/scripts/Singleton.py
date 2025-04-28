""""""
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Singleton Pattern Initialization"""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
