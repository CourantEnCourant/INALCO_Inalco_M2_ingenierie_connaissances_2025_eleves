""""""
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional

from TP.scripts.Node import Node
from TP.scripts.Singleton import Singleton


class NodeRepository(metaclass=Singleton):
    def __init__(self) -> None:
        self.id2node = {}
        self.edge_repository = None

    def set_edge_repository(self, edge_repository) -> None:
        """Injection dependency for edge repository"""
        self.edge_repository = edge_repository

    def create_node(self, **kwargs) -> None:
        node = Node(**kwargs)
        self.id2node[node.id] = node

    def find_node_by_id(self, node_id: int) -> Node:
        if not self.id2node.get(node_id):
            raise ValueError(f"Node {node_id} does not exist")
        return self.id2node[node_id]

    def find_node_by_name(self, name: str) -> Node:
        """Find the node by name"""
        for _, node in self.id2node.items():
            try:
                if node.name == name:
                    return node
            except AttributeError:
                continue
        # If found nothing
        raise self.NodeNotFoundError(f"Node with name {name!r} does not exist")

    def find_nodes_by_name(self, name: str) -> list[Node]:
        """Find nodes by startswith method"""
        found_nodes = []
        for _, node in self.id2node.items():
            try:
                if node.name.casefold().startswith(name.casefold()):
                    found_nodes.append(node)
            except AttributeError:
                continue
        return found_nodes

    def update_node(self, node_id, kwargs: dict) -> None:
        """Update node by a dictionary"""
        node = self.find_node_by_id(node_id)
        node.update(**kwargs)

    def delete_node_by_id(self, node_id: int) -> None:
        node = self.find_node_by_id(node_id)
        self.id2node.pop(node.id)
        for connected_node_id in node.neighbours:
            self.edge_repository.delete_edge_by_nodes_id(node_id, connected_node_id)

    def delete_node_by_name(self, name: str) -> None:
        target_id = self.find_node_by_name(name).id
        self.delete_node_by_id(target_id)

    class NodeNotFoundError(Exception):
        pass
