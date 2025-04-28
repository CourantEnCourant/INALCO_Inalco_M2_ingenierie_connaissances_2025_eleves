""""""
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional

from TP.scripts.Edge import Edge
from TP.scripts.NodeRepository import NodeRepository
from TP.scripts.Singleton import Singleton


class EdgeRepository(metaclass=Singleton):
    def __init__(self) -> None:
        self.id2edge = {}

    def create_edge(self, from_node_id, to_node_id) -> None:
        # Find nodes
        from_node = NodeRepository().find_node_by_id(from_node_id)
        to_node = NodeRepository().find_node_by_id(to_node_id)
        # Create edge
        edge = Edge(from_node_id, to_node_id)
        self.id2edge[edge.id] = edge
        # Update neighbour
        from_node.add_neighbours(to_node)
        to_node.add_neighbours(from_node)

    def find_edge_by_id(self, edge_id: int) -> Edge:
        if not self.id2edge.get(edge_id):
            raise self.EdgeNotFoundError(f"Edge with id {edge_id} does not exist")
        return self.id2edge[edge_id]

    def find_edge_by_nodes_id(self, from_node_id: int, to_node_id: int) -> Edge:
        for _, edge in self.id2edge.items():
            if edge.from_node_id == from_node_id and edge.to_node_id == to_node_id:
                return edge
        raise self.EdgeNotFoundError(f"Edge between id {from_node_id!r} and id {to_node_id!r} does not exist")

    def find_edge_by_nodes_name(self, from_node_name: str, to_node_name: str) -> Edge:
        from_node = NodeRepository().find_node_by_name(from_node_name)
        to_node = NodeRepository().find_node_by_name(to_node_name)
        for _, edge in self.id2edge.items():
            if edge.from_node_id == from_node.id and edge.to_node_id == to_node.id:
                return edge
        raise self.EdgeNotFoundError(f"Edge between {from_node_name!r} and {to_node_name!r} does not exist")

    def update_edge(self, edge_id, from_node_id, to_node_id):
        edge = self.find_edge_by_id(edge_id)
        edge.update(from_node_id, to_node_id)

    def add_attributes_to_edge(self, edge_id: int, kwargs):
        edge = self.find_edge_by_id(edge_id)
        edge.add_attribute(**kwargs)

    def delete_edge(self, edge_id):
        self.id2edge.pop(edge_id)

    def delete_edge_by_nodes_id(self, from_node_id: int, to_node_id: int) -> None:
        edge_id = self.find_edge_by_nodes_id(from_node_id, to_node_id).id
        self.delete_edge(edge_id)

    class EdgeNotFoundError(Exception):
        pass
