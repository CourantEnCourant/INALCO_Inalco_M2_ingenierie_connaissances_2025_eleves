""""""
from TP.scripts.Node import Node
from TP.scripts.NodeRepository import NodeRepository
from TP.scripts.Edge import Edge
from TP.scripts.EdgeRepository import EdgeRepository
from TP.scripts.Singleton import Singleton


class OneRingDB(metaclass=Singleton):
    """Graph database designed with repository pattern"""
    def __init__(self) -> None:
        self.node_repository = NodeRepository()
        self.edge_repository = EdgeRepository()
        self.node_repository.set_edge_repository(self.edge_repository)

    # Create
    def create_node(self, **kwargs) -> None:
        self.node_repository.create_node(**kwargs)

    def create_edge(self, from_node_id, to_node_id) -> None:
        self.edge_repository.create_edge(from_node_id, to_node_id)

    # Read
    def find_node_by_id(self, node_id: int) -> Node:
        return self.node_repository.find_node_by_id(node_id)

    def find_node_by_name(self, name: str) -> Node:
        """Find the node by name"""
        return self.node_repository.find_node_by_name(name)

    def find_nodes_by_name(self, name: str) -> list[Node]:
        return self.node_repository.find_nodes_by_name(name)

    def find_edge_by_id(self, edge_id: int) -> Edge:
        return self.edge_repository.find_edge_by_id(edge_id)

    def find_edge_by_nodes_id(self, from_node_id: int, to_node_id: int) -> Edge:
        return self.edge_repository.find_edge_by_nodes_id(from_node_id, to_node_id)

    def find_edge_by_nodes_name(self, from_node_name: str, to_node_name: str) -> Edge:
        return self.edge_repository.find_edge_by_nodes_name(from_node_name, to_node_name)

    # Update
    def update_node(self, node_id, kwargs: dict) -> None:
        """Update node by a dictionary"""
        self.node_repository.update_node(node_id, kwargs)

    def update_edge(self, edge_id, from_node_id, to_node_id):
        self.edge_repository.update_edge(edge_id, from_node_id, to_node_id)

    def add_attributes_to_edge(self, edge_id: int, kwargs) -> None:
        self.edge_repository.add_attributes_to_edge(edge_id, kwargs)

    # Delete
    def delete_node_by_id(self, node_id: int) -> None:
        self.node_repository.delete_node_by_id(node_id)

    def delete_node_by_name(self, name: str) -> None:
        self.node_repository.delete_node_by_name(name)

    def delete_edge(self, edge_id: int) -> None:
        self.edge_repository.delete_edge(edge_id)

    def delete_edge_by_nodes_id(self, from_node_id: int, to_node_id: int) -> None:
        self.edge_repository.delete_edge_by_nodes_id(from_node_id, to_node_id)
