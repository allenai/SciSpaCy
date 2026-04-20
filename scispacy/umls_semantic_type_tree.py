from typing import NamedTuple, List, Dict, Deque, Any, Optional
from collections import deque

from scispacy.file_cache import cached_path


class SemanticTypeNode(NamedTuple):
    type_id: str
    full_name: str
    children: List[Any]  # Mypy does not support nested types yet :(
    level: int


class UmlsSemanticTypeTree:
    """
    A utility class for manipulating the UMLS Semantic Type Hierarchy.
    Designed to be constructed from a TSV file using `construct_umls_tree_from_tsv`.
    """

    def __init__(self, root: SemanticTypeNode) -> None:
        children = self.get_children(root)
        children.append(root)
        # We'll store the nodes as a flattened list too, because
        # we don't just care about the leaves of the tree - sometimes
        # we'll need efficient access to intermediate nodes, and the tree
        # is tiny anyway.
        self.flat_nodes: List[SemanticTypeNode] = children
        self.type_id_to_node = {node.type_id: node for node in self.flat_nodes}
        self.depth = max([node.level for node in self.flat_nodes])

    def get_node_from_id(self, type_id: str) -> SemanticTypeNode:
        return self.type_id_to_node[type_id]

    def get_canonical_name(self, type_id: str) -> str:
        return self.type_id_to_node[type_id].full_name

    def get_nodes_at_depth(self, level: int) -> List[SemanticTypeNode]:
        """
        Returns nodes at a particular depth in the tree.
        """
        return [node for node in self.flat_nodes if node.level == level]

    def get_children(self, node: SemanticTypeNode) -> List[SemanticTypeNode]:
        """
        Recursively build up a flat list of all a node's children.
        """
        children = []
        for child in node.children:
            children.append(child)
            children.extend(self.get_children(child))
        return children

    def get_parent(self, node: SemanticTypeNode) -> Optional[SemanticTypeNode]:
        """
        Returns the parent of the input node, returning None if the input node is the root of the tree
        """
        current_depth = node.level
        possible_parents = self.get_nodes_at_depth(current_depth - 1)

        for possible_parent in possible_parents:
            for child in possible_parent.children:
                if child.type_id == node.type_id:
                    return possible_parent

        # If there are no parents, we are at the root and return None
        return None

    def get_collapsed_type_id_map_at_level(self, level: int) -> Dict[str, str]:
        """
        Constructs a label mapping from the original tree labels to a tree of a fixed depth,
        collapsing labels greater than the depth specified to the closest parent which is
        still present in the new fixed depth tree. This is effectively mapping to a _coarser_
        label space.
        """
        new_type_id_map: Dict[str, str] = {k: k for k in self.type_id_to_node.keys()}
        for node in self.get_nodes_at_depth(level):
            for child in self.get_children(node):
                new_type_id_map[child.type_id] = node.type_id
        return new_type_id_map


def construct_umls_tree_from_tsv(filepath: str) -> UmlsSemanticTypeTree:
    """
    Reads in a tsv file which is formatted as a depth first traversal of
    a hierarchy tree, where nodes are of the format:

    Name TAB UMLS Semantic Type TAB Tree Depth

    Event    T051    1
      Activity    T052    2
        Behavior    T053    3
          Social Behavior    T054    4
          Individual Behavior    T055    4
        Daily or Recreational Activity    T056    3
    """

    node_stack: Deque[SemanticTypeNode] = deque()
    with open(cached_path(filepath), "r") as tsv_file:
        for line in tsv_file:
            name, type_id, level = line.split("\t")
            name = name.strip()
            int_level = int(level.strip())
            node = SemanticTypeNode(type_id, name, [], int_level)

            node_stack.append(node)

    def attach_children(node: SemanticTypeNode, stack: Deque[SemanticTypeNode]):
        while stack and stack[0].level > node.level:
            popped = stack.popleft()
            attach_children(popped, stack)
            node.children.append(popped)

    first = node_stack.popleft()
    attach_children(first, node_stack)

    return UmlsSemanticTypeTree(first)
