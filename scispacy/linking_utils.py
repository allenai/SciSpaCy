"""
This submodule contains a data structure for storing lexical
indexes over various biomedical vocabularies.

There are several built-in vocabularies, which can be imported
and instantiated like in:

.. code-block:: python

    from scispacy.linking_utils import UmlsKnowledgeBase

    kb = UmlsKnowledgeBase()

In general, new :class:`KnowledgeBase` objects can be constructed
from a list of :class:`Entity` objects, or a path to a JSON or JSONL
file containing dictionaries shaped the same way:

.. code-block:: python

    from scispacy.linking_utils import KnowledgeBase

    # UMLS
    kb = KnowledgeBase(
        "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/"
        "data/kbs/2023-04-23/umls_mesh_2022.jsonl"
    )

"""

import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    List,
    Dict,
    NamedTuple,
    Optional,
    Set,
    Union,
    Iterable,
    Tuple,
    DefaultDict,
    Generator,
)

from scispacy.file_cache import cached_path
from scispacy.umls_semantic_type_tree import (
    UmlsSemanticTypeTree,
    construct_umls_tree_from_tsv,
)

__all__ = [
    "Entity",
    "KnowledgeBase",
    "UmlsKnowledgeBase",
    "Mesh",
    "GeneOntology",
    "HumanPhenotypeOntology",
    "RxNorm",
]


class Entity(NamedTuple):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    types: List[str] = []
    definition: Optional[str] = None

    def __repr__(self) -> str:
        rep = ""
        num_aliases = len(self.aliases)
        rep = rep + f"CUI: {self.concept_id}, Name: {self.canonical_name}\n"
        rep = rep + f"Definition: {self.definition}\n"
        rep = rep + f"TUI(s): {', '.join(self.types)}\n"
        if num_aliases > 10:
            rep = (
                rep
                + f"Aliases (abbreviated, total: {num_aliases}): \n\t {', '.join(self.aliases[:10])}"
            )
        else:
            rep = (
                rep + f"Aliases: (total: {num_aliases}): \n\t {', '.join(self.aliases)}"
            )
        return rep


DEFAULT_UMLS_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2023-04-23/umls_2022_ab_cat0129.jsonl"  # noqa
DEFAULT_UMLS_TYPES_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_semantic_type_tree.tsv"


@contextmanager
def _iter_entities(
    path_or_entities: Union[str, Path, Iterable[Entity]],
) -> Generator[Iterable[Entity], None, None]:
    """Iterate through entities from a JSON file, JSONL file, or pass through an existing iterable."""
    if isinstance(path_or_entities, (str, Path)):
        # normalize paths
        path_or_entities = cached_path(path_or_entities)

        with open(path_or_entities) as file:
            if path_or_entities.endswith("jsonl"):
                yield (Entity(**json.loads(line)) for line in file)
            else:
                yield (Entity(**record) for record in json.load(file))
    else:
        yield path_or_entities


def _index_entities(
    entities: Iterable[Entity],
) -> Tuple[Dict[str, Entity], Dict[str, Set[str]]]:
    """Create indexes over entities for use in a :class:`KnowledgeBase`.

    Parameters
    ----------
    entities :
        An iterable (e.g., a list) of entity objects

    Returns
    -------
    A pair of indexes for:

    1. A mapping from local unique identifiers (e.g., CUIs for UMLS) to entity objects
    2. A mapping from aliases (e.g., canonical names, aliases) to local unique identifiers
    """
    cui_to_entity: Dict[str, Entity] = {}
    alias_to_cuis: DefaultDict[str, Set[str]] = defaultdict(set)
    for entity in entities:
        for alias in set(entity.aliases + [entity.canonical_name]):
            alias_to_cuis[alias].add(entity.concept_id)
        cui_to_entity[entity.concept_id] = entity
    return cui_to_entity, dict(alias_to_cuis)


class KnowledgeBase:
    """
    A class representing two commonly needed views of a Knowledge Base:
    1. A mapping from concept_id to an Entity NamedTuple with more information.
    2. A mapping from aliases to the sets of concept ids for which they are aliases.

    Parameters
    ----------
    file_path: str, required.
        The file path to the json/jsonl representation of the KB to load.
    """

    cui_to_entity: Dict[str, Entity]
    alias_to_cuis: Dict[str, Set[str]]

    def __init__(
        self,
        file_path: Union[None, str, Path, Iterable[Entity]] = None,
    ) -> None:
        if file_path is None:
            raise ValueError(
                "Do not use the default arguments to KnowledgeBase. "
                "Instead, use a subclass (e.g UmlsKnowledgeBase) or pass a path to a kb."
            )
        with _iter_entities(file_path) as entities:
            self.cui_to_entity, self.alias_to_cuis = _index_entities(entities)


class UmlsKnowledgeBase(KnowledgeBase):
    def __init__(
        self,
        file_path: Union[str, Path, Iterable[Entity]] = DEFAULT_UMLS_PATH,
        types_file_path: str = DEFAULT_UMLS_TYPES_PATH,
    ) -> None:
        super().__init__(file_path)

        self.semantic_type_tree: UmlsSemanticTypeTree = construct_umls_tree_from_tsv(
            types_file_path
        )


class Mesh(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_mesh_2022.jsonl",  # noqa
    ) -> None:
        super().__init__(file_path)


class GeneOntology(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_go_2022.jsonl",  # noqa
    ) -> None:
        super().__init__(file_path)


class HumanPhenotypeOntology(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_hpo_2022.jsonl",  # noqa
    ) -> None:
        super().__init__(file_path)


class RxNorm(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_rxnorm_2022.jsonl",  # noqa
    ) -> None:
        super().__init__(file_path)
