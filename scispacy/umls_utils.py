from typing import Optional, List, Dict

# TODO(Mark): Remove in scispacy v1.0, for backward compatability only.
from scispacy.linking_utils import Entity as UmlsEntity, UmlsKnowledgeBase  # noqa

# preferred definition sources (from S2)
DEF_SOURCES_PREFERRED = {"NCI_BRIDG", "NCI_NCI-GLOSS", "NCI", "GO", "MSH", "NCI_FDA"}


def read_umls_file_headers(meta_path: str, filename: str) -> List[str]:
    """
    Read the file descriptor MRFILES.RRF from a UMLS release and get column headers (names)
    for the given file

    MRFILES.RRF file format: a pipe-separated values
    Useful columns:
        column 0: name of one of the files in the META directory
        column 2: column names of that file

    Args:
        meta_path: path to the META directory of an UMLS release
        filename: name of the file to get its column headers
    Returns:
        a list of column names
    """
    file_descriptors = f"{meta_path}/MRFILES.RRF"  # to get column names
    with open(file_descriptors, encoding="utf-8") as fin:
        for line in fin:
            splits = line.split("|")
            found_filename = splits[0]
            column_names = (splits[2] + ",").split(
                ","
            )  # ugly hack because all files end with an empty column
            if found_filename in filename:
                return column_names
    assert False, f"Couldn't find column names for file {filename}"
    return None


def read_umls_concepts(
    meta_path: str,
    concept_details: Dict,
    source: Optional[str] = None,
    lang: str = "ENG",
    non_suppressed: bool = True,
) -> None:
    """
    Read the concepts file MRCONSO.RRF from a UMLS release and store it in
    concept_details dictionary. Each concept is represented with
    - concept_id
    - canonical_name
    - aliases
    - types
    - definition
    This function fills the first three. If a canonical name is not found, it is left empty.

    MRFILES.RRF file format: a pipe-separated values
    Useful columns: CUI, LAT, SUPPRESS, STR, ISPREF, TS, STT

    Args:
        meta_path: path to the META directory of an UMLS release
        concept_details: a dictionary to be filled with concept informations
        source: An optional source identifier, used as a filter to extract only a
                specific source from UMLS.
        lang: An optional language identifier, used to filter terms by language
        non_suppressed: flag to indicate whether only non-suppressed concepts should be kept
    """
    concepts_filename = "MRCONSO.RRF"
    headers = read_umls_file_headers(meta_path, concepts_filename)
    with open(f"{meta_path}/{concepts_filename}", encoding="utf-8") as fin:
        for line in fin:
            splits = line.strip().split("|")
            assert len(headers) == len(splits), (headers, splits)
            concept = dict(zip(headers, splits))
            if (lang is not None and concept["LAT"] != lang) or (
                non_suppressed and concept["SUPPRESS"] != "N"
            ):
                continue  # Keep non-suppressed concepts in target language only

            if source is not None:
                if concept["SAB"] != source:
                    continue

            concept_id = concept["CUI"]
            if concept_id not in concept_details:  # a new concept
                # add it to the dictionary with an empty list of aliases and types
                concept_details[concept_id] = {
                    "concept_id": concept_id,
                    "aliases": [],
                    "types": [],
                }

            concept_name = concept["STR"]
            # this condition is copied from S2. It checks if the concept name is canonical or not
            is_canonical = (
                concept["ISPREF"] == "Y"
                and concept["TS"] == "P"
                and concept["STT"] == "PF"
            )

            if not is_canonical or "canonical_name" in concept_details[concept_id]:
                # not a canonical name or a canonical name already found
                concept_details[concept_id]["aliases"].append(
                    concept_name
                )  # add it as an alias
            else:
                concept_details[concept_id][
                    "canonical_name"
                ] = concept_name  # set as canonical name


def read_umls_types(meta_path: str, concept_details: Dict) -> None:
    """
    Read the types file MRSTY.RRF from a UMLS release and store it in
    concept_details dictionary. This function adds the `types` field
    to the information of each concept

    MRSTY.RRF file format: a pipe-separated values
    Useful columns: CUI, TUI

    Args:
        meta_path: path to the META directory of an UMLS release
        concept_details: a dictionary to be filled with concept informations
    """
    types_filename = "MRSTY.RRF"
    headers = read_umls_file_headers(meta_path, types_filename)
    with open(f"{meta_path}/{types_filename}", encoding="utf-8") as fin:
        for line in fin:
            splits = line.strip().split("|")
            assert len(headers) == len(splits)
            concept_type = dict(zip(headers, splits))

            concept = concept_details.get(concept_type["CUI"])
            if (
                concept is not None
            ):  # a small number of types are for concepts that don't exist
                concept["types"].append(concept_type["TUI"])


def read_umls_definitions(meta_path: str, concept_details: Dict) -> None:
    """
    Read the types file MRDEF.RRF from a UMLS release and store it in
    concept_details dictionary. This function adds the `definition` field
    to the information of each concept

    MRDEF.RRF file format: a pipe-separated values
    Useful columns: CUI, SAB, SUPPRESS, DEF

    Args:
        meta_path: path to the META directory of an UMLS release
        concept_details: a dictionary to be filled with concept informations
    """
    definitions_filename = "MRDEF.RRF"
    headers = read_umls_file_headers(meta_path, definitions_filename)
    with open(f"{meta_path}/{definitions_filename}", encoding="utf-8") as fin:
        headers = read_umls_file_headers(meta_path, definitions_filename)
        for line in fin:
            splits = line.strip().split("|")
            assert len(headers) == len(splits)
            definition = dict(zip(headers, splits))

            if definition["SUPPRESS"] != "N":
                continue
            is_from_preferred_source = definition["SAB"] in DEF_SOURCES_PREFERRED
            concept = concept_details.get(definition["CUI"])
            if (
                concept is None
            ):  # a small number of definitions are for concepts that don't exist
                continue

            if (
                "definition" not in concept
                or is_from_preferred_source
                and concept["is_from_preferred_source"] == "N"
            ):
                concept["definition"] = definition["DEF"]
                concept["is_from_preferred_source"] = (
                    "Y" if is_from_preferred_source else "N"
                )
