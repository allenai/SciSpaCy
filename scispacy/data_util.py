from typing import Optional, NamedTuple, List, Iterator, Dict, Tuple
import tarfile
import atexit
import os
import shutil
import tempfile

from scispacy.file_cache import cached_path


class MedMentionEntity(NamedTuple):
    start: int
    end: int
    mention_text: str
    mention_type: str
    umls_id: str


class MedMentionExample(NamedTuple):
    title: str
    abstract: str
    text: str
    pubmed_id: str
    entities: List[MedMentionEntity]


def process_example(lines: List[str]) -> MedMentionExample:
    """
    Processes the text lines of a file corresponding to a single MedMention abstract,
    extracts the title, abstract, pubmed id and entities. The lines of the file should
    have the following format:
    PMID | t | Title text
    PMID | a | Abstract text
    PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID
    ...
    """
    pubmed_id, _, title = [x.strip() for x in lines[0].split("|", maxsplit=2)]
    _, _, abstract = [x.strip() for x in lines[1].split("|", maxsplit=2)]

    entities = []
    for entity_line in lines[2:]:
        _, start, end, mention, mention_type, umls_id = entity_line.split("\t")
        mention_type = mention_type.split(",")[0]
        entities.append(
            MedMentionEntity(int(start), int(end), mention, mention_type, umls_id)
        )
    return MedMentionExample(
        title, abstract, title + " " + abstract, pubmed_id, entities
    )


def med_mentions_example_iterator(filename: str) -> Iterator[MedMentionExample]:
    """
    Iterates over a Med Mentions file, yielding examples.
    """
    with open(filename, "r", encoding="utf-8") as med_mentions_file:
        lines = []
        for line in med_mentions_file:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                yield process_example(lines)
                lines = []
        # Pick up stragglers
        if lines:
            yield process_example(lines)


def select_subset_of_overlapping_chain(
    chain: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    """
    Select the subset of entities in an overlapping chain to return by greedily choosing the
    longest entity in the chain until there are no entities remaining
    """
    sorted_chain = sorted(chain, key=lambda x: x[1] - x[0], reverse=True)
    selections_from_chain: List[Tuple[int, int, str]] = []
    chain_index = 0
    # dump the current chain by greedily keeping the longest entity that doesn't overlap
    while chain_index < len(sorted_chain):
        entity = sorted_chain[chain_index]
        match_found = False
        for already_selected_entity in selections_from_chain:
            max_start = max(entity[0], already_selected_entity[0])
            min_end = min(entity[1], already_selected_entity[1])
            if len(range(max_start, min_end)) > 0:
                match_found = True
                break

        if not match_found:
            selections_from_chain.append(entity)

        chain_index += 1

    return selections_from_chain


def remove_overlapping_entities(
    sorted_spacy_format_entities: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    """
    Removes overlapping entities from the entity set, by greedilytaking the longest
    entity from each overlapping chain. The input list of entities should be sorted
    and follow the spacy format.
    """
    spacy_format_entities_without_overlap = []
    current_overlapping_chain: List[Tuple[int, int, str]] = []
    current_overlapping_chain_start = 0
    current_overlapping_chain_end = 0
    for i, current_entity in enumerate(sorted_spacy_format_entities):
        current_entity = sorted_spacy_format_entities[i]
        current_entity_start = current_entity[0]
        current_entity_end = current_entity[1]

        if len(current_overlapping_chain) == 0:
            current_overlapping_chain.append(current_entity)
            current_overlapping_chain_start = current_entity_start
            current_overlapping_chain_end = current_entity_end
        else:
            min_end = min(current_entity_end, current_overlapping_chain_end)
            max_start = max(current_entity_start, current_overlapping_chain_start)
            if min_end - max_start > 0:
                current_overlapping_chain.append(current_entity)
                current_overlapping_chain_start = min(
                    current_entity_start, current_overlapping_chain_start
                )
                current_overlapping_chain_end = max(
                    current_entity_end, current_overlapping_chain_end
                )
            else:
                selections_from_chain = select_subset_of_overlapping_chain(
                    current_overlapping_chain
                )

                current_overlapping_chain = []
                spacy_format_entities_without_overlap.extend(selections_from_chain)
                current_overlapping_chain.append(current_entity)
                current_overlapping_chain_start = current_entity_start
                current_overlapping_chain_end = current_entity_end

    spacy_format_entities_without_overlap.extend(
        select_subset_of_overlapping_chain(current_overlapping_chain)
    )

    return sorted(spacy_format_entities_without_overlap, key=lambda x: x[0])


def read_full_med_mentions(
    directory_path: str,
    label_mapping: Optional[Dict[str, str]] = None,
    span_only: bool = False,
    spacy_format: bool = True,
    use_umls_ids: bool = False,
):
    def _cleanup_dir(dir_path: str):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    resolved_directory_path = cached_path(directory_path)
    if "tar.gz" in directory_path:
        # Extract dataset to temp dir
        tempdir = tempfile.mkdtemp()
        print(
            f"extracting dataset directory {resolved_directory_path} to temp dir {tempdir}"
        )
        with tarfile.open(resolved_directory_path, "r:gz") as archive:
            archive.extractall(tempdir)
        # Postpone cleanup until exit in case the unarchived
        # contents are needed outside this function.
        atexit.register(_cleanup_dir, tempdir)

        resolved_directory_path = tempdir

    expected_names = [
        "corpus_pubtator.txt",
        "corpus_pubtator_pmids_all.txt",
        "corpus_pubtator_pmids_dev.txt",
        "corpus_pubtator_pmids_test.txt",
        "corpus_pubtator_pmids_trng.txt",
    ]

    corpus = os.path.join(resolved_directory_path, expected_names[0])
    examples = med_mentions_example_iterator(corpus)

    with open(os.path.join(resolved_directory_path, expected_names[4])) as f:
        train_ids = {x.strip() for x in f}
    with open(os.path.join(resolved_directory_path, expected_names[2])) as f:
        dev_ids = {x.strip() for x in f}
    with open(os.path.join(resolved_directory_path, expected_names[3])) as f:
        test_ids = {x.strip() for x in f}

    train_examples = []
    dev_examples = []
    test_examples = []

    def label_function(label):
        if span_only:
            return "ENTITY"
        if label_mapping is None:
            return label
        else:
            return label_mapping[label]

    for example in examples:
        spacy_format_entities = [
            (
                x.start,
                x.end,
                label_function(x.mention_type) if not use_umls_ids else x.umls_id,
            )
            for x in example.entities
        ]
        spacy_format_entities = remove_overlapping_entities(
            sorted(spacy_format_entities, key=lambda x: x[0])
        )
        spacy_example = (example.text, {"entities": spacy_format_entities})
        if example.pubmed_id in train_ids:
            train_examples.append(spacy_example if spacy_format else example)

        elif example.pubmed_id in dev_ids:
            dev_examples.append(spacy_example if spacy_format else example)

        elif example.pubmed_id in test_ids:
            test_examples.append(spacy_example if spacy_format else example)

    return train_examples, dev_examples, test_examples


SpacyNerExample = Tuple[str, Dict[str, List[Tuple[int, int, str]]]]


def _handle_sentence(examples: List[Tuple[str, str]]) -> SpacyNerExample:
    """
    Processes a single sentence by building it up as a space separated string
    with its corresponding typed entity spans.
    """
    start_index = -1
    current_index = 0
    in_entity = False
    entity_type: str = ""
    sent = ""
    entities: List[Tuple[int, int, str]] = []
    for word, entity in examples:
        sent += word
        sent += " "
        if entity != "O":
            if in_entity:
                pass
            else:
                start_index = current_index
                in_entity = True
                entity_type = entity[2:].upper()
        else:
            if in_entity:
                end_index = current_index - 1
                entities.append((start_index, end_index, entity_type.replace("-", "_")))
            in_entity = False
            entity_type = ""
            start_index = -1
        current_index += len(word) + 1
    if in_entity:
        end_index = current_index - 1
        entities.append((start_index, end_index, entity_type))

    # Remove last space.
    sent = sent[:-1]
    return (sent, {"entities": entities})


def read_ner_from_tsv(filename: str) -> List[SpacyNerExample]:
    """
    Reads BIO formatted NER data from a TSV file, such as the
    NER data found here:
    https://github.com/cambridgeltl/MTL-Bioinformatics-2016

    Data is expected to be 2 tab seperated tokens per line, with
    sentences denoted by empty lines. Sentences read by this
    function will be already tokenized, but returned as a string,
    as this is the format required by SpaCy. Consider using the
    WhitespaceTokenizer(scispacy/util.py) to split this data
    with a SpaCy model.

    Parameters
    ----------
    filename : str
        The path to the tsv data.

    Returns
    -------
    spacy_format_data : List[SpacyNerExample]
        The BIO tagged NER examples.
    """
    spacy_format_data = []
    examples: List[Tuple[str, str]] = []
    with open(cached_path(filename)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            # We have reached the end of a sentence.
            if not line:
                if not examples:
                    continue
                spacy_format_data.append(_handle_sentence(examples))
                examples = []
            else:
                word, entity = line.split("\t")
                examples.append((word, entity))
    if examples:
        spacy_format_data.append(_handle_sentence(examples))

    return spacy_format_data
