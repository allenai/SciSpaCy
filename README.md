<p align="center"><img width="50%" src="docs/scispacy-logo.png" /></p>


This repository contains custom pipes and models related to using spaCy for scientific documents.

In particular, there is a custom tokenizer that adds tokenization rules on top of spaCy's
rule-based tokenizer, a POS tagger and syntactic parser trained on biomedical data and
an entity span detection model. Separately, there are also NER models for more specific tasks.

**Just looking to test out the models on your data? Check out our [demo](https://scispacy.apps.allenai.org)** (Note: this demo is running an older version of scispaCy and may produce different results than the latest version).


## Installation
Installing scispacy requires two steps: installing the library and intalling the models. To install the library, run:
```bash
pip install scispacy
```

to install a model (see our full selection of available models below), run a command like the following:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

Note: We strongly recommend that you use an isolated Python environment (such as virtualenv or conda) to install scispacy.
Take a look below in the "Setting up a virtual environment" section if you need some help with this.
Additionally, scispacy uses modern features of Python and as such is only available for **Python 3.9 through 3.13**.


#### Setting up a virtual environment

[Mamba](https://mamba.readthedocs.io/en/latest/) can be used set up a virtual environment with the
version of Python required for scispaCy.  If you already have a Python
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Follow the installation instructions for Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

2.  Create a Conda environment called "scispacy" with Python 3.13 (any supported version between 3.9 and 3.13 will work):

    ```bash
    mamba create -n scispacy python=3.13
    ```

3.  Activate the Mamba environment. You will need to activate the Conda environment in each terminal in which you want to use scispaCy.

    ```bash
    mamba activate scispacy
    ```

Now you can install `scispacy` and one of the models using the steps above.


Once you have completed the above steps and downloaded one of the models below, you can load a scispaCy model as you would any other spaCy model. For example:
```python
import spacy
nlp = spacy.load("en_core_sci_sm")
doc = nlp("Alterations in the hypocretin receptor 2 and preprohypocretin genes produce narcolepsy in some animals.")
```

#### Note on upgrading
If you are upgrading `scispacy`, you will need to download the models again, to get the model versions compatible with the version of `scispacy` that you have. The link to the model that you download should contain the version number of `scispacy` that you have.

## Available Models

To install a model, click on the link below to download the model, and then run 

```python
pip install </path/to/download>
```

Alternatively, you can install directly from the URL by right-clicking on the link, selecting "Copy Link Address" and running 
```python
pip install CMD-V(to paste the copied URL)
```

| Model          | Description       | Install URL
|:---------------|:------------------|:----------|
| en_core_sci_sm | A full spaCy pipeline for biomedical data with a ~100k vocabulary. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz)|
| en_core_sci_md |  A full spaCy pipeline for biomedical data with a ~360k vocabulary and 50k word vectors. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz)|
| en_core_sci_lg |  A full spaCy pipeline for biomedical data with a ~785k vocabulary and 600k word vectors. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz)|
| en_core_sci_scibert |  A full spaCy pipeline for biomedical data with a ~785k vocabulary and `allenai/scibert-base` as the transformer model. You may want to [use a GPU](https://spacy.io/usage#gpu) with this model. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz)|
| en_ner_craft_md|  A spaCy NER model trained on the CRAFT corpus.|[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz)|
| en_ner_jnlpba_md | A spaCy NER model trained on the JNLPBA corpus.| [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz)|
| en_ner_bc5cdr_md |  A spaCy NER model trained on the BC5CDR corpus. | [Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz)|
| en_ner_bionlp13cg_md |  A spaCy NER model trained on the BIONLP13CG corpus. |[Download](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz)|


## Additional Pipeline Components


### AbbreviationDetector
The AbbreviationDetector is a Spacy component which implements the abbreviation detection algorithm in "A simple algorithm
    for identifying abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

You can access the list of abbreviations via the `doc._.abbreviations` attribute and for a given abbreviation,
you can access it's long form (which is a `spacy.tokens.Span`) using `span._.long_form`, which will point to
another span in the document.


#### Example Usage
```python
import spacy

from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_sm")

# Add the abbreviation pipe to the spacy pipeline.
nlp.add_pipe("abbreviation_detector")

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

>>> Abbreviation	 Span	    Definition
>>> SBMA 		 (33, 34)   Spinal and bulbar muscular atrophy
>>> SBMA 	   	 (6, 7)     Spinal and bulbar muscular atrophy
>>> AR   		 (29, 30)   androgen receptor
```

> **Note**
> If you want to be able to [serialize your `doc` objects](https://spacy.io/usage/saving-loading), load the abbreviation detector with `make_serializable=True`, e.g. `nlp.add_pipe("abbreviation_detector", config={"make_serializable": True})`

### EntityLinker

The `EntityLinker` is a SpaCy component which performs linking to a knowledge base. The linker simply performs
a string overlap - based search (char-3grams) on named entities, comparing them with the concepts in a knowledge base
using an approximate nearest neighbours search.

Currently (v2.5.0), there are 5 supported linkers:

- `umls`: Links to the [Unified Medical Language System](https://www.nlm.nih.gov/research/umls/index.html), levels 0,1,2 and 9. This has ~3M concepts.
- `mesh`: Links to the [Medical Subject Headings](https://www.nlm.nih.gov/mesh/meshhome.html). This contains a smaller set of higher quality entities, which are used for indexing in Pubmed. MeSH contains ~30k entities. NOTE: The MeSH KB is derived directly from MeSH itself, and as such uses different unique identifiers than the other KBs.
- `rxnorm`: Links to the [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/index.html) ontology. RxNorm contains ~100k concepts focused on normalized names for clinical drugs. It is comprised of several other drug vocabularies commonly used in pharmacy management and drug interaction, including First Databank, Micromedex, and the Gold Standard Drug Database.
- `go`: Links to the [Gene Ontology](http://geneontology.org/). The Gene Ontology contains ~67k concepts focused on the functions of genes.
- `hpo`: Links to the [Human Phenotype Ontology](https://hpo.jax.org/app/). The Human Phenotype Ontology contains 16k concepts focused on phenotypic abnormalities encountered in human disease.

You may want to play around with some of the parameters
below to adapt to your use case (higher precision, higher recall etc).

- `resolve_abbreviations : bool = True, optional (default = False)`
    Whether to resolve abbreviations identified in the Doc before performing linking.
    This parameter has no effect if there is no `AbbreviationDetector` in the spacy
    pipeline.
- `k : int, optional, (default = 30)`
    The number of nearest neighbours to look up from the candidate generator per mention.
- `threshold : float, optional, (default = 0.7)`
    The threshold that a mention candidate must reach to be added to the mention in the Doc
    as a mention candidate.
-   `no_definition_threshold : float, optional, (default = 0.95)`
        The threshold that a entity candidate must reach to be added to the mention in the Doc
        as a mention candidate if the entity candidate does not have a definition.
- `filter_for_definitions: bool, default = True`
    Whether to filter entities that can be returned to only include those with definitions
    in the knowledge base.
- `max_entities_per_mention : int, optional, default = 5`
    The maximum number of entities which will be returned for a given mention, regardless of
    how many are nearest neighbours are found.

This class sets the `._.kb_ents` attribute on spacy Spans, which consists of a
List[Tuple[str, float]] corresponding to the KB concept_id and the associated score
for a list of `max_entities_per_mention` number of entities.

You can look up more information for a given id using the kb attribute of this class:
```
print(linker.kb.cui_to_entity[concept_id])
```

#### Example Usage
```python
import spacy
import scispacy

from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_sm")

# This line takes a while, because we have to download ~1GB of data
# and load a large JSON file (the knowledge base). Be patient!
# Thankfully it should be faster after the first time you use it, because
# the downloads are cached.
# NOTE: The resolve_abbreviations parameter is optional, and requires that
# the AbbreviationDetector pipe has already been added to the pipeline. Adding
# the AbbreviationDetector pipe and setting resolve_abbreviations to True means
# that linking will only be performed on the long form of abbreviations.
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

# Let's look at a random entity!
entity = doc.ents[1]

print("Name: ", entity)
>>> Name: bulbar muscular atrophy

# Each entity is linked to UMLS with a score
# (currently just char-3gram matching).
linker = nlp.get_pipe("scispacy_linker")
for umls_ent in entity._.kb_ents:
	print(linker.kb.cui_to_entity[umls_ent[0]])


>>> CUI: C1839259, Name: Bulbo-Spinal Atrophy, X-Linked
>>> Definition: An X-linked recessive form of spinal muscular atrophy. It is due to a mutation of the
  				gene encoding the ANDROGEN RECEPTOR.
>>> TUI(s): T047
>>> Aliases (abbreviated, total: 50):
         Bulbo-Spinal Atrophy, X-Linked, Bulbo-Spinal Atrophy, X-Linked, ....

>>> CUI: C0541794, Name: Skeletal muscle atrophy
>>> Definition: A process, occurring in skeletal muscle, that is characterized by a decrease in protein content,
                fiber diameter, force production and fatigue resistance in response to ...
>>> TUI(s): T046
>>> Aliases: (total: 9):
         Skeletal muscle atrophy, ATROPHY SKELETAL MUSCLE, skeletal muscle atrophy, ....

>>> CUI: C1447749, Name: AR protein, human
>>> Definition: Androgen receptor (919 aa, ~99 kDa) is encoded by the human AR gene.
                This protein plays a role in the modulation of steroid-dependent gene transcription.
>>> TUI(s): T116, T192
>>> Aliases (abbreviated, total: 16):
         AR protein, human, Androgen Receptor, Dihydrotestosterone Receptor, AR, DHTR, NR3C4, ...
```

### Hearst Patterns (v0.3.0 and up)

This component implements [Automatic Aquisition of Hyponyms from Large Text Corpora](https://www.aclweb.org/anthology/C92-2082.pdf) using the SpaCy Matcher component.

Passing `extended=True` to the `HyponymDetector` will use the extended set of hearst patterns, which include higher recall but lower precision hyponymy relations (e.g X compared to Y, X similar to Y, etc).

This component produces a doc level attribute on the spacy doc: `doc._.hearst_patterns`, which is a list containing tuples of extracted hyponym pairs. The tuples contain:

- The relation rule used to extract the hyponym (type: `str`)
- The more general concept  (type: `spacy.Span`)
- The more specific concept (type: `spacy.Span`)


#### Usage:

```python
import spacy
from scispacy.hyponym_detector import HyponymDetector

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})

doc = nlp("Keystone plant species such as fig trees are good for the soil.")

print(doc._.hearst_patterns)
>>> [('such_as', Keystone plant species, fig trees)]
```

## Extending scispaCy to new databases and ontologies

Arbitrary databases and ontologies can be loaded into scispaCy using the
external integration with [pyobo](https://github.com/biopragmatics/pyobo) following
`pip install "pyobo>=0.12.9"`. In the following example, the HGNC database
is used to link genes to standard identifiers.

```python
import pyobo
import spacy
from scispacy.linking import EntityLinker
from tabulate import tabulate

linker: EntityLinker = pyobo.get_scispacy_entity_linker("hgnc", filter_for_definitions=False)

# now, put it all together with a NER model
nlp = spacy.load("en_core_web_sm")

text = (
    "RAC(Rho family)-alpha serine/threonine-protein kinase "
    "is an enzyme that in humans is encoded by the AKT1 gene."
)
doc = linker(nlp(text))

rows = [
    (
        span,
        span.start_char,
        span.end_char,
        f"[{identifier}](https://bioregistry.io/{identifier})",
        score,
    )
    for span in doc.ents
    for identifier, score in span._.kb_ents
]
print(tabulate(rows, headers=["text", "start", "end", "prefix", "identifier"], tablefmt="github"))
```

| text | start | end | identifier                                  |    score |
| ---- | ----- | --- |---------------------------------------------| -------: |
| AKT1 | 100   | 104 | [hgnc:391](https://bioregistry.io/hgnc:391) |        1 |
| AKT1 | 100   | 104 | [hgnc:392](https://bioregistry.io/hgnc:392) | 0.776504 |
| AKT1 | 100   | 104 | [hgnc:393](https://bioregistry.io/hgnc:393) | 0.764049 |

For more information, see
[the tutorial](https://pyobo.readthedocs.io/en/latest/scispacy.html).


## Citing

If you use ScispaCy in your research, please cite [ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing](https://www.semanticscholar.org/paper/ScispaCy%3A-Fast-and-Robust-Models-for-Biomedical-Neumann-King/de28ec1d7bd38c8fc4e8ac59b6133800818b4e29). Additionally, please indicate which version and model of ScispaCy you used so that your research can be reproduced.
```
@inproceedings{neumann-etal-2019-scispacy,
    title = "{S}cispa{C}y: {F}ast and {R}obust {M}odels for {B}iomedical {N}atural {L}anguage {P}rocessing",
    author = "Neumann, Mark  and
      King, Daniel  and
      Beltagy, Iz  and
      Ammar, Waleed",
    booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5034",
    doi = "10.18653/v1/W19-5034",
    pages = "319--327",
    eprint = {arXiv:1902.07669},
    abstract = "Despite recent advances in natural language processing, many statistical models for processing text perform extremely poorly under domain shift. Processing biomedical and clinical text is a critically important application area of natural language processing, for which there are few robust, practical, publicly available models. This paper describes scispaCy, a new Python library and models for practical biomedical/scientific text processing, which heavily leverages the spaCy library. We detail the performance of two packages of models released in scispaCy and demonstrate their robustness on several tasks and datasets. Models and code are available at https://allenai.github.io/scispacy/.",
}
```

ScispaCy is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
