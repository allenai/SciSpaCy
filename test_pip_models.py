

import spacy


def test_scispacy_models_can_load():

    small = spacy.load("en_core_sci_sm")
    medium = spacy.load("en_core_sci_md")
    craft = spacy.load("en_ner_craft_md")
    jnlpba = spacy.load("en_ner_jnlpba_md")
    bc5cdr = spacy.load("en_ner_bc5cdr_md")
    bionlp = spacy.load("en_ner_bionlp13cg_md")
    print("All models can load!")

if __name__ == "__main__":
    test_scispacy_models_can_load()
