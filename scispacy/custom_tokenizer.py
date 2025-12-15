from typing import List

from spacy.lang import char_classes
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from spacy.util import (
    compile_infix_regex,
    compile_prefix_regex,
    compile_suffix_regex,
)
from scispacy.consts import ABBREVIATIONS


def remove_new_lines(text: str) -> str:
    """Used to preprocess away new lines in the middle of words. This function
       is intended to be called on a raw string before it is passed through a
       spaCy pipeline

    @param text: a string of text to be processed
    """
    text = text.replace("-\n\n", "")
    text = text.replace("- \n\n", "")
    text = text.replace("-\n", "")
    text = text.replace("- \n", "")
    return text


def combined_rule_prefixes() -> List[str]:
    """Helper function that returns the prefix pattern for the tokenizer.
    It is a helper function to accomodate spacy tests that only test
    prefixes.
    """
    # add lookahead assertions for brackets (may not work properly for unbalanced brackets)
    prefix_punct = char_classes.PUNCT.replace("|", " ")
    prefix_punct = prefix_punct.replace(r"\(", r"\((?![^\(\s]+\)\S+)")
    prefix_punct = prefix_punct.replace(r"\[", r"\[(?![^\[\s]+\]\S+)")
    prefix_punct = prefix_punct.replace(r"\{", r"\{(?![^\{\s]+\}\S+)")

    prefixes = (
        ["§", "%", "=", r"\+"]
        + char_classes.split_chars(prefix_punct)
        + char_classes.LIST_ELLIPSES
        + char_classes.LIST_QUOTES
        + char_classes.LIST_CURRENCY
        + char_classes.LIST_ICONS
    )
    return prefixes


def combined_rule_tokenizer(nlp: Language) -> Tokenizer:
    """Creates a custom tokenizer on top of spaCy's default tokenizer. The
    intended use of this function is to replace the tokenizer in a spaCy
    pipeline like so:

         nlp = spacy.load("some_spacy_model")
         nlp.tokenizer = combined_rule_tokenizer(nlp)

    @param nlp: a loaded spaCy model
    """
    # remove the first hyphen to prevent tokenization of the normal hyphen
    hyphens = char_classes.HYPHENS.replace("-|", "", 1)

    infixes = (
        char_classes.LIST_ELLIPSES
        + char_classes.LIST_ICONS
        + [
            r"×",  # added this special x character to tokenize it separately
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}])\.(?=[{au}])".format(
                al=char_classes.ALPHA_LOWER, au=char_classes.ALPHA_UPPER
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=char_classes.ALPHA),
            r'(?<=[{a}])[?";:=,.]*(?:{h})(?=[{a}])'.format(
                a=char_classes.ALPHA, h=hyphens
            ),
            # removed / to prevent tokenization of /
            r'(?<=[{a}"])[:<>=](?=[{a}])'.format(a=char_classes.ALPHA),
        ]
    )

    prefixes = combined_rule_prefixes()

    # add the last apostrophe
    quotes = char_classes.LIST_QUOTES.copy() + ["’"]

    # add lookbehind assertions for brackets (may not work properly for unbalanced brackets)
    suffix_punct = char_classes.PUNCT.replace("|", " ")
    # These lookbehinds are commented out because they are variable width lookbehinds, and as of spacy 2.1,
    # spacy uses the re package instead of the regex package. The re package does not support variable width
    # lookbehinds. Hacking spacy internals to allow us to use the regex package is doable, but would require
    # creating our own instance of the language class, with our own Tokenizer class, with the from_bytes method
    # using the regex package instead of the re package
    # suffix_punct = suffix_punct.replace(r"\)", r"(?<!\S+\([^\)\s]+)\)")
    # suffix_punct = suffix_punct.replace(r"\]", r"(?<!\S+\[[^\]\s]+)\]")
    # suffix_punct = suffix_punct.replace(r"\}", r"(?<!\S+\{[^\}\s]+)\}")

    suffixes = (
        char_classes.split_chars(suffix_punct)
        + char_classes.LIST_ELLIPSES
        + quotes
        + char_classes.LIST_ICONS
        + ["'s", "'S", "’s", "’S", "’s", "’S"]
        + [
            r"(?<=[0-9])\+",
            r"(?<=°[FfCcKk])\.",
            r"(?<=[0-9])(?:{})".format(char_classes.CURRENCY),
            # this is another place where we used a variable width lookbehind
            # so now things like 'H3g' will be tokenized as ['H3', 'g']
            # previously the lookbehind was (^[0-9]+)
            r"(?<=[0-9])(?:{u})".format(u=char_classes.UNITS),
            r"(?<=[0-9{}{}(?:{})])\.".format(
                char_classes.ALPHA_LOWER, r"%²\-\)\]\+", "|".join(quotes)
            ),
            # add |\d to split off the period of a sentence that ends with 1D.
            r"(?<=[{a}|\d][{a}])\.".format(a=char_classes.ALPHA_UPPER),
        ]
    )

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    # Update exclusions to include these abbreviations so the period is not split off
    exclusions = {
        abbreviation: [{ORTH: abbreviation}] for abbreviation in ABBREVIATIONS
    }
    tokenizer_exceptions = nlp.Defaults.tokenizer_exceptions.copy()
    tokenizer_exceptions.update(exclusions)

    tokenizer = Tokenizer(
        nlp.vocab,
        tokenizer_exceptions,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,  # type: ignore
    )
    return tokenizer
