from typing import List, Dict, Any

"""
BSD 3-Clause License

Copyright (c) 2020, Fourthought
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

hypernym = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
hyponym = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
punct = {"IS_PUNCT": True, "OP": "?"}
det = {"ORTH": "*", "OP": "*"}

BASE_PATTERNS: List[Dict[str, Any]] = [
    # '(NP_\\w+ (, )?such as (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "such_as",
        "pattern": [hypernym, punct, {"LEMMA": "such"}, {"LEMMA": "as"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "include",
        "pattern": [hypernym, punct, {"LEMMA": "include"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "especially",
        "pattern": [hypernym, punct, {"LEMMA": "especially"}, det, hyponym],
        "position": "first",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last'
    {
        "label": "other",
        "pattern": [
            hyponym,
            punct,
            {"LEMMA": {"IN": ["and", "or"]}},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
]

EXTENDED_PATTERNS = [
    # '(NP_\\w+ (, )?which may include (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "which_may_include",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "may"},
            {"LEMMA": "include"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?which be similar to (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "which_be_similar_to",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "be"},
            {"LEMMA": "similar"},
            {"LEMMA": "to"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?example of this be (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "example_of_this_be",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "example"},
            {"LEMMA": "of"},
            {"LEMMA": "this"},
            {"LEMMA": "be"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?type (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "type",
        "pattern": [hypernym, punct, {"LEMMA": "type"}, punct, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?mainly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "mainly",
        "pattern": [hypernym, punct, {"LEMMA": "mainly"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?mostly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "mostly",
        "pattern": [hypernym, punct, {"LEMMA": "mostly"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?notably (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "notably",
        "pattern": [hypernym, punct, {"LEMMA": "notably"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?particularly (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "particularly",
        "pattern": [hypernym, punct, {"LEMMA": "particularly"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?principally (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "principally",
        "pattern": [hypernym, punct, {"LEMMA": "principally"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?in particular (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "in_particular",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "in"},
            {"LEMMA": "particular"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?except (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "except",
        "pattern": [hypernym, punct, {"LEMMA": "except"}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?other than (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "other_than",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": {"IN": ["other", "oth"]}},
            {"LEMMA": "than"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?e.g. (, )?(NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "eg",
        "pattern": [hypernym, punct, {"LEMMA": {"IN": ["e.g.", "eg"]}}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?i.e. (, )?(NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "ie",
        "pattern": [hypernym, punct, {"LEMMA": {"IN": ["i.e.", "ie"]}}, det, hyponym],
        "position": "first",
    },
    # '(NP_\\w+ (, )?for example (, )?(NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "for_example",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "for"},
            {"LEMMA": "example"},
            punct,
            det,
            hyponym,
        ],
        "position": "first",
    },
    # 'example of (NP_\\w+ (, )?be (NP_\\w+ ? '(, )?(and |or )?)+)', 'first'
    {
        "label": "example_of_be",
        "pattern": [
            {"LEMMA": "example"},
            {"LEMMA": "of"},
            hypernym,
            punct,
            {"LEMMA": "be"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?like (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "like",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "like"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # 'such (NP_\\w+ (, )?as (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "such_NOUN_as",
        "pattern": [
            {"LEMMA": "such"},
            hypernym,
            punct,
            {"LEMMA": "as"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?whether (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "whether",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "whether"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?compare to (NP_\\w+ ? (, )?(and |or )?)+)', 'first'
    {
        "label": "compare_to",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "compare"},
            {"LEMMA": "to"},
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )?among -PRON- (NP_\\w+ ?(, )?(and |or )?)+)', 'first'
    {
        "label": "among_-PRON-",
        "pattern": [
            hypernym,
            punct,
            {"LEMMA": "among"},
            {"LEMMA": "-PRON-"},
            det,
            det,
            hyponym,
        ],
        "position": "first",
    },
    # '(NP_\\w+ (, )? (NP_\\w+ ? (, )?(and |or )?)+ for instance)', 'first'
    {
        "label": "for_instance",
        "pattern": [
            hypernym,
            punct,
            det,
            hyponym,
            {"LEMMA": "for"},
            {"LEMMA": "instance"},
        ],
        "position": "first",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?any other NP_\\w+)', 'last'
    {
        "label": "and-or_any_other",
        "pattern": [
            det,
            hyponym,
            punct,
            {"DEP": "cc"},
            {"LEMMA": "any"},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?some other NP_\\w+)', 'last'
    {
        "label": "some_other",
        "pattern": [
            det,
            hyponym,
            punct,
            {"DEP": "cc", "OP": "?"},
            {"LEMMA": "some"},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
    # '((NP_\\w+ ?(, )?)+(and |or )?be a NP_\\w+)', 'last'
    {
        "label": "be_a",
        "pattern": [
            det,
            hyponym,
            punct,
            {"LEMMA": "be"},
            {"LEMMA": "a"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "like_other",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?like other NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "like"},
            {"LEMMA": {"IN": ["other", "oth"]}},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "one_of_the",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?one of the NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "one"},
            {"LEMMA": "of"},
            {"LEMMA": "the"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "one_of_these",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?one of these NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "one"},
            {"LEMMA": "of"},
            {"LEMMA": "these"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "one_of_those",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?one of those NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"DEP": "cc", "OP": "?"},
            {"LEMMA": "one"},
            {"LEMMA": "of"},
            {"LEMMA": "those"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "be_example_of",
        "pattern": [
            # '((NP_\\w+ ?(, )?)+(and |or )?be example of NP_\\w+)',
            # added optional "an" to spaCy pattern for singular vs. plural
            # 'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "be"},
            {"LEMMA": "an", "OP": "?"},
            {"LEMMA": "example"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "which_be_call",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which be call NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "be"},
            {"LEMMA": "call"},
            hypernym,
        ],
        "position": "last",
    },
    #
    {
        "label": "which_be_name",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which be name NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "be"},
            {"LEMMA": "name"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "a_kind_of",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and|or)? a kind of NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "a"},
            {"LEMMA": "kind"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
    #                     '((NP_\\w+ ?(, )?)+(and|or)? kind of NP_\\w+)', - combined with above
    #                     'last'
    {
        "label": "form_of",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and|or)? form of NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "a", "OP": "?"},
            {"LEMMA": "form"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "which_look_like",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which look like NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "look"},
            {"LEMMA": "like"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "which_sound_like",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )?which sound like NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "which"},
            {"LEMMA": "sound"},
            {"LEMMA": "like"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "type",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and |or )? NP_\\w+ type)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "type"},
            hypernym,
        ],
        "position": "last",
    },
    {
        "label": "compare_with",
        "pattern": [
            #                     '(compare (NP_\\w+ ?(, )?)+(and |or )?with NP_\\w+)',
            #                     'last'
            {"LEMMA": "compare"},
            det,
            hyponym,
            punct,
            {"LEMMA": "with"},
            hypernym,
        ],
        "position": "last",
    },
    #             {"label" : "as", "pattern" : [
    # #                     '((NP_\\w+ ?(, )?)+(and |or )?as NP_\\w+)',
    # #                     'last'
    #                 hyponym, punct, {"LEMMA" : "as"}, hypernym
    #             ], "position" : "last"},
    {
        "label": "sort_of",
        "pattern": [
            #                     '((NP_\\w+ ?(, )?)+(and|or)? sort of NP_\\w+)',
            #                     'last'
            det,
            hyponym,
            punct,
            {"LEMMA": "sort"},
            {"LEMMA": "of"},
            hypernym,
        ],
        "position": "last",
    },
]
