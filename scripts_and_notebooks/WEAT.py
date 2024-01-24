'''
Author: Mariana Chaves
Date: August 2022

We apply WEAT to measure biases in our word2vec model, which was trained from scratch in the Cornell corpus. 
See the results of this code in WEAT_output.txt.

For human subjects, a method to measure biases is the Implicit Association Tests (IAT) which attempts to reveal and measure the strength of the connection between concepts through individuals' behaviour.
The IAT relies on the hypothesis that humans tend to perform better in association tasks when those align with preconceived biases.
For example, subjects should be quicker to tag the word "insect" with the label "disgusting" than with the label "pleasant".
The IAT proposes to quantify this phenomenon as a proxy for implicit biases.

It is possible to obtain similar measures for our embedder with an analogous statistical test, the Word-Embedding Association Test (WEAT).
When translating the IAT paradigm to word embedders, the WEAT can exploit the fact that for these models the association between words is quite evident since the semantic space is built to encode such relations in its geometry.
The method, thus, reduces to computing statistics similar to those employed by the IAT using a suitable concept of distance in the semantic space.
More precisely, for our case, the test considers four groups of words, two with terms that convey a target concept (e.g. ``flowers'' and ``insects'') and two representing attributes (e.g. ``beauty'' and ``disgust'').
Statistics are collected on the distance of pairs of words, one from a target group and one for an attribute group.
Finally, the effect size is expressed in terms of Cohen's d, with conventional small, medium, and high values being 0.2, 0.4, and 0.8, respectively.
'''

import random
import gensim

from wefe.datasets import load_weat
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.utils import run_queries
from wefe.word_embedding_model import WordEmbeddingModel


word2vec_model = WordEmbeddingModel(
    gensim.models.KeyedVectors.load("results/cornell_prepro.wordvectors", mmap='r'),
    'movie_dialogs_W2V'
)

terms = load_weat()

instruments = terms['instruments']
missing_instruments = ['bagpipe', 'lute', 'trombone', 'banjo', 'harmonica', 'mandolin', 'bassoon', 'harp', 'oboe', 'fiddle', 'harpsichord', 'bongo', 'flute']
for target in missing_instruments:
    instruments.remove(target)

weapons = terms['weapons']
missing_weapons = ['dagger', 'harpoon', 'hatchet', 'firearm', 'teargas', 'slingshot']
for target in missing_weapons:
    weapons.remove(target)

pleasant = terms['pleasant_5']
unpleasant = terms['unpleasant_5']

for l in [weapons, pleasant, unpleasant]:
    random.shuffle(l)
    del l[0:(len(l) - len(instruments))]

assert len(instruments) == len(weapons)
assert len(instruments) == len(pleasant)
assert len(instruments) == len(unpleasant)

print(terms['male_terms'])
print(terms['female_terms'])
print(terms['male_names'])
print(terms['female_names'])
print(terms['science'])
print(terms['arts'])
print(terms['career'])
print(terms['family'])
print(instruments)
print(weapons)
print(pleasant)
print(unpleasant)

gender_occupation_queries = [
    Query(
        [terms['male_terms'], terms['female_terms']],
        [terms['science'], terms['arts']],
        ['Male terms', 'Female terms'],
        ['Science', 'Arts']
    ),
    Query(
        [terms['male_terms'], terms['female_terms']],
        [terms['career'], terms['family']],
        ['Male terms', 'Female terms'],
        ['Career', 'Family']
    ),
    Query(
        [terms['male_names'], terms['female_names']],
        [terms['science'], terms['arts']],
        ['Male names', 'Female names'],
        ['Science', 'Arts']
    ),
    Query(
        [terms['male_names'], terms['female_names']],
        [terms['career'], terms['family']],
        ['Male names', 'Female names'],
        ['Career', 'Family']
    ),
    Query(
        [terms['male_names'], terms['female_names']],
        [terms['arts'], terms['science']],
        ['Male names', 'Female names'],
        ['Arts', 'Science']
    ),
    Query(
        [terms['male_names'], terms['female_names']],
        [terms['family'], terms['career']],
        ['Male names', 'Female names'],
        ['Family', 'Career']
    ),
    Query(
        [instruments, weapons],
        [pleasant, unpleasant],
        ['Musical instruments', 'Weapons'],
        ['Pleasant', 'Unpleasant']
    )
]

weat = WEAT()
results = [
    weat.run_query(
        query,
        word2vec_model,
        calculate_p_value=True,
        warn_not_found_words=True,
        preprocessors=[{"lowercase": True}]
    )
    for query in gender_occupation_queries
]


for res in results:
    print(res)
