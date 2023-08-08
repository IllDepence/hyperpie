""" Convert preprocessed data into PL-Marker format.
"""

import json
import os
import re
import sys
from collections import OrderedDict
from nltk.tokenize import sent_tokenize


def convert(annots_path):
    # load and pre-process annotated text segments
    save_path = '../data/'
    annots_fn = os.path.basename(annots_path)
    annots_fn_base, ext = os.path.splitext(annots_fn)
    annots_processed_fn = f'{annots_fn_base}_processed{ext}'

    # TODO
    # - tokenize paragraph into sentences while keeping track of offsets
    # - iterate over entities
    #   - iterate over surface forms
    #     - create "ner" dict entries
    #     - create "clusters" dict entries
    # - iterate over relations
    #   - create "relations" dict
    #     - open question: how to handle
    #       1. relations accross sentence boundaries
    #       2. relations between entities with multiple surface forms

    # save converted annotations
    with open(os.path.join(save_path, annots_processed_fn), 'w') as f:
        json.dump(annots_processed, f)


if __name__ == '__main__':
    # check command line arguments
    if len(sys.argv) != 2:
        print(
            'Usage: python hyperpie/data/convert_plmarker.py '
            '/path/to/annots.json'
        )
        sys.exit(1)
    annots_path = sys.argv[1]
    convert(annots_path)
