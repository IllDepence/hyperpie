""" Load data
"""

import json
from hyperpie import settings


def load_annotated_raw():
    """ Load “raw” data with Web Annotation Data Model format
        annotations as output by annotation UI.
    """

    with open(settings.annot_raw_fp) as f:
        return json.load(f)


def load_annotated(only_full=False):
    """ Load (preprocessed) annotated data paragraphs.

        If `filtered` is True, load data filtered to only include
        “full” annotations (a<-p<-v[<-c]).
    """

    if only_full:
        fp = settings.annot_onlyfull_fp
    else:
        fp = settings.annot_prep_fp

    with open(fp) as f:
        return json.load(f)
