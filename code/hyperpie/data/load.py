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


def load_annotated(filtered=True):
    """ Load (preprocessed) annotated data paragraphs.

        If `filtered` is True, load data filtered to only include
        “full” annotations (a<-p<-v[<-c]).
    """

    if filtered:
        fp = settings.annot_filtered_fp
    else:
        fp = settings.annot_prep_fp

    with open(fp) as f:
        return json.load(f)
