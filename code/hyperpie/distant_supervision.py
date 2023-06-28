""" Based on ground truth data use string matching on unannotated data.
"""

import re
from hyperpie import data, settings


def annotate(
    ground_truth_fp=None,
    target_para_fp=None
):
    """ Annotate unannotated data based on ground truth data.

        Parameters
            ground_truth_fp, target_para_fp
        default to
            settings.annot_prep_fp, settings.filtered_unannot_fp
        respectively.
    """

    if ground_truth_fp is None:
        ground_truth_fp = settings.annot_prep_fp
    if target_para_fp is None:
        target_para_fp = settings.filtered_unannot_fp

    airf_para_pairs = data.get_artifact_param_surface_form_pairs(
        paras_fp=ground_truth_fp
    )
    target_para_fp - settings.filtered_unannot_fp  # FIXME: set to unfiltered
    target_paras = data.load_unannotated(
        transformed_pprs_fp=target_para_fp
    )

    # go though paras and annotate
    for para in target_paras:
        text = para['text']
        # annotate any of
        # - known artifact param pair (anywhere in paragraph)
        # - known paragraph + closest number (within the same sentence)
        # TODO:
        # - outsource annotation creation fuctions from llm.convert to
        #   some util sub-ackage
        # - iterate over known artifact param pairs and create annotations
        #   (create regex patts w/ word boundaries)
        # - split para into sentences, iterate over known params, look for
        #   closest number and create annotation
