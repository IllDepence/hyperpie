""" Util annotation classes
"""

import uuid
from collections import OrderedDict


def surface_form_dict(s_id, surf_form, start, end):
    """ Create a surface form dict.
    """

    surface_form = OrderedDict({
      "id": s_id,
      "surface_form": surf_form,
      "start": start,
      "end": end
    })

    return surface_form


def entity_dict(e_id, e_type):
    """ Create an entity dict.
    """

    annot = OrderedDict({
      "id": e_id,
      "type": e_type,
      "subtype": None,
      "surface_forms":
      [
      ]
    })

    return annot


def relation_evidence_dict(src_srf, trg_srf):
    """ Create a relation evidence dict.
    """

    relation_evidence = OrderedDict({
        "id": str(uuid.uuid4()),
        "source_surface_form": src_srf["id"],
        "target_surface_form": trg_srf["id"],
        "evidence_sentence": None,
        "start": None,
        "end": None
    })

    return relation_evidence


def relation_dict(src_e_id, trg_e_id, src_srfs, trg_srfs):
    """ Create a relation dict.
    """

    relation = OrderedDict({
        "id": str(uuid.uuid4()),
        "source": src_e_id,
        "target": trg_e_id,
        "evidences": []
    })

    for src_srf_id in src_srfs:
        for trg_srf_id in trg_srfs:
            relation["evidences"].append(
                relation_evidence_dict(src_srf_id, trg_srf_id)
            )

    return relation


def empty_para_annotation(
    annotator_id, document_id, paragraph_index, para_text
):
    """ Create an empty annotation dict.
    """

    empty_para_annot = OrderedDict({
      "annotator_id": annotator_id,
      "document_id": document_id,
      "paragraph_index": paragraph_index,
      "text": para_text,
      "annotation":
      {
        "entities": {},
        "relations": {}
      },
      "annotation_raw": []
    })

    return empty_para_annot
