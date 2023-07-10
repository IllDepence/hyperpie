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


def raw_annotation_to_text(annot):
    """ Return a textual representation of an annotation for a
        whole context.
    """

    def get_annotated_text(sngl_annot):
        text = False
        for s in sngl_annot['target']['selector']:
            if s['type'] == 'TextQuoteSelector':
                text = s['exact']
        return text

    id_to_e_dict = dict()
    simple_rel_list = []
    entities = []
    parameters = []
    values = []
    contexts = []

    annot_list = annot['annotation']
    for a in annot_list:
        if 'body' in a and a['body'] and len(a['body']) > 0:
            typ = a['body'][0]['value']
            if typ[0] == "a":
                text = get_annotated_text(a)
                id_to_e_dict[a['id']] = (text, typ)
                entities.append(text)
            elif typ[0] == "p":
                text = get_annotated_text(a)
                id_to_e_dict[a['id']] = (text, typ)
                parameters.append(text)
            elif typ[0] == "v":
                text = get_annotated_text(a)
                id_to_e_dict[a['id']] = (text, typ)
                values.append(text)
            elif typ[0] == "c":
                text = get_annotated_text(a)
                id_to_e_dict[a['id']] = (text, typ)
                contexts.append(text)
            elif typ == 'r':
                simple_rel_list.append(
                    (a['target'][0]['id'], a['target'][1]['id'])
                )
    annotations = "Annotations:\n"
    entities_str = "Research artifact: " + ", ".join(entities) + "\n"
    parameters_str = "Parameters: " + ", ".join(parameters) + "\n"
    values_str = "Values: " + ", ".join(values) + "\n"
    contexts_str = "Contexts: " + ", ".join(contexts) + "\n"
    ret = annot['context'] + '\n\n' + annotations + \
        entities_str + parameters_str + values_str + contexts_str
    for rel in simple_rel_list:
        if (rel[1] in id_to_e_dict) and rel[0] in id_to_e_dict:
            tail = id_to_e_dict[rel[0]]
            head = id_to_e_dict[rel[1]]
            ret += '{}[{}] → {}[{}]\n'.format(
                tail[0], tail[1], head[0], head[1]
            )

    return ret


def _get_annot_surface_forms(entity_dict):
    surface_forms = [
        f'"{sf["surface_form"]}"' for sf in entity_dict["surface_forms"]
    ]
    return " / ".join(surface_forms)


def _print_rel_chain(entities, relation):
    source_id = relation["source"]
    target_id = relation["target"]
    print(f'[{target_id}] {_get_annot_surface_forms(entities[target_id])}\n↑')
    print(f'[{source_id}] {_get_annot_surface_forms(entities[source_id])}\n')


def print_annotation(data):
    entities = data["annotation"]["entities"]
    relations = data["annotation"]["relations"]

    # Finally, we'll print all the chains of relations
    for relation in relations.values():
        _print_rel_chain(entities, relation)
