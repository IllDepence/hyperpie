""" Deal with LLM output and make it compatible with the evaluation script.
"""

import sys
import json
import yaml


# LLM prompt output format (YAML)
#
# ---
# - text_contains_entities: true/false
# - entities (datasets, models, methods, loss functions, regularization techniques):  # noqa: E501
#     - entity<N>:
#         name: <entity name>
#         type: <entity type>
#         has_parameters: true/false
#         parameters:
#             - parameter<N>:
#                 name: <parameter name>
#                 value: <parameter value>/null
#                 context: <value context>/null
# ...

# Evaluation script input format (JSON)
#
# list of dicts w/
# "annotation": {
#     "entities": {
#         "aN": {
#             "id": <entity ID>,
#             "surface_forms": [
#                 {
#                     "id": <surface form ID>,
#                     "surface_form": <entity name>,
#                     "start": <offset start>,
#                     "end": <offset end>
#                 },
#                 ...
#             ]
#         },
#         ...
#     }
#     "relations": {
#         "rN": {
#             "source": <source entity ID>,
#             "target": <target entity ID>,
#          },
#          ...
#     }
# }


def surface_form_dict(s_id, surf_form, start, end):

    surface_form = {
      "id": s_id,
      "surface_form": surf_form,
      "start": start,
      "end": end
    }

    return surface_form


def annotation_dict(e_id, surf_id):

    annot = {
      "id": e_id,
      "type": "",
      "subtype": None,
      "surface_forms":
      [
      ]
    }

    return annot


def relation_dict(r_id, source_id, target_id):

    relation = {
        "id": r_id,
        "source": source_id,
        "target": target_id,
        "evidences": []
    }

    return relation


def empty_para_annotation():
    """ Return empty annotation dict.

    Returns:
        dict: Empty annotation dict.
    """

    empty_para_annot = {
      "annotator_id": "",
      "document_id": "",
      "paragraph_index": -1,
      "text": "",
      "annotation":
      {
        "entities": {},
        "relations": {}
      },
      "annotation_raw": []
    }

    return empty_para_annot


def convert(llm_output_yaml):
    """ Convert LLM output to evaluation script input format.

    Args:
        llm_output (str): LLM output in YAML format.

    Returns:
        str: Evaluation script input in JSON format.
    """

    try:
        llm_output = yaml.load(llm_output_yaml, Loader=yaml.Loader)
    except yaml.YAMLError as e:
        print('Error parsing LLM output YAML:')
        print(e)
        return None

    has_entities_key = 'text_contains_entities'
    if type(llm_output) != list:
        # unexpected format
        print(
            f'Expected list as top-level YAML element, '
            f'got {type(llm_output)}'
        )
        sys.exit(1)
    elif len(llm_output) == 1:
        # probably expected format
        try:
            assert (
                has_entities_key in llm_output[0].keys() and
                type(llm_output[0][has_entities_key]) == bool
            )
        except AssertionError:
            print(
                f'Expected key "{has_entities_key}" in LLM output, '
                f'got {llm_output}'
            )
            sys.exit(1)
    elif len(llm_output) != 2:
        # unexpected format
        print(
            f'Expected list of length 1 or 2 as top-level YAML element, '
            f'got {llm_output}'
        )
        sys.exit(1)
        # NOTE: might be possible to just get the two dict elements
        #       needed here and ignore the rest
    else:
        # expected format, good to proceed
        pass

    out = empty_para_annotation()

    if not llm_output[0][has_entities_key]:
        # If there are no entities, return the annotation dict empty
        return out

    entity_key = (
        'entities (datasets, models, methods, loss functions, '
        'regularization techniques)'
    )

    llm_edict = llm_output[1]

    if type(llm_edict) != dict:
        print(
            f'Expected dict as second element of LLM output, '
            f'got {type(llm_edict)}'
        )
        if type(llm_edict) == list:
            print('It is a list. Assuming it is a list of dicts.')
            # TODO: continue here
            #      - check that it is a list of dicts
            #      - treat it as the list of entity dicts
    if len(llm_edict.keys()) == 0:
        print(
            f'No entities even though {has_entities_key} is true. '
        )
        return out
    if len(llm_edict.keys()) > 1:
        print(
            f'More than one entity key in LLM output: {llm_edict.keys()}.'
        )
    if entity_key not in llm_edict.keys():
        print(
            f'Expected key "{entity_key}" in LLM output, '
            f'got {llm_output}. Using first key.'
        )
        llm_edict = llm_edict[list(llm_edict.keys())[0]]

    # example
    # [{'entity1': {'name': 'SciBERT-base',
    #    'type': 'model',
    #    'has_parameters': False,
    #    'parameters': None}},
    #  {'entity2': {'name': 'BiLSTM',
    #    'type': 'model',
    #    'has_parameters': True,
    #    'parameters': [{'parameter1': {'name': 'hidden state',
    #       'value': '128-d',
    #       'context': None}}]}},


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python llm_eval.py <llm_output>')
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        llm_output_dict = json.load(f)

    llm_output = llm_output_dict['completion']['choices'][0]['text']

    convert(llm_output)
