""" Deal with LLM output and make it compatible with the evaluation script.

    Use

        convert(input_paragraph, llm_output_yaml)

    to convert LLM output to the format expected by the evaluation function.
"""

import copy
import re
import sys
import json
import uuid
import yaml
from collections import defaultdict
from difflib import SequenceMatcher
from hyperpie.llm import prompt_templates
from hyperpie.util.annot import (
    entity_dict, empty_para_annotation, relation_dict,
    surface_form_dict
)


def find_surface_forms_in_para(para_text, e_name):
    """ Find all surface forms of an entity in a paragraph.

    Args:
        para_text (str): Paragraph text.
        e_name (str): Entity name.

    Returns:
        list: List of dicts with surface form information.
    """

    # use regex to identify the start and end offset of
    # all occurrences of entity name in paragraph

    surfs = []
    e_name_patt = r'\b' + re.escape(str(e_name)) + r'\b'
    for m in re.finditer(e_name_patt, para_text):
        start = m.start()
        end = m.end()
        surf_id = f'{start}-{end}'
        surfs.append(
            surface_form_dict(
                surf_id,
                e_name,
                start,
                end
            )
         )

    return surfs


def get_coarse_structure_entries(llm_output, verbose=False):
    """ Get the coarse structure entries from LLM output.

        Expected entries:
        - text_contains_entities
        - entities (short or long dict key version)

        Optional entries:
        - annotated_text
    """

    status_dict = {
        'coarse_structure_error': False,
    }

    annotation_info = {
        'text_contains_entities': None,
        'entities': None,
    }

    has_entities_key = 'text_contains_entities'
    entities_keys = [
        'entities',
        'entities (datasets, models, methods, loss functions, '
        'regularization techniques)'
    ]

    # check if LLM output is a list or a dict
    if type(llm_output) not in [list, dict]:
        # unexpected format
        if verbose:
            print(
                f'Expected list/dict as top-level YAML element, '
                f'got {type(llm_output)}'
            )
            print(f'LLM output: {llm_output}')
        status_dict['coarse_structure_error'] = True
        return annotation_info, status_dict

    # if it’s a list, convert it to a dict
    if type(llm_output) == list:
        # convernsion works if every element is a dict with a single key
        llm_output_fixed = {}
        for elem in llm_output:
            if type(elem) != dict:
                if verbose:
                    print(
                        f'Expected list of dicts as top-level YAML element, '
                        f'got {llm_output}'
                    )
                status_dict['coarse_structure_error'] = True
                return annotation_info, status_dict
            llm_output_fixed.update(elem)
        llm_output = llm_output_fixed

    # not a list, so it’s a dict
    has_expected_keys = (
        has_entities_key in llm_output.keys() and
        (
            llm_output[has_entities_key] is False or
            (
                entities_keys[0] in llm_output.keys() or
                entities_keys[1] in llm_output.keys()
            )
        )
    )
    if not has_expected_keys:
        # unexpected format
        if verbose:
            print(
                f'Expected dict with keys {has_entities_key} and '
                f'{entities_keys}, got {llm_output.keys()}'
            )
            print(f'LLM output: {llm_output}')
        status_dict['coarse_structure_error'] = True
        return annotation_info, status_dict
    else:
        annotation_info[
            'text_contains_entities'
        ] = llm_output[has_entities_key]
        for key in entities_keys:
            if key in llm_output.keys():
                annotation_info['entities'] = llm_output[key]
                break

    return annotation_info, status_dict


def llm_output2eval_input(
        llm_output_dict,
        llm_annotated_text=None,
        matched_surface_forms=None,
        preprocessor=None,
        output_format='yaml',
        verbose=False
):
    """ Convert LLM output to evaluation script input format.

    Args:
        llm_output_dict: LLM output with completion in YAML as well as
                         the original paragraph text.
        llm_annotated_text: LLM annotated text (optional).
        matched_surface_forms: If True, surface forms are matched in
                               the text from entity names. If False,
                               they need to be provided in
                               llm_annotated_text. If None, the value
                               will be determined automatically depending
                               on whether or not llm_annotated_text is
                               provided.
        preprocessor: Preprocessor function that takes the LLM output
                      and extracts the YAML part (optional)

    Returns:
        dict: Evaluation script input in JSON format.
        dict: Status dicts with info from parsing steps
    """

    llm_output_dict = copy.deepcopy(llm_output_dict)  # create working copy
    status_dicts = {}

    # determine if surface forms need to be matched in the text
    if matched_surface_forms is None:
        if llm_annotated_text is None:
            matched_surface_forms = True
        else:
            matched_surface_forms = False

    # use preprocessor if provided
    if preprocessor is not None:
        llm_output_dict, preprocessor_status_dict = preprocessor(
            llm_output_dict
        )
        status_dicts['preprocessor'] = preprocessor_status_dict

    # convert YAML to JSON
    if output_format == 'yaml':
        llm_output, yaml2json_status_dict = yaml2json(
            llm_output_dict, verbose=verbose
        )
    elif output_format == 'json':
        llm_output, yaml2json_status_dict = parse_llm_json(
            llm_output_dict, verbose=verbose
        )
    else:
        raise ValueError(f'Unknown output format {output_format}')
    status_dicts['yaml2json'] = yaml2json_status_dict

    # input paragraph (used to determine text offsets)
    para = llm_output_dict['paragraph']

    eval_input = empty_para_annotation(
        para.get('annotator_id'),
        para['document_id'],  # required
        para.get('paragraph_index'),
        para['text']  # required
    )

    # return if parsing failed
    if llm_output is None:
        # YAML parsing failed
        return eval_input, status_dicts

    # get coarse structure entries
    annotation_info, cs_status = get_coarse_structure_entries(
        llm_output,
        verbose
    )
    status_dicts['coarse_structure'] = cs_status

    if annotation_info['text_contains_entities'] is False:
        # If there are no entities, return the annotation dict empty
        return eval_input, status_dicts

    # check types
    if not (
        type(annotation_info['text_contains_entities']) == bool and
        type(annotation_info['entities']) == list
    ):
        if verbose:
            print(
                f'Expected bool and list for coarse structure entries, '
                f'got {type(annotation_info["text_contains_entities"])} and '
                f'{type(annotation_info["entities"])}'
            )
            print(f'LLM output: {llm_output}')
        status_dicts['coarse_structure']['coarse_structure_error'] = True
        return eval_input, status_dicts

    # coarse structure checking done
    # from hereon parse entity/relation dicts and build output
    # compatible with eval script
    if verbose:
        print('coarse structure looks good :)')

    if llm_annotated_text is None:
        if not matched_surface_forms:
            print(
                f'if matched_surface_forms set to False '
                f'llm_annotated_text must be provided'
            )
            sys.exit(1)
        # “single stage” prompt, surface forms  have to
        # be string matched in the paragraph text
        eval_input = singleprompt_llm_entities2eval_input(
            para,
            annotation_info,
            eval_input,
            verbose
        )
        return eval_input, status_dicts
    else:
        if not matched_surface_forms:
            # “two stage” prompt and surface forms are given as
            # annotated text with IDs
            eval_input, cntnt_stat = twostage_llm_entities2eval_input(
                para,
                annotation_info,
                llm_annotated_text,
                eval_input,
                verbose
            )
            status_dicts['json_content'] = cntnt_stat
            return eval_input, status_dicts
        else:
            # “two stage” prompt but surface forms are requested
            # to be extracted by matching entity names in the
            # paragraph text
            eval_input, cntnt_stat = onepointfivestage_llm_entities2eval_input(
                para,
                annotation_info,
                eval_input,
                verbose
            )
            status_dicts['json_content'] = cntnt_stat
            return eval_input, status_dicts


def get_llm_text_offset_mapping(llm_text, orig_text, annot_patt):
    """ Get offset mapping from LLM “repeated” text to original text.

        (“repeated” text is the text with annotation markers removed.)

        Example:
             if llm_text = 'abc' and orig_text = 'abbc', then the mapping
             would be {0:0, 1:1, 2:3}
    """

    # remove annotations from LLM text
    # (in theory should work without, but this turns out to be more robust)
    llm_text = annot_patt.sub(r'\3', llm_text)

    annot_to_orig = {}
    # get non-overlapping matching subsequences
    blocks = SequenceMatcher(None, llm_text, orig_text).get_matching_blocks()
    for i, j, n in blocks:
        for k in range(n):
            annot_to_orig[i+k] = j+k
    return annot_to_orig


def get_llm_annotated_entities(llm_text, orig_text):
    """ Parse text annotated by LLM with entity anntations in the form of

            [e1|entity name]

        Returns:
            dict: entity annotations.
            dict: offset mapping from annotated text to original text.

        Entity annotations are directly created with the functions
        entity_dict and surface_form_dict
    """

    annot_patt = re.compile(r'\[([a-z])([0-9\.]+)\|([^]]+)]')

    # prepare offset mapping
    llm_to_orig = get_llm_text_offset_mapping(
        llm_text,
        orig_text,
        annot_patt
    )

    # parse LLM annotated text
    entities = {}
    shift = 0
    for match in annot_patt.finditer(llm_text):
        # get entity ID, surface form and offsets
        entity_type = match.group(1)
        # set 'e' type entities to 'a' type (prompt uses 'e', eval 'a')
        if entity_type == 'e':
            entity_type = 'a'
        entity_id = entity_type + match.group(2)
        surface_form = match.group(3)
        start_annot = match.start()
        end_annot = match.end()
        # determine start and end in original text (without annotations)
        # offsets are shifted as shown in the example below
        #   annotated text: "fo [e1|ba] r"
        #   original text:  "fo ba r"
        #   offset_mapping: {0: 0, 1: 1, 2: 2, 7: 3, 8: 4, 10: 5, 11: 6}
        len_mrkr_pre = 1 + len(entity_id) + 1  # [e1|
        len_mrkr_suf = 1                       # ]
        # NOTE: the LLM sometimes repeats in input text with slight
        #       modifications, e.g. spaces added/missing or punctuation
        #       changed.
        #       we therefore have use the offset mapping prepared above
        #       to undo the changes
        start_orig_pre = start_annot - shift
        end_orig_pre = end_annot - shift - len_mrkr_pre - len_mrkr_suf
        # FIXME: check if text at the offsets is actually matching and
        #        only swap to difflib version if not
        if start_orig_pre in llm_to_orig.keys():
            start_orig = llm_to_orig[start_orig_pre]
        else:
            start_orig = start_orig_pre
        if end_orig_pre in llm_to_orig.keys():
            end_orig = llm_to_orig[end_orig_pre]
        else:
            end_orig = end_orig_pre
        # create new entity dict or update existing one to add surface form
        if entity_id not in entities.keys():
            e = entity_dict(entity_id, entity_type)
        else:
            e = entities[entity_id]
        surf_id = str(uuid.uuid4())
        surf = surface_form_dict(
            surf_id,
            surface_form,
            start_orig,
            end_orig
        )

        e['surface_forms'].append(surf)
        entities[entity_id] = e
        # keep track of overall offset
        shift += len_mrkr_pre + len_mrkr_suf

    return entities


def _in_scope_artifact_type(typ):
    scope = [
        'dataset',
        'data set',
        'model',
        'method',
        'loss function',
        'loss',
        'regularization technique',
        'regularization'
    ]

    return typ in scope


def _artif_id_format_valid(aid):
    if not type(aid) == str:
        return False
    return re.match(r'^a[0-9]+$', aid) is not None


def _param_id_format_valid(pid):
    if not type(pid) == str:
        return False
    return re.match(r'^p[0-9]+\.[0-9]+$', pid) is not None


def _value_context_id_format_valid(vcid):
    if not type(vcid) == str:
        return False
    return re.match(r'^[vc][0-9]+\.[0-9]+\.[0-9]+$', vcid) is not None


def _twostage_llm_parse_yaml(annotation_info, para_text):
    """ Parse LLM generated YAML and return entities, relations,
        and a status dict.
    """

    entities = {}
    relations = []
    status_dict = {
        'num_ents_intext_notintext': [0, 0],
        'num_ent_types_valid_invalid': [0, 0],
        'num_aids_valid_invalid': [0, 0],
        'num_pids_valid_invalid': [0, 0],
        'num_vids_valid_invalid': [0, 0],
        'num_cids_valid_invalid': [0, 0],
    }
    for artf_dict in annotation_info['entities']:
        if artf_dict is None:
            continue
        # parse artifact
        if 'parameters' in artf_dict.keys():
            # already “unpacked”
            artf = artf_dict
        else:
            artf = next(iter(artf_dict.values()))
        if (
            artf is None or
            type(artf) != dict or
            'id' not in artf.keys() or
            'name' not in artf.keys() or
            'type' not in artf.keys()
        ):
            continue
        # set 'e' type entities to 'a' type (prompt uses 'e', eval 'a')
        artf['id'] = re.sub(
            r'[a-z]([0-9\.]+)',
            r'a\1',
            str(artf.get('id'))
        )
        # check id format
        if _artif_id_format_valid(artf['id']):
            status_dict['num_aids_valid_invalid'][0] += 1
        else:
            status_dict['num_aids_valid_invalid'][1] += 1
        # check if artifact is actually in the text
        if artf['name'] in para_text:
            status_dict['num_ents_intext_notintext'][0] += 1
        else:
            status_dict['num_ents_intext_notintext'][1] += 1
        # check type
        if _in_scope_artifact_type(artf['type']):
            # NOTE: might consider distinguishing between artifacts
            #       in scope + in text vs in scope but not in text
            status_dict['num_ent_types_valid_invalid'][0] += 1
        else:
            status_dict['num_ent_types_valid_invalid'][1] += 1
        # create entity
        entities[artf['id']] = {
            'id': artf['id'],
            'type': 'a',
            'name': artf['name']
        }
        if not artf.get('has_parameters', False):
            continue
        if 'parameters' not in artf or artf['parameters'] is None:
            continue
        for param_dict in artf['parameters']:
            if param_dict is None:
                continue
            # parse parameter
            if 'values' in param_dict.keys():
                # already “unpacked”
                param = param_dict
            else:
                param = next(iter(param_dict.values()))
            if (
                param is None or
                type(param) != dict or
                'id' not in param.keys() or
                'name' not in param.keys()
            ):
                continue
            # check id format
            if _param_id_format_valid(param['id']):
                status_dict['num_pids_valid_invalid'][0] += 1
            else:
                status_dict['num_pids_valid_invalid'][1] += 1
            entities[param['id']] = {
                'id': param['id'],
                'type': 'p',
                'name': param['name']
            }
            # set relation
            relations.append([
                param['id'],
                artf['id']
            ])
            if not param.get('has_values', False):
                continue
            if 'values' not in param or param['values'] is None:
                continue
            for val_dict in param['values']:
                if val_dict is None:
                    continue
                # parse value and context
                if 'context' in val_dict.keys():
                    # already “unpacked”
                    val = val_dict
                else:
                    val = next(iter(val_dict.values()))
                if (
                    val is None or
                    type(val) != dict or
                    'value_id' not in val.keys() or
                    'value' not in val.keys()
                ):
                    continue
                # check id format
                if _value_context_id_format_valid(val['value_id']):
                    status_dict['num_vids_valid_invalid'][0] += 1
                else:
                    status_dict['num_vids_valid_invalid'][1] += 1
                entities[val['value_id']] = {
                    'id': val['value_id'],
                    'type': 'v',
                    'name': val['value']
                }
                # set relation
                relations.append([
                    val['value_id'],
                    param['id']
                ])
                if (
                    'context' not in val.keys() or
                    'context_id' not in val.keys()
                ):
                    continue
                # check id format
                if _value_context_id_format_valid(val['context_id']):
                    status_dict['num_cids_valid_invalid'][0] += 1
                else:
                    status_dict['num_cids_valid_invalid'][1] += 1
                entities[val['context_id']] = {
                    'id': val['context_id'],
                    'type': 'c',
                    'name': val['context']
                }
                # set relation
                relations.append([
                    val['context_id'],
                    val['value_id']
                ])

    return entities, relations, status_dict


def twostage_llm_entities2eval_input(
    para,
    annotation_info,
    llm_annotated_text,
    eval_input,
    verbose
):
    # built dict of entities with ID, type, and name
    # list of relations (using entity IDs)
    entities, relations, entrel_status_dict = _twostage_llm_parse_yaml(
        annotation_info,
        para['text']
    )

    # get entities from LLM annotated text
    llm_entity_annots = get_llm_annotated_entities(
        llm_annotated_text,
        para['text']
    )

    # create relation annots
    rel_annots = {}
    for from_id, to_id in relations:
        # make sure both entities exist in LLM annotations
        if from_id not in llm_entity_annots or to_id not in llm_entity_annots:
            continue
        rel_dict = relation_dict(
            from_id,
            to_id,
            llm_entity_annots[from_id]['surface_forms'],
            llm_entity_annots[to_id]['surface_forms']
        )
        rel_annots[rel_dict['id']] = rel_dict

    eval_input['annotation']['entities'] = llm_entity_annots
    eval_input['annotation']['relations'] = rel_annots

    return eval_input, entrel_status_dict


def onepointfivestage_llm_entities2eval_input(
    para,
    annotation_info,
    eval_input,
    verbose
):
    """ First prompt of twostage setup was used but surface
        forms are to be determined by text matching.
    """

    # built dict of entities with ID, type, and name
    # list of relations (using entity IDs)
    entities, relations, entrel_status_dict = _twostage_llm_parse_yaml(
        annotation_info,
        para['text']
    )

    ent_annots = {}
    for e_id, ent in entities.items():
        ent_dict = entity_dict(e_id, ent['type'])
        # find surface forms
        artif_surfs = find_surface_forms_in_para(
            para['text'],
            ent['name']
        )
        ent_dict['surface_forms'] = artif_surfs
        ent_annots[ent['id']] = ent_dict

    # create relation annots
    rel_annots = {}
    for from_id, to_id in relations:
        rel_dict = relation_dict(
            from_id,
            to_id,
            [],
            []
        )
        rel_annots[rel_dict['id']] = rel_dict

    eval_input['annotation']['entities'] = ent_annots
    eval_input['annotation']['relations'] = rel_annots

    return eval_input, entrel_status_dict


def singleprompt_llm_entities2eval_input(
        para,
        annotation_info,
        eval_input,
        verbose
):

    for artf_wrapper in annotation_info['entities']:
        if artf_wrapper is None:
            continue

        # unwrap artifact dict
        artf = artf_wrapper[list(artf_wrapper.keys())[0]]

        # create the artifact entity
        artif_name = artf.get('name', None)
        if artif_name is None:
            print(f'No name for artifact: {artf}')
            continue
        elif type(artif_name) == list:
            # YAML output “blunder” when a citation marker like "[27]" is
            # identified as an artifact name and parsed as a list b/c of
            # the missing quotes
            artif_name = f'[{artif_name[0]}]'
        # check if (identically named) artifact entity already exists
        if artif_name in eval_input['annotation']['entities']:
            # not sure if this is sensible
            print('Duplicate artifact entity name, reusing existing entity')
            artif_annot = eval_input['annotation']['entities'][artif_name]
        else:
            artif_annot = entity_dict(artif_name, 'a')
        # find surface forms
        artif_surfs = find_surface_forms_in_para(
            para['text'],
            artif_name
        )
        artif_annot['surface_forms'] = artif_surfs
        eval_input['annotation']['entities'][artif_name] = artif_annot

        # check for parameters
        if (
            (not artf.get('has_parameters', False)) or
            artf['parameters'] is None
        ):
            # no parameters, just add artifact entity
            eval_input['annotation']['entities'][artif_name] = artif_annot
            continue
        # create parameter entities
        for prm_wrapper in artf['parameters']:
            # unwrap weird YAML->JSON conversion structure
            prm = prm_wrapper[list(prm_wrapper.keys())[0]]
            prm_name = prm.get('name', None)
            if prm_name is None:
                print(f'No name for parameter: {prm}')
                continue
            prm_annot = entity_dict(prm_name, 'p')
            prm_surfs = find_surface_forms_in_para(
                para['text'],
                prm_name
            )
            prm_annot['surface_forms'] = prm_surfs
            # add parameter entity
            eval_input['annotation']['entities'][prm_name] = prm_annot
            # add relation between parameter and artifact
            rel_annot = relation_dict(
                prm_name,
                artif_name,
                prm_surfs,
                artif_surfs
            )
            eval_input['annotation']['relations'][rel_annot["id"]] = rel_annot
            # check for a value
            if prm.get('value', None) is None or prm['value'] == '':
                continue
            # add value entity
            val_name = str(prm['value'])
            val_annot = entity_dict(val_name, 'v')
            val_surfs = find_surface_forms_in_para(
                para['text'],
                val_name
            )
            val_annot['surface_forms'] = val_surfs
            eval_input['annotation']['entities'][val_name] = val_annot
            # add relation between parameter and value
            rel_annot = relation_dict(
                val_name,
                prm_name,
                val_surfs,
                prm_surfs
            )
            eval_input['annotation']['relations'][rel_annot["id"]] = rel_annot
            # check for a context
            if prm.get('context', None) is None or prm['context'] == '':
                continue
            # add context entity
            ctx_name = prm['context']
            ctx_annot = entity_dict(ctx_name, 'c')
            ctx_surfs = find_surface_forms_in_para(
                para['text'],
                ctx_name
            )
            ctx_annot['surface_forms'] = ctx_surfs
            eval_input['annotation']['entities'][ctx_name] = ctx_annot
            # add relation between value and context
            rel_annot = relation_dict(
                ctx_name,
                val_name,
                ctx_surfs,
                val_surfs
            )
            eval_input['annotation']['relations'][rel_annot["id"]] = rel_annot

    if verbose:
        print(f'Found {len(eval_input["annotation"]["entities"])} entities')
        print(f'Found {len(eval_input["annotation"]["relations"])} relations')

    return eval_input


def _preprocessor_status_dict(
    no_yaml_found,
    empty_yaml,
    garbage_around_yaml
):
    """ Return a dictionary to be used as part of the return value of
        a preprocessor function for LLM output, used before YAML parsing.
    """

    return {
        'no_yaml_found': no_yaml_found,
        'empty_yaml': empty_yaml,
        'garbage_around_yaml': garbage_around_yaml
    }


def aggregate_format_stats(stats_dicts):
    """ Aggregate a list of stats return by llm_output2eval_input.
    """

    aggregate_stats = {
        'num_total': len(stats_dicts),
        'preprocessor': {
            'no_yaml_found': 0, 'empty_yaml': 0, 'garbage_around_yaml': 0
        },
        'yaml2json': {
            'parse_fail': 0, 'parsing_error_dict': defaultdict(dict)
        },
        'coarse_structure': {'coarse_structure_error': 0},
        'json_content': {
            'num_ents_intext_notintext': [0, 0],
            'num_ent_types_valid_invalid': [0, 0],
            'num_aids_valid_invalid': [0, 0],
            'num_pids_valid_invalid': [0, 0],
            'num_vids_valid_invalid': [0, 0],
            'num_cids_valid_invalid': [0, 0]
        }
    }

    for stats_dict in stats_dicts:
        # Aggregate preprocessor stats
        if 'preprocessor' in stats_dict:
            for stat_name, value in stats_dict['preprocessor'].items():
                if value is not None:
                    aggregate_stats['preprocessor'][stat_name] += value

        # Aggregate 'yaml2json' stats
        if (
            'parse_fail' in stats_dict['yaml2json'] and
            stats_dict['yaml2json']['parse_fail']
        ):
            aggregate_stats['yaml2json']['parse_fail'] += 1
        if 'parsing_error_dict' in stats_dict['yaml2json']:
            pe_dict = stats_dict['yaml2json']['parsing_error_dict']
            for error_type, msg in pe_dict.items():
                ag_pe_dict = aggregate_stats['yaml2json']['parsing_error_dict']
                if msg not in ag_pe_dict[error_type]:
                    ag_pe_dict[error_type][msg] = 0
                ag_pe_dict[error_type][msg] += 1

        if 'coarse_structure' in stats_dict:
            if stats_dict['coarse_structure']['coarse_structure_error']:
                aggregate_stats['coarse_structure'][
                    'coarse_structure_error'
                ] += 1

        if 'json_content' in stats_dict:
            # Aggregate 'json_content' stats
            for stat_name, values in stats_dict['json_content'].items():
                aggregate_stats['json_content'][stat_name] = [
                    sum(x) for x in
                    zip(values, aggregate_stats['json_content'][stat_name])
                ]

    print(_format_eval_markdown_table(aggregate_stats))

    return aggregate_stats


def _format_eval_markdown_table(data):
    paragraph_eval_keys = {
        'no_yaml_found': 'No YAML found',
        'empty_yaml': 'Empty YAML',
        'garbage_around_yaml': 'Garbage around YAML',
        'parse_fail': 'YAML parse fail',
        'coarse_structure_error': 'Coarse Structure error'
    }

    entity_eval_keys = {
        'num_ents_intext_notintext': 'Entity in text',
        'num_ent_types_valid_invalid': 'Entity type',
        'num_aids_valid_invalid': 'Artifact ID',
        'num_pids_valid_invalid': 'Parameter ID',
        'num_vids_valid_invalid': 'Value ID',
        'num_cids_valid_invalid': 'Context ID'
    }

    md = f"Paragraphs total: {data['num_total']}\n\n"
    md += "##### Paragraph eval\n\n"
    md += "| Category               | Count |\n"
    md += "| ---------------------- | ----- |\n"

    for key, value in data['preprocessor'].items():
        key_string = paragraph_eval_keys[key].ljust(22)
        md += f"| {key_string} | {str(value).rjust(5)} |\n"

    key_string = paragraph_eval_keys['parse_fail'].ljust(22)
    md += (
        f"| {key_string} | {str(data['yaml2json']['parse_fail']).rjust(5)} |\n"
    )

    key_string = paragraph_eval_keys['coarse_structure_error'].ljust(20)
    md += (
        f"| {key_string} | "
        f"{str(data['coarse_structure']['coarse_structure_error']).rjust(5)} "
        f"|\n"
    )

    md += "\n##### Entity eval\n\n"
    md += "| Criterion      | Valid | Invalid |\n"
    md += "| -------------- | ----- | ------- |\n"

    for key, values in data['json_content'].items():
        key_string = entity_eval_keys[key].ljust(14)
        md += (
            f"| {key_string} | {str(values[0]).rjust(5)} | "
            f"{str(values[1]).rjust(7)} |\n"
        )

    return md


def fix_indent(yaml):
    """ Fix YAML indentation to fit pre-defined schema.
    """

    fixed_yaml = ''
    if re.match(r'^\s*(true|false)', yaml):
        # preserve space in initial key/value pair split between
        # prompt and completion
        fixed_yaml += ' '
    indent = 0
    for line in yaml.split('\n'):
        m_ent = re.match(r'^\s*(-\s*entity.*)$', line)
        m_par = re.match(r'^\s*(-\s*parameter.*)$', line)
        m_val = re.match(r'^\s*(-\s*value.*)$', line)
        if m_ent:
            # beginning an entity list element
            indent = 2
            fixed_yaml += ' '*indent + m_ent.group(1) + '\n'
            indent = 2 + 4
        elif m_par:
            # beginning a parameter list element
            indent = 2 + 4 + 2
            fixed_yaml += ' '*indent + m_par.group(1) + '\n'
            indent = 2 + 4 + 2 + 4
        elif m_val:
            # beginning a value list element
            indent = 2 + 4 + 2 + 4 + 2
            fixed_yaml += ' '*indent + m_val.group(1) + '\n'
            indent = 2 + 4 + 2 + 4 + 2 + 4
        else:
            # continuation of the current block of key value pairs
            fixed_yaml += ' '*indent + line.lstrip() + '\n'

    return fixed_yaml.rstrip()


def falcon_yaml_extract(llm_output_dict, verbose=False):
    """ Preprocessor for Falcon output where the start of the YAML is part
        of the prompt.
    """

    llm_out = llm_output_dict['completion']['choices'][0]['text']

    # look for Falcon specific garbage
    garbage_patt = re.compile(
        r'^The (output|assistant|user)',  # typically fonud after YAML
        flags=re.I | re.M
    )
    if garbage_patt.search(llm_out):
        falcon_garbage = True
    else:
        falcon_garbage = False

    # fix Falcon specific bools w/ qoutes
    qbool_patt = re.compile(
        r'("|\')?(true|false)("|\')?',
        flags=re.I | re.M
    )
    llm_out = qbool_patt.sub(r'\2', llm_out)

    # fix Falcon specific missing space at the beginning
    llm_out = ' ' + llm_out

    llm_output_dict['completion']['choices'][0]['text'] = llm_out

    # parse YAML
    llm_output, stats = vicuna_yaml_extract(llm_output_dict)
    # adjust preprocessor stats if necessary
    stats['garbage_around_yaml'] = (
        stats['garbage_around_yaml'] or falcon_garbage
    )

    return llm_output, stats


def wizard_lm_yaml_extract(llm_output_dict, verbose=False):
    """ Preprocessor for WizardLM output where the start of the YAML is part
        of the prompt and needs to be added back.
        Also detects garbage text output after the YAML block.
    """

    llm_out = llm_output_dict['completion']['choices'][0]['text']
    # ret preprocessor stats
    lod_copy = copy.deepcopy(llm_output_dict)
    lod_copy, raw_stats = vicuna_yaml_extract(lod_copy)
    # ensure correct YAML indentation
    llm_output_dict['completion']['choices'][0]['text'] = fix_indent(llm_out)
    # get YAML to be used for parsing
    clean_llm_output, stats = vicuna_yaml_extract(llm_output_dict)
    # add proper preprocessor stats
    for k in ['no_yaml_found', 'empty_yaml', 'garbage_around_yaml']:
        stats[k] = raw_stats[k]
    return clean_llm_output, stats


def gpt3_json_extract(llm_output_dict, verbose=False):
    """ Preprocessor for GPT3 output.
    """

    llm_output_text = llm_output_dict['completion']['choices'][0]['text']

    json_code_block_patt = re.compile(
        r'```python\n?(.*)(^```$)',
        re.S | re.M
    )
    json_simple_patt = re.compile(
        r'\s*{\s*"text_contains_entities":.*}',
        re.S | re.M
    )

    m = json_code_block_patt.search(llm_output_text)
    if m is not None:
        json_text = m.group(1)
    else:
        ms = json_simple_patt.search(llm_output_text)
        if ms is not None:
            json_text = ms.group(0)
        else:
            print(llm_output_text)
            raise

    status_dict = _preprocessor_status_dict(
        None, None, None
    )

    llm_output_dict['completion']['choices'][0]['text'] = json_text

    return llm_output_dict, status_dict


def vicuna_json_extract(llm_output_dict, verbose=False):
    """ Preprocessor for Vicuna output where the start of the JSON is part
        of the prompt and needs to be added back.
        Also detects garbage text output after the JSON block.
    """

    llm_output_text = llm_output_dict['completion']['choices'][0]['text']

    # try to extract JSON block
    json_patt = re.compile(
        r"(\s*['\"]?(true|false|1|0)['\"]?,\s*\n.*?^\}$)(\n```)?(.*)",
        re.S | re.M
    )
    json_beginning_patt = re.compile(
        r"\s*['\"]?(true|false|1|0)['\"]?,\n\s+\"entities\":\s+\[\s*{",
        re.S | re.M
    )
    m = json_patt.search(llm_output_text)
    json_found = True
    if m is not None:
        json_text = m.group(1)
        garbage = m.group(4)
    else:
        m_beg = json_beginning_patt.search(llm_output_text)
        if m_beg is None:
            json_found = False
        json_text = llm_output_text
        garbage = ''  # unknown so leave empty

    # add back beginning which was part of the prompt
    json_text = '{"text_contains_entities": ' + json_text
    status_dict = _preprocessor_status_dict(
        not json_found, None, len(garbage) > 3
    )

    llm_output_dict['completion']['choices'][0]['text'] = json_text

    return llm_output_dict, status_dict


def vicuna_yaml_extract(llm_output_dict, verbose=False):
    """ Preprocessor for Vicuna output where the start of the YAML is part
        of the prompt and needs to be added back.
        Also detects garbage text output after the YAML block.
    """

    yaml_start = prompt_templates.start_completion
    yaml_end = '\n...'
    yaml_with_garbage_patt = re.compile(
        r"^(\s*['\"]?(true|false)['\"]?\n.*)\n\.\.\.\n.{7,}$",
        flags=re.S
    )
    yaml_end_patt = re.compile(r'^(.*)\.\.\.(\W+---)?$', flags=re.S)

    llm_out = llm_output_dict['completion']['choices'][0]['text']
    status_dict = _preprocessor_status_dict(None, None, None)

    # ensure clean YAML end and take stats
    m_garbage = yaml_with_garbage_patt.match(llm_out)
    m_end = yaml_end_patt.match(llm_out)
    if m_garbage:
        llm_out = m_garbage.group(1)
        status_dict['garbage_around_yaml'] = True
    elif m_end:
        llm_out = m_end.group(1)
    else:
        # assume cut off YAML, remove last line
        llm_out = '\n'.join(llm_out.split('\n')[:-1])

    yaml_full = yaml_start + llm_out + yaml_end
    llm_output_dict['completion']['choices'][0]['text'] = yaml_full

    return llm_output_dict, status_dict


def galactica_yaml_extract(llm_output_dict, verbose=False):
    """ Extract YAML part of a response that GALACTICA gave and return
        modified llm_output_dict.

        Returns a tuple of the form (llm_output_dict, preprocessor_status_dict)

    """

    gal_yaml_patt = re.compile(
        (
            r'\[YAML [\w\s]+ start\].*text_contains_entities:(.*)'
            r'\[YAML [\w\s]+ end\]'
        ),
        flags=re.S | re.I  # dot matches newlines, case insensitive
    )

    gal_yaml_patt_cut = re.compile(
        (
            r'\[YAML [\w\s]+ start\].*text_contains_entities:(.*)'
            r'$'
        ),
        flags=re.S | re.I
    )

    gal_empty_yaml_block_patt = re.compile(
        r'\[YAML [\w\s]+ start\]\s+\[YAML [\w\s]+ end\]',  # empty YAML block
        flags=re.S | re.I
    )

    gal_post_yaml_garbage_patt = re.compile(
        r'\[YAML [\w\s]+ end\].*(\w{5,}.*\n).*$',  # unwanted text after YAML
        flags=re.S | re.I
    )

    llm_output_gal = llm_output_dict['completion']['choices'][0]['text']

    gal_yaml_pre = None
    if gal_yaml_patt.search(llm_output_gal):
        # expected format
        yaml_match = gal_yaml_patt.search(llm_output_gal)
        gal_yaml_pre = yaml_match.group(1)
    elif gal_yaml_patt_cut.search(llm_output_gal):
        # expected format but cut off, match w/o ending marker
        yaml_match = gal_yaml_patt_cut.search(llm_output_gal)
        gal_yaml_pre = yaml_match.group(1)
        # remove last line
        gal_yaml_pre = '\n'.join(gal_yaml_pre.split('\n')[:-1])
    else:
        # assume it’s just YAML and hope for the best
        gal_yaml_pre = llm_output_gal

    if gal_yaml_pre is None:
        print('No YAML output found')
        return llm_output_dict, _preprocessor_status_dict(True, None, None)

    if gal_yaml_pre[-4:] in ['...', '```', '---']:
        # remove YAML/Markdown code ending
        gal_yaml_pre = gal_yaml_pre[:-4]

    gal_yaml = 'text_contains_entities:' + gal_yaml_pre

    llm_output_dict['completion']['choices'][0]['text'] = gal_yaml

    preprocess_status_dict = _preprocessor_status_dict(False, None, None)

    if gal_empty_yaml_block_patt.search(gal_yaml):
        preprocess_status_dict['empty_yaml'] = True

        # replace empty YAML block with stating no entities were found
        no_entities_yaml = 'text_contains_entities: false'
        llm_output_dict['completion']['choices'][0]['text'] = no_entities_yaml

    if gal_post_yaml_garbage_patt.search(llm_output_gal):
        preprocess_status_dict['garbage_around_yaml'] = True

    return llm_output_dict, preprocess_status_dict


def parse_llm_json(llm_output_dict, verbose=False):
    """ Try to parse LLM output JSON.

        Returns a tuple of the form (llm_output, status_dict)
    """

    status_dict = {
        'parse_fail': False,
        'parsing_error_dict': {},
    }
    leading_space_patt = re.compile(r'^[ ]*')

    # predicted annotations in JSON
    llm_output_json = llm_output_dict['completion']['choices'][0]['text']

    # try to parse
    parse_errors = {}
    llm_output = None

    try:
        llm_output = json.loads(llm_output_json)
    except json.JSONDecodeError as e_general:
        parse_errors['general'] = str(e_general)
        parse_fail = True
        # check if output template was just copied
        onlycopy_test_strs = [
            '"name": "<entity name>",',
            (
                '"type": "dataset/model/method/loss function/'
                'regularization technique",'
            ),
            '"has_parameters": true/false,',
        ]
        all_in = True
        for tstr in onlycopy_test_strs:
            if tstr not in llm_output_json:
                all_in = False
        if all_in:
            parse_errors['only_copy'] = True
        # 1.1. assume invalid escape sequence
        try:
            esc_json_lines = []
            for line in llm_output_json.split('\n'):
                m = re.match(r'^(\s+"[a-z]+": )"(.+)"(,?)$', line)
                if m:
                    key_part = m.group(1)
                    val_part = m.group(2)
                    delim_part = m.group(3)
                    # escape backsplashes in value part
                    val_part = re.sub(r'\\', r'\\\\', val_part)
                    line = f'{key_part}"{val_part}"{delim_part}'
                esc_json_lines.append(line)
            esc_json = '\n'.join(esc_json_lines)
            llm_output = json.loads(esc_json)
            parse_fail = False
        except json.JSONDecodeError as e_quotes:
            parse_errors['quotes+esc'] = str(e_quotes)
            pass  # handle further down
        # 1. 2 try to fix (assume output is cut off)
        # - remove last line
        llm_output_json = '\n'.join(llm_output_json.split('\n')[:-1])
        # - add necessary delimiters according to indent
        last_line = llm_output_json.split('\n')[-1]
        leading_spaces = leading_space_patt.match(last_line).group(0)
        num_leading_spaces = len(leading_spaces)
        if num_leading_spaces >= 18:
            llm_output_json += '}}]}}]}}]}'
        elif num_leading_spaces >= 16:
            llm_output_json += '}]}}]}}]}'
        elif num_leading_spaces >= 14:
            llm_output_json += ']}}]}}]}'
        elif num_leading_spaces >= 12:
            llm_output_json += '}}]}}]}'
        elif num_leading_spaces >= 10:
            llm_output_json += '}]}}]}'
        elif num_leading_spaces >= 8:
            llm_output_json += ']}}]}'
        elif num_leading_spaces >= 6:
            llm_output_json += '}}]}'
        elif num_leading_spaces >= 4:
            llm_output_json += '}]}'
        elif num_leading_spaces >= 2:
            llm_output_json += ']}'
        else:
            llm_output_json += '}'
        try:
            llm_output = json.loads(llm_output_json)
            parse_fail = False
        except json.JSONDecodeError as e_cut_delim_fix:
            parse_errors['fixed'] = str(e_cut_delim_fix)
        # if not fixed until here, give up
        if parse_fail:
            if verbose:
                print('Error parsing LLM output JSON:')
                print(f'JSON errors:')
                for k, v in parse_errors.items():
                    print(f'  {k}: {v}')
                print(f'LLM output:\n{llm_output_json}')
            status_dict['parse_fail'] = True
            status_dict['parsing_error_dict'] = parse_errors
            return None, status_dict

    # if we reach this point, parsing was successful
    return llm_output, status_dict


def yaml2json(llm_output_dict, verbose=False):
    """ Try to parse LLM output YAML and convert it to JSON.

        Returns a tuple of the form (llm_output_JSON, status_dict)
    """

    status_dict = {
        'parse_fail': False,
        'parsing_error_dict': {},
    }

    # predicted annotations in YAML
    llm_output_yaml = llm_output_dict['completion']['choices'][0]['text']

    # try to parse
    yaml_errors = {}
    llm_output = None
    try:
        llm_output = yaml.load(llm_output_yaml, Loader=yaml.Loader)
    except yaml.YAMLError as e_general:
        yaml_errors['general'] = str(e_general)
        parse_fail = True
        # try to fix YAML
        # 1. assume output is cut off, remove last line
        last_line_cut = '\n'.join(llm_output_yaml.split('\n')[:-1])
        try:
            llm_output = yaml.load(last_line_cut, Loader=yaml.Loader)
            parse_fail = False
        except yaml.YAMLError as e_last_line_cut:
            yaml_errors['last_line_cut'] = str(e_last_line_cut)
            pass  # try next fix
        # 2. assume special characters in dict values. add quotes
        #    to dict values that don't have quotes
        try:
            new_yaml_lines = []
            for line in llm_output_yaml.split('\n'):
                if line.endswith(': null'):
                    # don't add quotes to null values, keep line as is
                    new_yaml_lines.append(line)
                    continue
                patt = r'^(\s+[a-z]+: )([^\"\n\r]+)$'
                if re.match(patt, line):
                    line = re.sub(patt, r'\1"\2"', line)
                new_yaml_lines.append(line)
            new_yaml = '\n'.join(new_yaml_lines)
            llm_output = yaml.load(new_yaml, Loader=yaml.Loader)
            parse_fail = False
        except yaml.YAMLError as e_quotes:
            yaml_errors['quotes'] = str(e_quotes)
            # 2.1. assume backslashes causing “unknown escape character” error
            try:
                esc_yaml_lines = []
                for line in new_yaml.split('\n'):
                    m = re.match(r'^(\s+[a-z]+: )"(.+)"$', line)
                    if m:
                        key_part = m.group(1)
                        val_part = m.group(2)
                        # escape backsplashes in value part
                        val_part = re.sub(r'\\', r'\\\\', val_part)
                        line = f'{key_part}"{val_part}"'
                    esc_yaml_lines.append(line)
                esc_yaml = '\n'.join(esc_yaml_lines)
                llm_output = yaml.load(esc_yaml, Loader=yaml.Loader)
                parse_fail = False
            except yaml.YAMLError as e_quotes:
                yaml_errors['quotes+esc'] = str(e_quotes)
                pass  # handle further down
            # 2.2. assume text after closing quotes
            try:
                esc_yaml_lines = []
                for line in new_yaml.split('\n'):
                    m = re.match(r'^(\s+[a-z]+: )"(.+)"(.+)$', line)
                    if m:
                        key_part = m.group(1)
                        val_part = m.group(2)
                        trailing_part = m.group(3)
                        # move trailing part inside quotes
                        line = f'{key_part}"{val_part} {trailing_part}"'
                    esc_yaml_lines.append(line)
                esc_yaml = '\n'.join(esc_yaml_lines)
                llm_output = yaml.load(esc_yaml, Loader=yaml.Loader)
                parse_fail = False
            except yaml.YAMLError as e_quotes:
                yaml_errors['quotes+trailing'] = str(e_quotes)
                pass  # handle further down

        # if parsing still fails, print error and return None
        if parse_fail:
            print(llm_output_dict['completion']['choices'][0]['text'])
            input()
            # max tokens reached, fixable w/ adding brackets: 2
            # unescaped backslash from LaTeX in string: 3
            # JSON format error `{'k': 'v', {'k': 'v'}, ...}`:  1
            if verbose:
                print('Error parsing LLM output YAML:')
                print(f'YAML errors:')
                for k, v in yaml_errors.items():
                    print(f'  {k}: {v}')
                print(f'LLM output:\n{llm_output_yaml}')
            status_dict['parse_fail'] = True
            status_dict['parsing_error_dict'] = yaml_errors
            return None, status_dict

    # if we reach this point, parsing was successful
    return llm_output, status_dict


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python llm_eval.py <llm_output>')
        sys.exit(1)

    fn_in = sys.argv[1]

    with open(sys.argv[1], 'r') as f:
        llm_output_dict = json.load(f)

    llm_out_conv = llm_output2eval_input(llm_output_dict, verbose=True)

    fn_out = fn_in.replace('.json', '_conv.json')

    print(f'Writing converted output to {fn_out}')
    with open(fn_out, 'w') as f:
        json.dump(llm_out_conv, f, indent=2)
