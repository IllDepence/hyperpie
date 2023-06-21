""" Deal with LLM output and make it compatible with the evaluation script.

    Use

        convert(input_paragraph, llm_output_yaml)

    to convert LLM output to the format expected by the evaluation function.
"""

import re
import sys
import json
import uuid
import yaml
from collections import OrderedDict
from difflib import SequenceMatcher


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


def get_coarse_structure_entries(llm_output):
    """ Get the coarse structure entries from LLM output.

        Expected entries:
        - text_contains_entities
        - entities (short or long dict key version)

        Optional entries:
        - annotated_text
    """

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
        print(
            f'Expected list/dict as top-level YAML element, '
            f'got {type(llm_output)}'
        )
        print(f'LLM output: {llm_output}')
        sys.exit(1)

    # if it’s a list, convert it to a dict
    if type(llm_output) == list:
        # convernsion works if every element is a dict with a single key
        llm_output_fixed = {}
        for elem in llm_output:
            if type(elem) != dict:
                print(
                    f'Expected list of dicts as top-level YAML element, '
                    f'got {llm_output}'
                )
                sys.exit(1)
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
        print(
            f'Expected dict with keys {has_entities_key} and '
            f'{entities_keys}, got {llm_output.keys()}'
        )
        print(f'LLM output: {llm_output}')
        sys.exit(1)
    else:
        annotation_info[
            'text_contains_entities'
        ] = llm_output[has_entities_key]
        for key in entities_keys:
            if key in llm_output.keys():
                annotation_info['entities'] = llm_output[key]
                break

    return annotation_info


def llm_output2eval_input(
        llm_output_dict,
        llm_annotated_text=None,
        matched_surface_forms=None,
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

    Returns:
        dict: Evaluation script input in JSON format.
    """

    # determine if surface forms need to be matched in the text
    if matched_surface_forms is None:
        if llm_annotated_text is None:
            matched_surface_forms = True
        else:
            matched_surface_forms = False

    # convert YAML to JSON
    llm_output = yaml2json(llm_output_dict, verbose=verbose)
    if llm_output is None:
        # YAML parsing failed
        return None

    # input paragraph (used to determine text offsets)
    para = llm_output_dict['paragraph']

    eval_input = empty_para_annotation(
        para['annotator_id'],
        para['document_id'],
        para['paragraph_index'],
        para['text']
    )

    # get coarse structure entries
    annotation_info = get_coarse_structure_entries(llm_output)

    if annotation_info['text_contains_entities'] is False:
        # If there are no entities, return the annotation dict empty
        return eval_input

    # check types
    if not (
        type(annotation_info['text_contains_entities']) == bool and
        type(annotation_info['entities']) == list
    ):
        print(
            f'Expected bool and list for coarse structure entries, '
            f'got {type(annotation_info["text_contains_entities"])} and '
            f'{type(annotation_info["entities"])}'
        )
        print(f'LLM output: {llm_output}')
        sys.exit(1)

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
        return singleprompt_llm_entities2eval_input(
            para,
            annotation_info,
            eval_input,
            verbose
        )
    else:
        if not matched_surface_forms:
            # “two stage” prompt and surface forms are given as
            # annotated text with IDs
            return twostage_llm_entities2eval_input(
                para,
                annotation_info,
                llm_annotated_text,
                eval_input,
                verbose
            )
        else:
            # “two stage” prompt but surface forms are requested
            # to be extracted by matching entity names in the
            # paragraph text
            return onepointfivestage_llm_entities2eval_input(
                para,
                annotation_info,
                eval_input,
                verbose
            )


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


def _twostage_llm_parse_yaml(annotation_info):
    """ Parse LLM generated YAML and return entities and relations.
    """

    entities = {}
    relations = []
    for artf_dict in annotation_info['entities']:
        if artf_dict is None:
            continue
        # parse artifact
        artf = next(iter(artf_dict.values()))
        # set 'e' type entities to 'a' type (prompt uses 'e', eval 'a')
        artf['id'] = re.sub(r'[a-z]([0-9\.]+)', r'a\1', artf['id'])
        entities[artf['id']] = {
            'id': artf['id'],
            'type': 'a',
            'name': artf['name']
        }
        if not artf.get('has_parameters', False):
            continue
        for param_dict in artf['parameters']:
            # parse parameter
            param = next(iter(param_dict.values()))
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
            for val_dict in param['values']:
                # parse value and context
                val = next(iter(val_dict.values()))
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
                if val.get('context', None) is None:
                    continue
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

    return entities, relations


def twostage_llm_entities2eval_input(
    para,
    annotation_info,
    llm_annotated_text,
    eval_input,
    verbose
):
    # built dict of entities with ID, type, and name
    # list of relations (using entity IDs)
    entities, relations = _twostage_llm_parse_yaml(annotation_info)

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

    return eval_input


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
    entities, relations = _twostage_llm_parse_yaml(annotation_info)

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

    return eval_input


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


def yaml2json(llm_output_dict, verbose=False):
    """ Try to parse LLM output YAML and convert it to JSON.
    """

    # predicted annotations in YAML
    llm_output_yaml = llm_output_dict['completion']['choices'][0]['text']

    # try to parse
    yaml_errors = {}
    try:
        llm_output = yaml.load(llm_output_yaml, Loader=yaml.Loader)
    except yaml.YAMLError as e_general:
        yaml_errors['general'] = e_general
        parse_fail = True
        # try to fix YAML
        # 1. assume output is cut off, remove last line
        last_line_cut = '\n'.join(llm_output_yaml.split('\n')[:-1])
        try:
            llm_output = yaml.load(last_line_cut, Loader=yaml.Loader)
            parse_fail = False
        except yaml.YAMLError as e_last_line_cut:
            yaml_errors['last_line_cut'] = e_last_line_cut
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
            yaml_errors['quotes'] = e_quotes
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
                yaml_errors['quotes+esc'] = e_quotes
                pass  # handle further down

        # if parsing still fails, print error and return None
        if parse_fail:
            print('Error parsing LLM output YAML:')
            print(f'YAML errors:')
            for k, v in yaml_errors.items():
                print(f'  {k}: {v}')
            print(f'LLM output:\n{llm_output_yaml}')
            return None

    # if we reach this point, parsing was successful
    return llm_output


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
