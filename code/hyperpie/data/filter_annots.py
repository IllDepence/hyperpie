""" Filter annotations to only contain those which fullfil given
    requirements.

    All other annotations are removed.
"""

import sys
import json


def require_apv(paras, verbose=False):
    """ See `require_apv_single` for details.
    """

    filtered_paras = []
    num_full_triples = 0
    for para in paras:
        filtered_para, num_full_triples_para = require_apv_single(para)
        num_full_triples += num_full_triples_para
        filtered_paras.append(filtered_para)

    return filtered_paras, num_full_triples


def require_apv_single(para, verbose=False):
    """ Filter annotations of a single paragraph to only
        contain those which are part of at least a “set”

            (artifact ← parameter ← value)

        and optionally with a context

            (artifact ← parameter ← value ← context)

        All other annotations are removed.

        Returns a tuple of the filtered paragraph and the
        number of full triples found.
    """

    return _require_rel_sets(para, cascade=False, verbose=verbose)


def require_parent(paras, verbose=False):
    """ See `require_parent_single` for details.
    """

    filtered_paras = []
    num_full_triples = 0
    for para in paras:
        filtered_para, num_full_triples_para = require_parent_single(para)
        num_full_triples += num_full_triples_para
        filtered_paras.append(filtered_para)

    return filtered_paras, num_full_triples


def require_parent_single(para, verbose=False):
    """ Filter annotations of a single paragraph to only
        contain those in which

            - parameters are related to artifacts
            - values are related to parameters
            - contexts are related to values

        All other annotations are removed.
        Artifacts are always kept.

        Returns a tuple of the filtered paragraph and the
        number of full triples found.
    """

    return _require_rel_sets(para, cascade=True, verbose=verbose)


def _require_rel_sets(para, cascade, verbose=False):
    """ Require a certain cet of relations.

        If cascade == False, require full
            (artifact ← parameter ← value)
        with context being optional.

        If cascade == True,
            - keep all artifacts
            - require artifacts for parameters
            - require parameters for values
            - require values for contexts
    """

    filtered_para = {}

    copy_keys = [
        'annotator_id',
        'document_id',
        'paragraph_index',
        'text',
    ]

    for key in copy_keys + ['annotation', 'annotation_raw']:
        if key in copy_keys:
            filtered_para[key] = para[key]
        else:
            filtered_para[key] = None

    keepers_entities = []
    keepers_relations = []

    # if cascade, keep all artifacts
    if cascade:
        ents = para['annotation']['entities']
        for ent_id, ent in ents.items():
            if ent['type'] == 'a':
                keepers_entities.append(ent_id)

    num_full_triples = 0
    ents = para['annotation']['entities']
    rels = para['annotation']['relations']
    for rel_ap_id, rel_ap in rels.items():
        # look for artif <- param
        if ents.get(rel_ap['target'], {}).get('type') != 'a':
            # need artifact, skip
            continue
        if cascade:
            # found a parameter with artifact, keep
            keepers_entities.append(rel_ap['source'])  # parameter
            keepers_relations.append(rel_ap_id)  # artifact <- parameter
        for rel_pv_id, rel_pv in rels.items():
            # look for matching param <- value
            if rel_ap['source'] != rel_pv['target']:
                # not connected, skip
                continue
            # found a set, keep all three entities and
            # both relations
            keepers_entities.extend(
                [
                    rel_ap['target'],  # artifact
                    rel_ap['source'],  # parameter
                    rel_pv['source']   # value
                ]
            )
            keepers_relations.extend(
                [
                    rel_ap_id,  # artifact <- parameter
                    rel_pv_id   # parameter <- value
                ]
            )
            num_full_triples += 1
            for rel_vc_id, rel_vc in rels.items():
                # look for optional matching value <- context
                if rel_pv['source'] != rel_vc['target']:
                    # not connected, skip
                    continue
                # additionally found a context, keep
                keepers_entities.append(rel_vc['source'])  # context
                keepers_relations.append(rel_vc_id)  # value <- context

                # sice there is at most one context per value,
                # we can break here
                break

    filtered_para['annotation'] = {}

    # copy entities
    filtered_para['annotation']['entities'] = {}
    for ent_id, ent in ents.items():
        if ent_id in keepers_entities:
            filtered_para['annotation']['entities'][ent_id] = ent

    # copy relations
    filtered_para['annotation']['relations'] = {}
    for rel_id, rel in rels.items():
        if rel_id in keepers_relations:
            filtered_para['annotation']['relations'][rel_id] = rel

    if verbose:
        # report on number of entities, relations and sets
        # before and after filtering
        print(f'[{para["document_id"]} - {para["paragraph_index"]}]')
        print('Before filtering:')
        print(f'  {len(ents)} entities')
        print(f'  {len(rels)} relations')
        print('After filtering:')
        print(f'  {len(filtered_para["annotation"]["entities"])} entities')
        print(f'  {len(filtered_para["annotation"]["relations"])} relations')
        print()

    return filtered_para, num_full_triples


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python filter_full_annots.py <input_file>')
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        paras = json.load(f)

    filtered_paras_onlyfull = []
    filtered_paras_reqparents = []
    num_full_triples = 0
    num_paras_with_annotations = 0
    num_paras_with_full_triples = 0
    for para in paras:
        filtered_para_onlyfull, n_full_triples_para = require_apv_single(para)
        num_full_triples += n_full_triples_para
        if n_full_triples_para > 0:
            num_paras_with_full_triples += 1
        if len(para['annotation']['entities']) > 0:
            num_paras_with_annotations += 1
        filtered_paras_onlyfull.append(filtered_para_onlyfull)

        filtered_para_reqparents, _ = require_parent_single(para)
        filtered_paras_reqparents.append(filtered_para_reqparents)

    print(f'Found {num_full_triples} full triples in total.')
    print(f'Found {num_paras_with_annotations} paragraphs with annotations.')
    print(f'Found {num_paras_with_full_triples} paragraphs with full triples.')

    fp_onlyfull = input_file.replace('.json', '_onlyfull.json')
    with open(fp_onlyfull, 'w') as f:
        json.dump(filtered_paras_onlyfull, f)
    fp_reqparents = input_file.replace('.json', '_withparent.json')
    with open(fp_reqparents, 'w') as f:
        json.dump(filtered_paras_reqparents, f)
