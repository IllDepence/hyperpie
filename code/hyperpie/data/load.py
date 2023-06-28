""" Load data
"""

import json
import re
from collections import defaultdict
from hyperpie import settings


def load_annotated_raw():
    """ Load “raw” data with Web Annotation Data Model format
        annotations as output by annotation UI.
    """

    with open(settings.annot_raw_fp) as f:
        return json.load(f)


def load_annotated(only_full=False, with_parent=False):
    """ Load (preprocessed) annotated data paragraphs.

        If `filtered` is True, load data filtered to only include
        “full” annotations (a<-p<-v[<-c]).
    """

    if only_full:
        fp = settings.annot_onlyfull_fp
    elif with_parent:
        fp = settings.annot_withparent_fp
    else:
        fp = settings.annot_prep_fp

    with open(fp) as f:
        return json.load(f)


def load_filtered_unannotated():
    """ Load filtered unannotated data paragraphs.
    """

    return load_unannotated(settings.filtered_unannot_fp)


def load_unannotated(transformed_pprs_fp=False):
    """ Load unannotated data paragraphs.
    """

    paras = []
    with open(transformed_pprs_fp) as f:
        # flatten from papers with paras to list of paras
        for line in f:
            ppr = json.loads(line)
            for para_text in ppr['paragraphs']:
                para = {
                    'document_id': ppr['id'],
                    'text': para_text,
                }
                paras.append(para)
    return paras


def get_artifact_param_surface_form_pairs(paras_fp=None):
    """ Get list of pairs of artifact and parameter surface forms.

        Optionally an alternative file path to annotated data other
        than the preprocessed manually annotated data can be provided.
    """

    if paras_fp is None:
        paras_fp = settings.annot_prep_fp

    with open(paras_fp) as f:
        paras = json.load(f)

    rel_tups = []
    for para in paras:
        ents = para['annotation']['entities']
        # iterate over all relations in this paragraph’s annotations
        for rid, rel in para['annotation']['relations'].items():
            # only consider param -> artifact relations
            rel_source_type = ents[rel['source']]['type']
            rel_target_type = ents[rel['target']]['type']
            if rel_source_type != 'p' or rel_target_type != 'a':
                continue
            # iterate over all surface forms of both entities
            # and add all combinations to the pair list
            for surf_dict_param in ents[rel['source']]['surface_forms']:
                for surf_dict_artif in ents[rel['target']]['surface_forms']:
                    # skip when artifact surface form is too generic
                    if _artifact_surface_form_general(
                        surf_dict_artif['surface_form']
                    ):
                        continue
                    # add pair to the list
                    rel_tups.append((
                        surf_dict_param['surface_form'],
                        surf_dict_artif['surface_form']
                    ))

    return rel_tups


def _artifact_surface_form_general(surface_form):
    """ Heuristically determine whether a surface form is generic
        (and might therefore e.g. not be suitable for use in distant
        supervision).
    """

    # patterns for generic names not to include
    generic_patt = re.compile((
        r'([a-z-_]*\s*)?(model|method|system|approach|'
        r'framework|work|dataset|data set)'
    ))

    if generic_patt.match(surface_form) or len(surface_form) < 3:
        return True
    return False


def get_artifact_names(flat=False):
    """ Get list of artifact entities  where each is represented by a list
        of its names and its number of occurrences in the annotated data.

        If `flat` is True, return a flat list of all unique artifact names.
    """

    with open(settings.annot_prep_fp) as f:
        paras = json.load(f)

    # dict for keeping track of artifact IDs within a paper (b/c they are
    # consistent within a paper but not across papers)
    paper_artifacts_global = {}
    for para in paras:
        # get the known artifact IDs for this paper
        paper_id = para['document_id']
        if paper_id not in paper_artifacts_global:
            paper_artifacts_global[paper_id] = defaultdict(dict)
        # iterate over all annotations in this paragraph
        for eid, entity in para['annotation']['entities'].items():
            # only consider entities of type artifact ("a")
            if entity['type'] != 'a':
                continue
            # get the artifact ID
            artifact_id = entity['id']
            # iterate over all surface forms of this artifact
            for surf_dict in entity['surface_forms']:
                # get the surface form
                surf = surf_dict['surface_form']
                # skip if the name is generic
                if _artifact_surface_form_general(surf):
                    continue
                # get the known names for this artifact
                known_names = paper_artifacts_global[
                    paper_id
                ].get(artifact_id, None)
                # if there are no known names yet, add a list for them
                if known_names is None:
                    paper_artifacts_global[paper_id][artifact_id] = []
                # add the surface form to the list of known names
                paper_artifacts_global[paper_id][artifact_id].append(surf)

    # list of artifact dicts ({"names": [...], "count": int})
    artifacts = []
    # from the known artifacts per paper, get the names and counts an#
    # aggregate them
    for paper_id, known_artifacts_ppr in paper_artifacts_global.items():
        for artifact_id, known_names in known_artifacts_ppr.items():
            # get the count of this artifact
            count = len(known_names)
            # check if this artifact is already in the list
            for artifact in artifacts:
                for name in known_names:
                    if name in artifact['names']:
                        # artifact is already in the list, so we merge the
                        # names and increase the count
                        artifact['names'].extend(known_names)
                        artifact['count'] += count
                        break
            else:
                # artifact is not yet in the list, so we add it
                artifacts.append({
                    'names': known_names,
                    'count': count
                })

    # sort the artifacts by count
    artifacts.sort(key=lambda a: a['count'], reverse=True)

    flat_names = set()
    if flat:
        for artifact in artifacts:
            flat_names.update(artifact['names'])
        return list(flat_names)

    return artifacts
