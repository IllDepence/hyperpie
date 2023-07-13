""" Calculate inter annotator agreement.
"""

import json
from sklearn.metrics import cohen_kappa_score


def entity_exact_overlap(ed1, ed2):
    """ Given two sets of annotations, determine the number of
        excactly matching spans and the total number of unique
        spans.
    """

    span2type_dicts = []
    for ed in [ed1, ed2]:
        span2type_dicts.append(dict())
        for k, e in ed['annotation']['entity_dict'].items():
            span_key = (
                '-'.join([str(o) for o in e['offset']]) +
                ' (' +
                e['surface_text'] +
                ')'
            )
            span2type_dicts[-1][span_key] = e['type']
    combi_keys = set(
        list(span2type_dicts[0].keys()) +
        list(span2type_dicts[1].keys())
    )
    n_total = len(combi_keys)
    n_match = 0
    for k in combi_keys:
        typ1 = span2type_dicts[0].get(k)
        typ2 = span2type_dicts[1].get(k)
        # print(k)
        if typ1 == typ2:
            n_match += 1
    return n_match, n_total


def char_kappa(ed1, ed2):
    context_len = len(ed1['context'])
    char_etype_lists = []
    char_rellabel_lists = []
    for ed in [ed1, ed2]:
        char_etype_lists.append([])
        char_rellabel_lists.append([])
        for i in range(context_len):
            etype = '_'
            rellabel = '_'
            for k, e in ed['annotation']['entity_dict'].items():
                if i >= e['offset'][0] and i < e['offset'][1]:
                    etype = e['type']
                    rel_entity = None
                    for r in ed['annotation']['relation_tuples']:
                        if k == r[0]:
                            rel_entity = ed['annotation']['entity_dict'][
                                r[1]
                            ]
                    if rel_entity:
                        rellabel = '-'.join(
                            [str(o) for o in rel_entity['offset']]
                        )
            char_etype_lists[-1].append(etype)
            char_rellabel_lists[-1].append(rellabel)
    return (
        char_etype_lists,
        char_rellabel_lists
    )


def relation_exact_overlap(an1, an2):
    """ Given two sets of annotations, determine the number of exact
        matches in relations (exact span to exact span), as well as
        the total number of relations.
    """

    rel_key_lists = []
    for an in [an1, an2]:
        rel_key_lists.append([])
        for rt in an['annotation']['relation_tuples']:
            e1_key = '-'.join(
                [str(o) for o
                 in an['annotation']['entity_dict'][rt[0]]['offset']]
            )
            e2_key = '-'.join(
                [str(o) for o
                 in an['annotation']['entity_dict'][rt[1]]['offset']]
            )
            rel_key_lists[-1].append(f'{e1_key}-->{e2_key}')
    n_total = len(set(rel_key_lists[0] + rel_key_lists[1]))
    n_match = len(set(rel_key_lists[0]).intersection(set(rel_key_lists[1])))
    return n_match, n_total


def calc_iaa(an1, an2):
    n_e_match = 0
    n_e_total = 0
    n_r_match = 0
    n_r_total = 0
    char_etype_lists = [[], []]
    char_rellabel_lists = [[], []]
    for i in range(len(an1)):
        assert an1[i]['context'] == an2[i]['context']
        assert an1[i]['annotator_id'] != an2[i]['annotator_id']
        nem, net = entity_exact_overlap(an1[i], an2[i])
        n_e_match += nem
        n_e_total += net
        nrm, nrt = relation_exact_overlap(an1[i], an2[i])
        n_r_match += nrm
        n_r_total += nrt
        etypes, rellabels = char_kappa(an1[i], an2[i])
        for j in [0, 1]:
            char_etype_lists[j].extend(etypes[j])
            char_rellabel_lists[j].extend(rellabels[j])
    print((
        f'{n_e_match} out of {n_e_total} entities ({n_e_match/n_e_total:.3f})'
        f' match exactly'
    ))
    print((
        f'{n_r_match} out of {n_r_total} relations ({n_r_match/n_r_total:.3f})'
        f' match exactly'
    ))
    chr_etype_kappa = cohen_kappa_score(
        char_etype_lists[0],
        char_etype_lists[1],
    )
    print((
        f'Cohen’s kappa based on character level entity class: '
        f'{chr_etype_kappa:.3f}'
    ))
    chr_rellabel_kappa = cohen_kappa_score(
        char_rellabel_lists[0],
        char_rellabel_lists[1],
    )
    print((
        f'Cohen’s kappa based on character level entity class and relation'
        f'target span: {chr_rellabel_kappa:.3f}'
    ))


if __name__ == '__main__':
    infps = [
        './jkr_221110.json',
        './tsa_221110.json'
    ]
    ans = []
    for fp in infps:
        with open(fp) as f:
            ans.append(json.load(f))
    iaa = calc_iaa(ans[0], ans[1])
