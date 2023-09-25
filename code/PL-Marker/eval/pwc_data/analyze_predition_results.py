import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def _get_para_ntpus(pred_para):
    """ Get the number of value to parameter, parameter to artifact,
        and value to parameter to artifact tu/triples for the given
        paragraph.
    """

    vp_rels = dict()
    pa_rels = dict()

    # iterate over sentences
    para_delta = 0
    for sent_idx, sent in enumerate(pred_para['sentences']):
        ner_pred = pred_para['predicted_ner'][sent_idx]
        ner_gold = pred_para['ner'][sent_idx]
        offs_to_label = {}
        # predicted NER
        for (start, end, label) in ner_pred:
            offs_to_label[(start, end)] = label
        # "gold" (dist sup) NER
        for (start, end, label) in ner_gold:
            offs_to_label[(start, end)] = label
        if 'predicted_relations' in pred_para:
            re_pred = pred_para['predicted_relations'][sent_idx]
            for (
                start_from, end_from, start_to, end_to, label
            ) in re_pred:
                label_from = offs_to_label.get((start_from, end_from))
                label_to = offs_to_label.get((start_to, end_to))
                if label_from == 'v' and label_to == 'p':
                    vp_rels[(start_from, end_from)] = (start_to, end_to)
                elif label_from == 'p' and label_to == 'a':
                    pa_rels[(start_from, end_from)] = (start_to, end_to)
        para_delta += len(sent)

    vpa_trips = []
    for from_v, to_p in vp_rels.items():
        for from_p, to_a in pa_rels.items():
            if to_p == from_p:
                vpa_trips.append((from_v, to_p, to_a))

    num_vp_rels = len(vp_rels)
    num_pa_rels = len(pa_rels)
    num_vpa_trips = len(vpa_trips)

    return num_vp_rels, num_pa_rels, num_vpa_trips


def _get_number_of_ent_type(etyp, pred_para):
    n = 0
    for ner_sent in pred_para['predicted_ner']:
        for pred in ner_sent:
            if pred[2] == etyp:
                n += 1
    return n


def analyse(ppr_preds, arxiv_md, ppr_pwc_gh_md):
    print(f'ppr_preds: {len(ppr_preds)}')
    print(f'arxiv_md: {len(arxiv_md)}')
    print(f'ppr_pwc_gh_md: {len(ppr_pwc_gh_md)}')

    hyperparam_info_pos = []
    hyperparam_info_num = []
    hp_hists = defaultdict(list)

    for aid in ppr_preds.keys():
        ppr_pred = ppr_preds[aid]
        ppr_axmd = arxiv_md.get(aid)
        ppr_pgmd = ppr_pwc_gh_md.get(aid)
        num_paras = len(ppr_pred)
        if ppr_axmd is None:
            continue
        category = ppr_axmd.get('categories').split(' ')[0]
        if category is None:
            continue
        for para_idx, para in enumerate(ppr_pred):
            # TODO:
            # write method to get number of
            # - artifacts *not* in a relation
            # - a<-p relations *without* a relation to a value
            # then plot below histogram w/ a, a<-p, a<-p<-v as categories
            vp, pa, vpa = _get_para_ntpus(para)
            hp_pos = para_idx / num_paras
            hp_num = vpa
            hyperparam_info_pos.append(hp_pos)
            hyperparam_info_num.append(hp_num)
            for i in range(hp_num):
                # as often as the number of triples
                hp_hists[category].append(hp_pos)

    print(hp_hists.keys())

    # plot histogram of hyperparam_info_num (y-axis) vs
    # hyperparam_info_pos (x-axis)

    num_bins = 20
    joint_bins = np.linspace(0, 1, num_bins+1)
    cats = ['cs.LG', 'cs.CV', 'cs.CL']  # , 'cs.DL'

    # determine bin borders for joint plotting of categories
    cat_bins = set()
    for i in range(num_bins):
        joint_bin_start = joint_bins[i]  # e.g. 0.0 for 20 bins at i=0
        joint_bin_end = joint_bins[i+1]  # e.g. 0.05 for 20 bins at i=0
        # split into #cats sub bins
        sub_bins = np.linspace(
            joint_bin_start, joint_bin_end, len(cats)+1
        )
        cat_bins.update(sub_bins)
    cat_bins = sorted(list(cat_bins))  # sort

    # plot for each category
    for i, cat in enumerate(cats):
        # get numpy histogram
        counts, bins = np.histogram(
            hp_hists[cat], bins=num_bins, range=(0, 1), density=True
        )
        # determine values of sub bins (need to set counts to 0 for
        # sub bins that belong to other categories)
        cat_counts = []
        count_idx = 0
        for j in range(num_bins):
            # for each joint bin:
            # we add the count if the category matches or zero otherwise
            for k in range(len(cats)):
                # for each sub bin:
                # check if the sub bin belongs to the current category
                # and assign the count accordingly
                if i == k:
                    # our sub bin
                    cat_counts.append(counts[count_idx])
                    count_idx += 1
                else:
                    # not our sub bin
                    cat_counts.append(0)
        # adjust count values to percentages
        cat_counts = [(c/num_bins)*100 for c in cat_counts]
        plt.stairs(
            cat_counts, cat_bins,
            fill=True, orientation='horizontal', label=cat
        )
        # plt.fill_betweenx(
        #     bins[1:], counts, step='pre', alpha=0.5, label=cat
        # )
    plt.gca().invert_yaxis()
    # put a "start" and "end" tick rotated by 90 degrees at the
    # top and bottom of the y-axis instead of the default ticks
    plt.yticks([0, 1], ['start', 'end'], rotation=90)
    plt.ylim(1, 0)
    # set axis labels
    plt.xlabel('Probability of hyper-\nparameter information [%]')
    plt.ylabel('Position in the paper')
    plt.legend()
    # set figure size
    plt.gcf().set_size_inches(2, 3.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_fp', type=str)
    parser.add_argument('arxiv_metadata_fp', type=str)
    parser.add_argument('pwc_gh_metadata_fp', type=str)
    args = parser.parse_args()

    # load prediction data
    ppr_preds = defaultdict(list)
    with open(args.pred_fp, 'r') as f:
        for line in f:
            para = json.loads(line)
            doc_key = para['doc_key']
            arxiv_id, para_uuid = doc_key.split('-', maxsplit=1)
            ppr_preds[arxiv_id].append(para)

    # load arXiv metadata
    arxiv_md = {}
    with open(args.arxiv_metadata_fp) as f:
        for line in f:
            md = json.loads(line)
            aid = md.get('id')
            arxiv_md[aid] = md

    # # load PwC and GitHub metadata
    pwc_gh_md = {}
    # with open(args.pwc_gh_metadata_fp, 'r') as f:
    #     pwc_gh_md = json.load(f)

    analyse(ppr_preds, arxiv_md, pwc_gh_md)
