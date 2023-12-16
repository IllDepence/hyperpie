import json
import hyperpie as hp

target_para_fp = (
    '/home/tarek/proj/hyperparam_paper/xiao_collab_repo/data/'
    'transformed_pprs_15k_sample.jsonl'
)
ground_truth_fp = (
    '/home/tarek/proj/hyperparam_paper/xiao_collab_repo/data/'
    'tsa_processed.json'
)

dist_annot_paras = hp.distant_supervision.annotate(
    ground_truth_fp=ground_truth_fp,
    target_para_fp=target_para_fp,
    verbose=True
)

with open('transformed_pprs_15k_sample_dist_annot.jsonl', 'w') as f:
    for para in dist_annot_paras:
        f.write(json.dumps(para) + '\n')
