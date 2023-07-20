""" Minimal PoC for format adherance evaluation using GALACICA complections
"""

import json
import hyperpie as hp


# load GALACTICA data from completion cache
with open('hyperpie/llm/completion_cache.json') as f:
    cc = json.load(f)
cc_dicts = [cd for cd in cc['4207bd25b65a13f7663c1cea7a8073df'].values()]

stats_dicts = []

for cc_dict in cc_dicts:
    annot_j, stats_dict = hp.llm.convert.llm_output2eval_input(
        cc_dict,
        llm_annotated_text='foo',
        matched_surface_forms=True,
        preprocessor=hp.llm.convert.galactica_yaml_extract
    )
    stats_dicts.append(stats_dict)

aggregate_stats = hp.llm.convert.aggregate_format_stats(stats_dicts)

print(json.dumps(aggregate_stats, indent=4))
