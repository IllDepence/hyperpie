""" Minimal PoC for format adherance evaluation using GALACICA complections
"""

import json
import hyperpie as hp
from collections import defaultdict

# load data from completion cache
with open('hyperpie/llm/completion_cache.json') as f:
    cc = json.load(f)

cc_dicts = [cd for cd in cc['4207bd25b65a13f7663c1cea7a8073df'].values()]

# aggregate stats
aggregate_data = {
    'preprocessor': {
        'no_yaml_found': 0, 'empty_yaml': 0, 'garbage_around_yaml': 0
    },
    'yaml2json': {'parse_fail': 0, 'parsing_error_dict': defaultdict(list)},
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

for cc_dict in cc_dicts:
    annot_j, data = hp.llm.convert.llm_output2eval_input(
        cc_dict,
        llm_annotated_text='foo',
        matched_surface_forms=True,
        preprocessor=hp.llm.convert.galactica_yaml_extract
    )
    # Aggregate 'preprocessor' data
    if 'preprocessor' in data:
        for stat_name, value in data['preprocessor'].items():
            if value is not None:
                aggregate_data['preprocessor'][stat_name] += value

    # Aggregate 'yaml2json' data
    if 'parse_fail' in data['yaml2json'] and data['yaml2json']['parse_fail']:
        aggregate_data['yaml2json']['parse_fail'] += 1
    if 'parsing_error_dict' in data['yaml2json']:
        for error_type, msg in data['yaml2json']['parsing_error_dict'].items():
            aggregate_data['yaml2json']['parsing_error_dict'][
                error_type
            ].append(msg)

    if 'coarse_structure' in data:
        if data['coarse_structure']['coarse_structure_error']:
            aggregate_data['coarse_structure']['coarse_structure_error'] += 1

    if 'json_content' in data:
        # Aggregate 'json_content' data
        for stat_name, values in data['json_content'].items():
            aggregate_data['json_content'][stat_name] = [
                sum(x) for x in
                zip(values, aggregate_data['json_content'][stat_name])
            ]

print(json.dumps(aggregate_data, indent=4))
