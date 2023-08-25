""" Print llm evaluation stats. """

import json
import os


def print_stats():
    """ Print stats.
    """

    decimals = 1
    data_dir = 'data'
    fns = {
        'Falcon': 'eval_tiiuae_falcon_40b_instruct.json',
        'GALACTICA_json': 'eval_facebook_galactica_120b_json-first105.json',
        'GALACTICA': 'eval_facebook_galactica_120b.json',
        'WizardLM_json': 'eval_WizardLM_WizardLM_13B_V1_1_json.json',
        'WizardLM': 'eval_WizardLM_WizardLM_13B_V1_1.json',
        'Vicuna_json': 'eval_lmsys_vicuna_13b_v1_3_json.json',
        'Vicuna': 'eval_lmsys_vicuna_13b_v1_3.json',
        'GPT3_json': 'eval_text_davinci_003_json.json',
        'GPT3': 'eval_text_davinci_003.json',
    }

    for model, fn in fns.items():
        print(f'=== {model} ===')
        with open(os.path.join(data_dir, fn), 'r') as f:
            stats = json.load(f)
        for typ in ['exact', 'partial_overlap']:
            print(f'- {typ} -')
            for measure in [
                'entity_recognition',
                'entity_recognition_classification',
                'co_reference_resolution',
                'relation_extraction'
            ]:
                p = stats[typ][measure]['p'] * 100
                r = stats[typ][measure]['r'] * 100
                f1 = stats[typ][measure]['f1'] * 100
                print(
                    f'{measure} [%]: P: {p:.{decimals}f} R: {r:.{decimals}f}'
                    f' F1: {f1:.{decimals}f}'
                )
            print()


if __name__ == '__main__':
    print_stats()
