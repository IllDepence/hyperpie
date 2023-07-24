""" Generate plots for format eval results using matplotlib. """

import json
import matplotlib.pyplot as plt

eval_result_fns = {
    'GPT3': 'format_eval_text_davinci_003.json',
    'Vicuna': 'format_eval_lmsys_vicuna_13b_v1_3.json',
    'WizardLM': 'format_eval_WizardLM_WizardLM_13B_V1_1.json',
    'GALACTICA': 'format_eval_facebook_galactica_120b.json',
    'Falcon': 'format_eval_tiiuae_falcon_40b_instruct.json',
}

# # Example format eval results
# {
#   "num_total": 444,
#   "preprocessor": {
#     "no_yaml_found": 0,
#     "empty_yaml": 0,
#     "garbage_around_yaml": 42
#   },
#   "yaml2json": {
#     "parse_fail": 62
#   },
#   "coarse_structure": {
#     "coarse_structure_error": 0
#   },
#   "json_content": {
#     "num_ents_intext_notintext": [
#       2089,
#       1070
#     ],
#     "num_ent_types_valid_invalid": [
#       2788,
#       371
#     ],
#     "num_aids_valid_invalid": [
#       2951,
#       208
#     ],
#     "num_pids_valid_invalid": [
#       0,
#       929
#     ],
#     "num_vids_valid_invalid": [
#       0,
#       879
#     ],
#     "num_cids_valid_invalid": [
#       0,
#       818
#     ]
#   }
# }


def plot_format_eval(eval_results, save_path):
    """ Create and save three types of bar plots for format eval results.

        All subplots have the same y-axis, which is the model name.

        1. Number of errors (1 by 4)
        2. Number of valid / invalid (enitiy) (1 by 2)
        3. Number of valid / invalid (enitiy) (1 by 4)
    """

    # Prepare color map
    cmap = plt.get_cmap('tab10')

    # - - - 1. Number of errors (1 by 4) - - -

    error_eval_types = {
        'YAML parse error':
            lambda x: x['yaml2json']['parse_fail'],
        'Empty YAML':
            lambda x: x['preprocessor']['empty_yaml'],
        'Garbage around YAML':
            lambda x: x['preprocessor']['garbage_around_yaml'],
        'Coarse Structure error':
            lambda x: x['coarse_structure']['coarse_structure_error'],
    }

    # Create figure and axes with shared y-axis
    fig, axs = plt.subplots(
        1, len(error_eval_types),
        figsize=(10, 1.5),
        sharey=True,
        sharex=True
    )

    # Plot error counts for each model
    for i, eval_type in enumerate(error_eval_types):
        eval_type_name, accsses_func = eval_type, error_eval_types[eval_type]
        axs[i].set_title(eval_type_name)

        # Set y limit
        axs[i].set_xlim(0, 300)

        # Horizontal lot bars for each model
        axs[i].barh(
            list(eval_results.keys()),
            [accsses_func(eval_results[model]) for model in eval_results],
            align='center',
            color=[cmap(i) for i in range(len(eval_results))]
        )

    # Adjust layout
    fig.tight_layout()

    # Save figure
    plt.savefig(save_path + 'format_eval_errors.png')

    # - - - 2. Number of valid / invalid (enitiy) (1 by 2) - - -

    val_ent_eval_types = {
        'Entity in text':
            lambda x: x['json_content']['num_ents_intext_notintext'],
        'Entity type in scope':
            lambda x: x['json_content']['num_ent_types_valid_invalid'],
    }

    # validity_eval_types_e = [
    #     'Entity in text',
    #     'Entity type'
    # ]
    # validity_eval_types_id = [
    #     'Artifact ID',
    #     'Parameter ID',
    #     'Value ID',
    #     'Context ID'
    # ]


if __name__ == '__main__':
    # Load format eval results
    eval_results = {}
    for model in eval_result_fns:
        with open(eval_result_fns[model], 'r') as f:
            eval_results[model] = json.load(f)

    # Plot and save results
    plot_format_eval(eval_results, './')
