""" Generate plots for format eval results using matplotlib. """

import json
import matplotlib.pyplot as plt
# import matplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

eval_result_fns = {
    'GPT3': 'format_eval_text_davinci_003.json',
    'Vicuna': 'format_eval_lmsys_vicuna_13b_v1_3.json',
    'WizardLM': 'format_eval_WizardLM_WizardLM_13B_V1_1.json',
    'GALACTICA': 'format_eval_facebook_galactica_120b.json',
    'Falcon': 'format_eval_tiiuae_falcon_40b_instruct.json',
}


def plot_format_eval(eval_results, save_path):
    """ Create horizontal bar plots for format eval results.

        All subplots have the same y-axis, which is the model name.

        Overall a (3 by 4) grid of subplots with rows as follows

        1. Number of errors (1 by 4)
        2. Number of valid / invalid (enitiy) (1 by 2, two empty)
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
        'Coarse structure error':
            lambda x: x['coarse_structure']['coarse_structure_error'],
    }

    # Create figure and axes with shared y-axis
    fig, axs = plt.subplots(
        3, 4,
        figsize=(10, 4.5),
        sharey=True,
        sharex=True,
        layout='constrained'
    )

    # Plot relative error counts for each model
    for i, eval_type in enumerate(error_eval_types):
        eval_type_name, accsses_func = eval_type, error_eval_types[eval_type]
        axs[0, i].set_title(eval_type_name)

        # Set y limit
        axs[0, i].set_xlim(0, 100)

        # Horizontal lot bars for each model
        axs[0, i].barh(
            list(eval_results.keys()),
            [
                100 * accsses_func(eval_results[model]) / 444
                for model in eval_results
            ],
            align='center',
            color=[cmap(i) for i in range(len(eval_results))]
        )

    # - - - 2. Number of valid / invalid (enitiy) (1 by 2) - - -

    val_ent_eval_types = {
        'Entity not in text':
            lambda x: x['json_content']['num_ents_intext_notintext'],
        'Entity type out of scope':
            lambda x: x['json_content']['num_ent_types_valid_invalid'],
    }

    # Plot relative distribution of valid/invalid entities for each model
    for i, eval_type in enumerate(val_ent_eval_types):
        eval_type_name, accsses_func = eval_type, val_ent_eval_types[eval_type]
        axs[1, i].set_title(eval_type_name)

        # Set y limit
        axs[1, i].set_xlim(0, 100)

        # Horizontal lot bars for each model where the x-axis shows the
        # percentage of invalid entities
        axs[1, i].barh(
            list(eval_results.keys()),
            [100*(accsses_func(eval_results[model])[1] /
             (accsses_func(eval_results[model])[0] +
              accsses_func(eval_results[model])[1]))
             for model in eval_results],
            align='center',
            color=[cmap(i) for i in range(len(eval_results))]
        )

    # add two empty subplots
    axs[1, 2].axis('off')
    axs[1, 3].axis('off')

    # - - - 3. Number of valid / invalid (enitiy) (1 by 4) - - -

    val_ent_eval_ids = {
        'Invalid artifact ID':
            lambda x: x['json_content']['num_aids_valid_invalid'],
        'Invalid parameter ID':
            lambda x: x['json_content']['num_pids_valid_invalid'],
        'Invalid value ID':
            lambda x: x['json_content']['num_vids_valid_invalid'],
        'Invalid context ID':
            lambda x: x['json_content']['num_cids_valid_invalid'],
    }

    # Plot relative distribution of valid/invalid IDs for each model
    for i, eval_type in enumerate(val_ent_eval_ids):
        eval_type_name, accsses_func = eval_type, val_ent_eval_ids[eval_type]
        axs[2, i].set_title(eval_type_name)

        # Set y limit
        axs[2, i].set_xlim(0, 100)

        # Horizontal lot bars for each model where the x-axis shows the
        # percentage of invalid entities
        axs[2, i].barh(
            list(eval_results.keys()),
            [100*(accsses_func(eval_results[model])[1] /
             (accsses_func(eval_results[model])[0] +
              accsses_func(eval_results[model])[1]))
             for model in eval_results],
            align='center',
            color=[cmap(i) for i in range(len(eval_results))]
        )

    # Set x-axis label without using fig.text
    fig.supxlabel(
        'Percentage of [row 1: samples | row 2,3: predicted entities]'
    )

    # Save figure
    # plt.savefig(save_path + 'format_eval.pgf')
    plt.savefig(save_path + 'format_eval.svg')


if __name__ == '__main__':
    # Load format eval results
    eval_results = {}
    for model in eval_result_fns:
        with open(eval_result_fns[model], 'r') as f:
            eval_results[model] = json.load(f)

    # Plot and save results
    plot_format_eval(eval_results, './')
