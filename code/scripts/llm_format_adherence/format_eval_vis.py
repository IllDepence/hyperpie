""" Generate plots for format eval results using matplotlib. """

import json
import os
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
    'Vicuna 16k (Y)': 'data/format_eval_lmsys_vicuna_13b_v1_5_16k.json',
    'Vicuna 16k (J)': 'data/format_eval_lmsys_vicuna_13b_v1_5_16k_json.json',
    'GPT-3.5 (Y)': 'data/format_eval_text_davinci_003.json',
    'GPT-3.5 (J)': 'data/format_eval_text_davinci_003_json.json',
    'GALACTICA (Y)': 'data/format_eval_facebook_galactica_120b.json',
    'GALACTICA (J)': 'data/format_eval_facebook_galactica_120b_json.json',
    'Falcon (Y)': 'data/format_eval_tiiuae_falcon_40b_instruct.json',
    'Falcon (J)': 'data/format_eval_tiiuae_falcon_40b_instruct_json.json',
    'Vicuna 4k (Y)': 'data/format_eval_lmsys_vicuna_13b_v1_3.json',
    'Vicuna 4k (J)': 'data/format_eval_lmsys_vicuna_13b_v1_3_json.json',
    'WizardLM (Y)': 'data/format_eval_WizardLM_WizardLM_13B_V1_1.json',
    'WizardLM (J)': 'data/format_eval_WizardLM_WizardLM_13B_V1_1_json.json',
}


def bar_lbl_fmt(val):
    if val == 100:
        return ''  # we don’t want the label to overflow the blot
    dec_val = f' {val:,.1f}'
    clean_val = dec_val.rstrip('0').rstrip('.')
    perc_val = clean_val + '%'
    return perc_val


def plot_format_eval_content(eval_results, save_path):
    """ Create horizontal bar plots for format eval results.

        All subplots have the same y-axis, which is the model name.

        Overall a (1 by 4) grid of subplots with rows as follows

        1. Number of valid / invalid (enitiy) (1 by 2, two empty)
        ~~2. Number of valid / invalid (enitiy) (1 by 4)~~
    """

    # Prepare color map
    cmap = plt.get_cmap('tab20')

    # Create figure and axes with shared y-axis
    fig, axs = plt.subplots(
        1, 2,
        figsize=(6, 2.6),
        sharey=True,
        sharex=True,
        layout='constrained'
    )

    # - - - 1. Number of valid / invalid (enitiy) (1 by 2) - - -

    val_ent_eval_types = {
        'Entity not in text':
            lambda x: x['json_content']['num_ents_intext_notintext'],
        'Entity type out of scope':
            lambda x: x['json_content']['num_ent_types_valid_invalid'],
    }

    # Plot relative distribution of valid/invalid entities for each model
    for i, eval_type in enumerate(val_ent_eval_types):
        eval_type_name, accsses_func = eval_type, val_ent_eval_types[eval_type]
        axs[i].set_title(eval_type_name)

        # Set y limit
        axs[i].set_xlim(0, 100)

        # Calculate values
        barvals = [
            100*(
                accsses_func(eval_results[model])[1] / (
                    accsses_func(eval_results[model])[0] +
                    accsses_func(eval_results[model])[1]
                )
            )
            for model in eval_results
        ]
        barcolors = [cmap(i) for i in range(len(eval_results))]

        # Horizontal lot bars for each model where the x-axis shows the
        # percentage of invalid entities
        bars = axs[i].barh(
            list(eval_results.keys()),
            barvals,
            align='center',
            color=barcolors
        )
        axs[i].bar_label(bars, fmt=bar_lbl_fmt, color='grey')

    # # - - - 2. Number of valid / invalid (enitiy) (1 by 4) - - -

    # val_ent_eval_ids = {
    #     'Invalid artifact ID':
    #         lambda x: x['json_content']['num_aids_valid_invalid'],
    #     'Invalid parameter ID':
    #         lambda x: x['json_content']['num_pids_valid_invalid'],
    #     'Invalid value ID':
    #         lambda x: x['json_content']['num_vids_valid_invalid'],
    #     'Invalid context ID':
    #         lambda x: x['json_content']['num_cids_valid_invalid'],
    # }

    # # Plot relative distribution of valid/invalid IDs for each model
    # for i, eval_type in enumerate(val_ent_eval_ids):
    #     eval_type_name, accsses_func = eval_type, val_ent_eval_ids[eval_type]
    #     axs[1, i].set_title(eval_type_name)

    #     # Set y limit
    #     axs[1, i].set_xlim(0, 100)

    #     # Horizontal lot bars for each model where the x-axis shows the
    #     # percentage of invalid entities
    #     bars = axs[1, i].barh(
    #         list(eval_results.keys()),
    #         [100*(accsses_func(eval_results[model])[1] /
    #          (accsses_func(eval_results[model])[0] +
    #           accsses_func(eval_results[model])[1]))
    #          for model in eval_results],
    #         align='center',
    #         color=[cmap(i) for i in range(len(eval_results))]
    #     )
    #     axs[1, i].bar_label(bars, fmt=bar_lbl_fmt, color='grey')

    # Set x-axis label without using fig.text
    fig.supxlabel(
        'Percentage of predicted entities'
    )

    # Save figure
    # plt.savefig(save_path + 'format_eval.pgf')
    plt.savefig(os.path.join(save_path, 'format_eval.pdf'))


def plot_format_eval_parsing(eval_results, save_path):
    """ Create horizontal bar plots for format eval results.

        All subplots have the same y-axis, which is the model name.

        Overall a (1 by 4) grid of subplots with rows as follows

        1. Number of errors (1 by 4)
    """

    # Prepare color map
    cmap = plt.get_cmap('tab20')

    # - - - 1. Number of errors (1 by 4) - - -

    error_eval_types = {
        'J/Y parse error':
            lambda x: x['parse_yaml_json']['parse_fail'] if 'parse_yaml_json' in x else x['yaml2json']['parse_fail'],  # noqa (support confusing legacy dict key)
        'Empty J/Y':
            lambda x: x['preprocessor']['empty_yaml'],
        'Text around J/Y':
            lambda x: x['preprocessor']['garbage_around_yaml'],
        'Data structure error':
            lambda x: x['coarse_structure']['coarse_structure_error'],
    }

    # Create figure and axes with shared y-axis
    fig, axs = plt.subplots(
        1, 4,
        figsize=(10, 2.6),
        sharey=True,
        sharex=True,
        layout='constrained'
    )

    # Plot relative error counts for each model
    for i, eval_type in enumerate(error_eval_types):
        eval_type_name, accsses_func = eval_type, error_eval_types[eval_type]
        axs[i].set_title(eval_type_name)

        # Set y limit
        axs[i].set_xlim(0, 100)

        # Calculate values
        barvals = []
        for model in eval_results:
            num_samples = 444
            bv = 100 * accsses_func(eval_results[model]) / num_samples
            barvals.append(bv)
        barcolors = [cmap(i) for i in range(len(eval_results))]

        # Horizontal lot bars for each model
        bars = axs[i].barh(
            list(eval_results.keys()),
            barvals,
            align='center',
            color=barcolors
        )
        axs[i].bar_label(bars, fmt=bar_lbl_fmt, color='grey')

    # Set x-axis label without using fig.text
    fig.supxlabel(
        'Percentage of samples'
    )

    # Save figure
    # plt.savefig(save_path + 'format_eval.pgf')
    plt.savefig(os.path.join(save_path, 'format_eval_json_yaml.pdf'))


def plot_format_eval_mix(eval_results, save_path):
    """ Create horizontal bar plots for format eval results.

        All subplots have the same y-axis, which is the model name.
    """

    # Prepare color map
    cmap = plt.get_cmap('tab20')
    print(cmap)

    # - - - 1. Number of errors (1 by 4) - - -

    error_eval_types = {
        '(a) J/Y parse error':
            lambda x: x['parse_yaml_json']['parse_fail'] if 'parse_yaml_json' in x else x['yaml2json']['parse_fail'],  # noqa (support confusing legacy dict key)
        '(b) Text around J/Y':
            lambda x: x['preprocessor']['garbage_around_yaml'],
    }
    val_ent_eval_types = {
        '(c) Entity not in text':
            lambda x: x['json_content']['num_ents_intext_notintext'],
        '(d) Ent. type out of scope':
            lambda x: x['json_content']['num_ent_types_valid_invalid'],
    }

    # Create figure and axes with shared y-axis
    fig, axs = plt.subplots(
        1, 4,
        figsize=(10, 2.6),
        sharey=True,
        sharex=True,
        layout='constrained'
    )

    # Plot relative error counts for each model
    for i, eval_type in enumerate(error_eval_types):
        eval_type_name, accsses_func = eval_type, error_eval_types[eval_type]
        axs[i].set_title(eval_type_name)

        # Set y limit
        axs[i].set_xlim(0, 100)

        # Calculate values
        barvals = []
        for model in eval_results:
            num_samples = 444
            bv = 100 * accsses_func(eval_results[model]) / num_samples
            barvals.append(bv)
        barcolors = [cmap(i) for i in range(len(eval_results))]

        # Horizontal lot bars for each model
        bars = axs[i].barh(
            list(eval_results.keys()),
            barvals,
            align='center',
            color=barcolors
        )
        axs[i].bar_label(bars, fmt=bar_lbl_fmt, color='grey')

    # Plot relative distribution of valid/invalid entities for each model
    for i, eval_type in enumerate(val_ent_eval_types):
        j = i + 2
        eval_type_name, accsses_func = eval_type, val_ent_eval_types[eval_type]
        axs[j].set_title(eval_type_name)

        # Set y limit
        axs[j].set_xlim(0, 100)

        # Calculate values
        barvals = []
        for i, model in enumerate(eval_results):
            barval = 100*(
                accsses_func(eval_results[model])[1] / (
                    accsses_func(eval_results[model])[0] +
                    accsses_func(eval_results[model])[1]
                )
            )
            # For eval type "Entity type out of scope", don’t plot values
            # for the Vicuna 16k model (set value to 100)
            if (
                eval_type_name == '(d) Ent. type out of scope' and
                i in [0, 1]  # Vicuna 16k
            ):
                barval = 100
            barvals.append(barval)

        barcolors = []
        for i in range(len(eval_results)):
            barcolor = cmap(i)
            # For eval type "Entity type out of scope", don’t plot values
            # for the Vicuna 16k model (set color to white)
            if (
                eval_type_name == '(d) Ent. type out of scope' and
                i in [0, 1]  # Vicuna 16k
            ):
                barcolor = 'white'
            barcolors.append(barcolor)

        # For eval type "Entity type out of scope", don’t plot values
        # for the Vicuna 16k model
            'vicuna_16k' in eval_results

        # Horizontal lot bars for each model where the x-axis shows the
        # percentage of invalid entities
        bars = axs[j].barh(
            list(eval_results.keys()),
            barvals,
            align='center',
            color=barcolors
        )
        axs[j].bar_label(bars, fmt=bar_lbl_fmt, color='grey')

    # Add empty x-axis label to force padding
    fig.supxlabel(
        ' '
    )
    # Add joint x-axis labels
    fig.text(
        0.33, 0.035,
        'Percentage of samples',
        ha='center'
    )
    fig.text(
        0.77, 0.035,
        'Percentage of predicted entities',
        ha='center'
    )
    # For Vicuna 16k model in eval type "Entity type out of scope"
    # put a “n/a” label in the middle of the x-axis
    fig.text(
        0.886, 0.233,
        'n/a',
        ha='center',
        color='grey'
    )
    fig.text(
        0.886, 0.286,
        'n/a',
        ha='center',
        color='grey'
    )

    # Save figure
    # plt.savefig(save_path + 'format_eval.pgf')
    plt.savefig(os.path.join(save_path, 'format_eval_mix.pdf'))


if __name__ == '__main__':
    # Load format eval results
    eval_results = {}
    for model in eval_result_fns:
        with open(eval_result_fns[model], 'r') as f:
            eval_results[model] = json.load(f)
    # Plot and save results
    # plot_format_eval_parsing(eval_results, 'figures')
    # plot_format_eval_content(eval_results, 'figures')
    plot_format_eval_mix(eval_results, 'figures')
