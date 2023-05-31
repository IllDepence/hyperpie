import datetime
import json
import openai


def prompt_and_save(para, prompt, params=None, verbose=False):
    """ Convenience method that
        - gets completion from OpenAI API
        - saves result to a JSON file
        - echoes the prompt
        - echoes the completion
        - returns the completion
    """

    if params is None:
        hyperpie.llm.default_params

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fn = f"prompt_result_{timestamp}.json"

    completion = openai.Completion.create(prompt=prompt, **params)
    out_dict = {
        "paragraph": para,
        "timestamp": timestamp,
        "prompt": prompt,
        "params": params,
        "completion": completion
    }

    with open(out_fn, "w") as f:
        json.dump(out_dict, f, indent=4)

    if verbose:
        print('- - - Prompt - - -')
        print(prompt)
        print('- - - Completion - - -')
        print(out_dict["completion"]["choices"][0]["text"])
        print('- - - Saved to - - -')
        print(out_fn)

    return out_dict
