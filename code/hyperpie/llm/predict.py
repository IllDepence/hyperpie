import datetime
import json


def prompt_and_save(para, prompt, params, o_ai):
    """ Convenience method that
        - gets completion from OpenAI API
        - saves result to a JSON file
        - echoes the prompt
        - echoes the completion
        - returns the completion
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fn = f"prompt_result_{timestamp}.json"

    completion = o_ai.Completion.create(prompt=prompt, **params)
    out_dict = {
        "paragraph": para,
        "timestamp": timestamp,
        "prompt": prompt,
        "params": params,
        "completion": completion
    }

    with open(out_fn, "w") as f:
        json.dump(out_dict, f, indent=4)

    print('- - - Prompt - - -')
    print(prompt)
    print('- - - Completion - - -')
    print(out_dict["completion"]["choices"][0]["text"])
    print('- - - Saved to - - -')
    print(out_fn)

    return out_dict
