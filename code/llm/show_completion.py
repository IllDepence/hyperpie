import sys
import json


def print_completion(saved_completion):
    """ Print a completion saved by playground.py:prompt_and_save()
    """

    print('- - - Timestamp - - -')
    print(saved_completion["timestamp"])
    print('- - - Prompt - - -')
    print(saved_completion["prompt"])
    print('- - - Completion - - -')
    print(saved_completion["completion"]["choices"][0]["text"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <completion.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        saved_completion = json.load(f)

    print_completion(saved_completion)
