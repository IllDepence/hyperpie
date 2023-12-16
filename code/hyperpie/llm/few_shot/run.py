# Run external experiments
#
#
# Usage:
#    $ python run.py \
#        --model_path \
#        --output_path \
#        --log_path \
#        --data_path \
#        --vectorstore_path \
#        --format "yaml" \
#        --n_examples 5 \
#        --n_gpu_layers 40 \
#        --n_batch 512 \
#        --n_ctx \
#        --f16_kv \
#        --verbose \
#        --logits_all \
#        --temperature 0.0 \
#        --top_p 1.0 \
#        --repeat_penalty 1.0 \
#        --max_tokens 2048 \
#        --grammar \
#


from pathlib import Path
from tqdm.auto import tqdm
import joblib

from llama_cpp import LlamaGrammar

from .llama_patch import LlamaPatched
from .prompt_builder import PromptBuilder, JSON_GRAMMAR, YAML_GRAMMAR
from .utils import load_data, construct_docs, search_neighbours


def main(args):
    # check paths
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    data_path = Path(args.data_path)
    if args.n_examples > 0:
        vectorstore_path = Path(args.vectorstore_path)

    # prepare data
    data = load_data(data_path)
    docs = construct_docs(data)
    if args.n_examples > 0:
        docs = search_neighbours(docs, vectorstore_path, top_k=args.n_examples)

    # prepare model
    llm = LlamaPatched(
        model_path=model_path.as_posix(),
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_ctx=args.n_ctx,
        f16_kv=args.f16_kv,
        verbose=args.verbose,
        logits_all=args.logits_all,
    )

    # prepare prompt template
    prompt_builder = PromptBuilder(format=args.format)

    # prepare grammar
    grammar = LlamaGrammar.from_string(
        JSON_GRAMMAR if args.format == "json" else YAML_GRAMMAR
    )

    # loop for all instances in docs
    outputs = []
    for doc in tqdm(docs, disable=not args.verbose):
        if args.n_examples > 0:
            # retrive similar examples
            examples = [
                docs[i] for i in doc.metadata['neighbours'][:args.n_examples]
            ]
            assert len(examples) == args.n_examples
        else:
            examples = None

        # build prompt for the current doc
        prompt = prompt_builder.format(
            text=doc.page_content,
            data=data,
            examples=examples,
        )

        # run completion
        output = llm(
            prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            max_tokens=args.max_tokens,
            grammar=grammar,
        )
        output['prompt'] = prompt
        outputs.append(output)

        # save output
        completion = output['choices'][0]['text'].strip().strip('`')
        name = f"{doc.metadata['source']}-{doc.metadata['paragraph_index']}.txt"
        (output_path / name).save_text(completion)

    # save log
    joblib.dump({log_path.stem: outputs}, log_path)


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser("Inference via llama.cpp")
    ap.add_argument("model_path", type=str, required=True, help="path to the model file")
    ap.add_argument("data_path", type=str, required=True, help="path to the data json")
    ap.add_argument("vectorstore_path", type=str, required=True, help="path to the faiss index dir")
    ap.add_argument("output_path", type=str, required=True, help="path to the output dir")
    ap.add_argument("log_path", type=str, required=True, help="path to the log dir")
    ap.add_argument("format", type=str, required=True, choice=["yaml", "json"])
    ap.add_argument("n_examples", type=int, default=5, help="number of examples in prompt")
    ap.add_argument("n_gpu_layers", type=int, default=80, help="number of gpu layers to offload")
    ap.add_argument("n_batch", type=int, default=2048, help="batch size")
    ap.add_argument("n_ctx", type=int, default=4096, help="context size")
    ap.add_argument("f16_kv", action='store_true', help="half precision")
    ap.add_argument("logits_all", action='store_true', help="whether to output logits")
    ap.add_argument("temperature", type=float, default=0.0, help="temperature parameter")
    ap.add_argument("top_p", type=float, default=1.0, help="top_p parameter")
    ap.add_argument("repeat_penalty", type=float, default=1.1, help="repeatition penalty")
    ap.add_argument("max_tokens", type=int, default=2048, help="max number of tokens to generate")
    ap.add_argument("grammar", action='store_true', help="whether to use grammar constraints")
    ap.add_argument("verbose", action='store_true', help="verbose flag")
    args = parser.parse_args()

    main(args)

