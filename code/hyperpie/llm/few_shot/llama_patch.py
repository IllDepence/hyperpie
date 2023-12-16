
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt

import llama_cpp
from llama_cpp import (
    Generator,
    Llama,
    LlamaGrammar,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from llama_cpp.llama_types import Embedding



class ForceTokensLogitsProcessor(LogitsProcessor):
    """Force to generate tokens in the input text only.

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

    Note:
    Unlike Huggingface's ForceTokensLogitsProcessor, this ForceTokensLogitsProcessor
    takes the banned tokens list in self.force_token_map dictionary
    instead of the allowed tokens list.
    """

    def __init__(self, force_token_map: Dict[int, List[int]], vocab_size: int):
        self.force_token_map = force_token_map

    def __call__(
        self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]
    ) -> npt.NDArray[np.single]:
        generation_idx = input_ids.shape[-1] # previous step
        banned_tokens = self.force_token_map.get(generation_idx, None)
        if banned_tokens is not None:
            scores[:, banned_tokens] = -float("inf")
        return scores


class EmbeddingDataKNN(TypedDict):
    object: str
    index: int
    embedding: List[float]
    offset: int
    next_token: int


class LlamaPatched(Llama):
    def generate(
        self,
        tokens: Sequence[int],
        top_k: int = 40,
        top_p: float = 0.95,
        temp: float = 0.80,
        repeat_penalty: float = 1.1,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
            ...     print(llama.detokenize([token]))

        Args:
            tokens: The prompt tokens.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
            reset: Whether to reset the model state.

        Yields:
            The generated tokens.
        """
        assert self.ctx is not None
        if reset and len(self._input_ids) > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                if self.verbose:
                    print("Llama.generate: prefix-match hit", file=sys.stderr)
                reset = False
                tokens = tokens[longest_prefix:]
                self.n_tokens = longest_prefix

        if reset:
            self.reset()

        if grammar is not None:
            grammar.reset()

        generation_steps = 0
        while True:
            # workaround to avoid timeout: disable grammar after 256 steps
            generation_steps += 1
            if grammar is not None and generation_steps > 2**8:
                grammar = None

            if self.verbose:
                print("Llama.generate: token length=" + str(len()), file=sys.stderr)
            self.eval(tokens)
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                logits_processor=logits_processor,
                grammar=grammar,
            )
            if stopping_criteria is not None and stopping_criteria(
                self._input_ids, self._scores[-1, :]
            ):
                return
            tokens_or_none = yield token
            tokens = [token]
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)
    
    def create_embedding_knn(
        self,
        input: Union[Tuple[str, str], List[Tuple[str, str]]],
        model: Optional[str] = None
    ) -> Embedding:
        """Embed a tuple of strings.

        Args:
            input: A tuple of the utf-8 encoded strings to embed.

        Returns:
            An embedding object.
        """
        assert self.ctx is not None
        model_name: str = model if model is not None else self.model_path

        if self.params.embedding == False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        if isinstance(input, list):
            inputs = input
        else:
            inputs = [input]

        data: List[EmbeddingDataKNN] = []
        total_tokens = 0
        for index, (input_a, input_b) in tqdm(
            enumerate(inputs),
            disable=not self.versbose,
        ):
            tokens = self.tokenize(input_a.encode("utf-8"))
            next_tokens = self.tokenize(input_b.encode("utf-8"), add_bos=False)
            next_tokens.append(self._token_eos)
            for offset, next_token in enumerate(next_tokens):
                self.reset()
                self.eval(tokens)
                n_tokens = len(tokens)
                total_tokens += n_tokens
                embedding = llama_cpp.llama_get_embeddings(self.ctx)[
                    : llama_cpp.llama_n_embd(self.ctx)
                ]
    
                data.append(
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": index,
                        "offset": offset,
                        "next_token": next_token,
                        #"history": tokens,
                    }
                )
                
                tokens.append(next_token)

        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

