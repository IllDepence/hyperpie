""" Prompt templates
"""

instruction_context = "In the context of machine learning and related fields,"
instruction_yn_question = "Output only either \"yes\" or \"no\"."  # noqa: E501
instruction_entity = "Entity (a model/method/dataset)"
instruction_output = "Output:\n"

text_has_artifs = f"""{instruction_context} does the Input Text below mention any model/method/dataset?

[Input Text start]
{{text}}
[Input Text end]

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

artif_is_out_of_scope = f"""{instruction_context} does the {instruction_entity} below fall into any of the following categories?

- a machine learning library
- a machine learning framework
- a machine learning task
- a metric
- an API

Entity: {{entity}}

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

text_which_artifs = f"""{instruction_context} what are the entities (dataset/model/method/loss function/regularization technique) mentioned in the Input Text below?

[Input Text start]
{{text}}
[Input Text end]

Answer in the following YAML format.

Format:
---
- entity<N>:
    name: <entity name>
    type: <entity type>
...

Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

artif_has_paras = f"""{instruction_context} does the following {instruction_entity} have parameters for which a value can be chosen?

Entity: {{entity}}

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

artif_which_paras = f"""{instruction_context} what are the parameter(s) of the following {instruction_entity}, for which a value can be chosen when using it?

Entity: {{entity}}

For each parameter, name its formula symbol (if it has one), as well as typical values of the parameter. Answer in the following YAML format.

Format:
---
- parameter<N>:
    name: <parameter name>
    formula symbol: <formula symbol>/null
    typical values: <typical values>
...

only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

text_has_values = f"""{instruction_context} does the Input Text below menion any numerical quantities?

[Input Text start]
{{text}}
[Input Text end]

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

text_which_relations = f"""{instruction_context} we consider the following list of Entities.

Entities:
{{entities}}

What (if any) are parameters and values of the Entities above described in the Input Text below?

[Input Text start]
{{text}}
[Input Text end]

For each entitiy, list its parameter(s) and value(s), if any, in in the following YAML format.

Format:
---
- entity<N>:
    name: <entity name>
    has_parameters: true/false
    parameters:
        - parameter<N>:
            name: <parameter name>
            value: <parameter value>/null
            context: <value context>/null
...

Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

# first end-to-end prompt after some initial tweaking
# - re-state entity types in YAML
# - prefix “Input Text” with “LaTeX”
text_e2e = f"""{instruction_context} what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
{{text}}
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
- text_contains_entities: true/false
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity<N>:
        name: <entity name>
        type: <entity type>
        has_parameters: true/false
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>/null
                context: <value context>/null
...

Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

# first revision of end-to-end prompt with
# - quotation marks around YAML dict values
# - additionaly text to prevent inclusion of entities with out-of-scope type
text_e2e_rev230602 = f"""{instruction_context} what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
{{text}}
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
- text_contains_entities: true/false
- entities:
    - entity<N>:
        name: "<entity name>"
        type: dataset/model/method/loss function/regularization technique
        has_parameters: true/false
        parameters:
            - parameter<N>:
                name: "<parameter name>"
                value: "<parameter value>"/null
                context: "<value context>"/null
...

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type.
Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501


text_e2e_fillin = f"""{instruction_context} what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
{{text}}
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
- text_contains_entities: true/false
- entities:
    - entity<N>:
        id: <entity id>
        name: "<entity name>"
        type: dataset/model/method/loss function/regularization technique
        has_parameters: true/false
        parameters:
            - parameter<N>:
                id: <parameter id>
                name: "<parameter name>"
                value: "<parameter value>"/null
                vid: <value id>/null
                context: "<value context>"/null
                cid: <context id>/null
- text_annotated: "<annotated text>"
...

where in <annotated text> all entities, parameters, values, and contexts are annotated by enclosing them in square brackets, e.g. [eN|foo], [pN|bar], [vN|bam], or [cN|baz].

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type.
Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501


text_e2e_fillin_twostep_1 = f"""{instruction_context} what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
{{text}}
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
text_contains_entities: true/false
entities:
  - entity<N>:
      id: e<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type. Do not include entities of type task, metric, library, software, or API.
Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501


text_e2e_fillin_twostep_2 = f"""{instruction_context} consider the following two pieces of information below. (1) YAML Entity Information, and (2) LaTeX Input Text. Based on (1) YAML Entity Information, annotate all mentions of the entities, parameters, values, and contexts in (2) LaTeX Input Text, by enclosing the mentions in square brackets, e.g. [eN|foo], [pN.M|bar], [vN.M.O|bam], or [cN.M.O|baz].

[(1) YAML Entity Information start]
{{yaml}}
[(1) YAML Entity Information end]

[(2) LaTeX Input Text start]
{{text}}
[(2) LaTeX Input Text end]

Based on (1) YAML Entity Information, annotate all mentions of the entities, parameters, values, and contexts in (2) LaTeX Input Text, by enclosing the mentions in square brackets, e.g. [eN|foo], [pN.M|bar], [vN.M.O|bam], or [cN.M.O|baz].

Annotated text:\n"""  # noqa: E501


# Alpcaca style prompt
# (see: github.com/tatsu-lab/alpaca_farm/blob/main/examples/prompts/v0_inputs_noinputs.json)  # noqa: E501
# (see: github.com/tatsu-lab/stanford_alpaca/issues/8#issuecomment-1468487028)
alpaca_style_prompt_start = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"""  # noqa: E501
alpaca_style_prompt_instruction_pre = """### Instruction:\n"""
alpaca_style_prompt_response_pre = """### Response:"""

text_e2e_fillin_twostep_1_alpaca_style_nointro = f"""{alpaca_style_prompt_instruction_pre}{instruction_context} what are the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text below? What are their parameters and values?

[LaTeX Input Text start]
{{text}}
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
text_contains_entities: true/false
entities:
  - entity<N>:
      id: e<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Only produce output in the YAML format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text. Then end your response.

{alpaca_style_prompt_response_pre}"""  # noqa: E501

text_has_artifs_alpaca_style_nointro = f"""{alpaca_style_prompt_instruction_pre}{instruction_context} does the text below mention any datasets, models, methods, loss functions, or regularization techniques? Answer with only either "yes" or "no".

Text:
{{text}}

{alpaca_style_prompt_response_pre}"""  # noqa: E501

text_e2e_fillin_twostep_1_galacitca = text_e2e_fillin_twostep_1_alpaca_style_nointro  # noqa: E501

start_completion = """
---
text_contains_entities:"""

text_e2e_fillin_twostep_1_vicuna = text_e2e_fillin_twostep_1_alpaca_style_nointro + start_completion  # noqa: E501
