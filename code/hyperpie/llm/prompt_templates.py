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

# TODO:
# - test revision of e2e zero shot prompt (test single examples, see if reduces FPs)
# - test prompot which results in [a1]foo[/a1] types inserts in input text
# - test few shot scenarios

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
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity<N>:
        name: "<entity name>"
        type: "<entity type>"
        has_parameters: true/false
        parameters:
            - parameter<N>:
                name: "<parameter name>"
                value: "<parameter value>"/null
                context: "<value context>"/null
...

Only include entities that are of type dataset, model, method, loss function, or regularization technique.
Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501
