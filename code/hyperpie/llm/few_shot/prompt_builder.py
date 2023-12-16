# Prompt Bilder
#
#
from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from utils import get_yaml, yaml2json


JSON_FORMAT = """{{
  "has_entities": true/false,
  "entities": [
    {{
      "entity<N>": {{
        "id": "a<N>",
        "name": "<entity name>",
        "type": "dataset/model/method/loss function/regularization technique",
        "has_parameters": true/false,
        "parameters": [
          {{
            "parameter<M>": {{
              "id": "p<N.M>",
              "name": "<parameter name>",
              "has_values": true/false,
              "values": [
                {{
                  "value<O>": {{
                    "value_id": "v<N.M.O>",
                    "value": "<parameter value>",
                    "context": "<value context>"/null,
                    "context_id": "c<N.M.O>"/null
                  }}
                }}
              ]
            }}
          }}
        ]
      }}
    }}
  ]
}}
"""

YAML_FORMAT = """has_entities: true/false
entities:
  - entity<N>:
      id: a<N>
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
"""

PROMPT_TEMPLATE = """### Instruction:
In the context of machine learning and related fields, what are the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text below? What are their parameters and values?

Answer in the following {format_name} format.

Format:
```
{format}
```{examples}

Only produce output in the {format_name} format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text. Then end your response.

[LaTeX Input Text start]
{text}
[LaTeX Input Text end]

### Response:
```
"""

EXAMPLE_TEMPLATE = """### Example{n}:

[LaTeX Input Text start]
{text}
[LaTeX Input Text end]

### Response{n}:
```
{response}```"""


JSON_GRAMMAR = """
root ::= "\n"? "{\n" tab1 hasentities "}\n"  "```\n"

hasentities ::= ["] "has_entities" ["] ": " ( "false\n" | "true,\n" tab1 entities)

entities ::= ["] "entities" ["] ": [\n" (entity)+ tab1 "]\n"

entity ::= tab2 "{\n" tab3 entitykey "{\n" tab4 idkey eid ",\n" tab4 namekey name ",\n" (tab4 typekey type ",\n")? tab4 hasparameters tab3 "}\n" tab2 "}" ","? "\n"

typekey ::= ["] "type" ["] ": "

type ::= ["] ("dataset" | "model" | "method" | "loss function" | "regularization technique") ["]

hasparameters ::= ["] "has_parameters" ["] ": " ( "false\n" | "true,\n" tab4 parameters)

parameters ::= ["] "parameters" ["] ": [\n" (parameter)+ tab4 "]\n"

parameter ::= tab5 "{\n" tab6 parameterkey "{\n" tab7 idkey pid ",\n" tab7 namekey name ",\n" tab7 hasvalues tab6 "}\n" tab5 "}" ","? "\n"

hasvalues ::= ["] "has_values" ["] ": " ( "false\n" | "true,\n" tab7 values)

values ::= ["] "values" ["] ": [\n" (value)+ tab7 "]\n"

value ::= tab8 "{\n" tab9 valuekey "{\n" tab10 vidkey vid ",\n" tab10 vnamekey name ",\n" tab10 cnamekey (name | "null") ",\n" tab10 cidkey (cid | "null") "\n" tab9 "}\n" tab8 "}\n"

word ::= [^ \n\t"]+

words ::= word (ws word)* 

ws ::= [ ]+

name ::= ["] words ["]

entitykey ::= ["] "entitiy" n3 ["] ": "

parameterkey ::= ["] "parameter" n3 ["] ": "

valuekey ::= ["] "value" n3 ["] ": "

idkey ::= ["] "id" ["] ": "

vidkey ::= ["] "value_id" ["] ": "

cidkey ::= ["] "context_id" ["] ": "

namekey ::= ["] "name" ["] ": "

vnamekey ::= ["] "value" ["] ": "

cnamekey ::= ["] "context" ["] ": "

tab1 ::= "  "

tab2 ::= tab1 tab1

tab3 ::= tab2 tab1

tab4 ::= tab3 tab1

tab5 ::= tab4 tab1

tab6 ::= tab5 tab1

tab7 ::= tab6 tab1

tab8 ::= tab7 tab1

tab9 ::= tab8 tab1

tab10 ::= tab9 tab1

eid ::= ["] "a" n3 ["]

pid ::= ["] "p" n3 ["]

vid ::= ["] "v" n3 ["]

cid ::= ["] "c" n3 ["]

n3 ::= [1-9][0-9]?[0-9]?
"""

YAML_GRAMMAR = """
root ::= "\n"? "has_entities: " ("false\n" | "true\nentities:\n" (entity)+) "```\n"

entity ::= ws2 "- entity" n3 ":" linebreak1 "id: " eid linebreak1 "name: " name linebreak1 ("type: " type linebreak1)? "has_parameters: " ("false\n" | "true" linebreak1 "parameters:\n" (parameter)+)

parameter ::= indent ws2 "- parameter" n3 ":" linebreak2 "id: " pid linebreak2 "name: " name linebreak2 "has_values: " ("false\n" | "true" linebreak2 "values:\n" (value)+)

value ::= indent indent ws2 "- value" n3 ":" linebreak3 "value_id: " vid linebreak3 "value: " name linebreak3 "context: " ("null" | name ) linebreak3 "context_id: " ("null" | cid ) "\n"

type ::= ("dataset" | "model" | "method" | "loss function" | "regularization technique")

linebreak1 ::= "\n" indent

linebreak2 ::= "\n" indent indent

linebreak3 ::= "\n" indent indent indent

word ::= [^ \n\t"]+

words ::= word (ws word)* 

name ::= ["] words ["]

ws ::= [ ]+

ws2 ::= "  "

ws4 ::= "    "

indent ::= "      "

eid ::= "a" n3

pid ::= "p" n3

vid ::= "v" n3

cid ::= "c" n3

n3 ::= [1-9][0-9]?[0-9]?
"""

class PromptBuilder:
    def __init__(self, format="yaml"):
        assert format in ["json", "yaml"], f"{format} format is not supported."
        format_template = JSON_FORMAT if format == "json" else YAML_FORMAT
        template = PROMPT_TEMPLATE.replace('{format}', format_template)
        template = template.replace('{format_name}', format.upper())
        self.prompt = PromptTemplate.from_template(template)
        self.example = PromptTemplate.from_template(EXAMPLE_TEMPLATE)
        self.format_type = format


    def format(self, text: str, data: List[dict], examples: Optional[List[Document]]=None):
        if examples is None or isinstance(examples, list) and len(examples) == 0:
            # zero-shot
            prompt_str = self.prompt.format(text=text, examples="")

        elif len(examples) > 0:
            if len(examples) == 1:
                # one-shot
                example_str = "\n\nHere is an example.\n\n"
                example_text = examples[0].page_content
                example_anno = get_yaml(data[examples[0].metadata['row']]['annotation'])
                if self.format_type == "json":
                    example_anno = yaml2json(example_anno)
                example_str = self.example.format(
                    n="",
                    text=example_text,
                    response=example_anno,
                )

            elif len(examples) > 1:
                # few-shot
                example_str = "\n\nHere are several examples."
                for i, ex in enumerate(examples):
                    example_str += "\n\n"
                    example_text = ex.page_content
                    example_anno = get_yaml(data[ex.metadata['row']]['annotation'])
                    if self.format_type == "json":
                        example_anno = yaml2json(example_anno)
                    example_str += self.example.format(
                        n=f" {i+1}",
                        text=example_text,
                        response=example_anno,
                    )

            prompt_str = self.prompt.format(text=text, examples=example_str)
        return prompt_str
