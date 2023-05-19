# Format description

Each element of the JSON file is a dictionary with the following keys.

- `annotator_id` (String): The unique ID of the annotator.
- `document_id` (String): The unique ID of the document where the paragraph is located.
- `paragraph_index` (Integer): The index of the paragraph within the document.
- `text` (String): The actual text of the paragraph.
- `annotation` (Dictionary): A dictionary of processed annotation information with the following keys:
- `entities` (Dictionary): A dictionary where each key is an entity ID and the value is another dictionary with the following keys:
  - `id` (String): The unique ID of the entity.
  - `type` (String): The type of the entity, which can be 'a', 'p', 'v', or 'c'.
  - `subtype` (String, optional): The subtype of the entity, only applicable when type is 'v'. It can be 'n' (number), 'r' (range), 's' (set), 'o' (other).
  - `surface_forms` (List): A list of dictionaries, each representing a surface form of the entity with the following keys:
    - `id` (String): The unique ID of the surface form.
    - `surface_form` (String): The actual text of the surface form.
    - `start` (Integer): The start offset of the surface form within the paragraph.
    - `end` (Integer): The end offset of the surface form within the paragraph.
- `relations` (Dictionary): A dictionary where each key is a relation ID and the value is another dictionary with the following keys:
  - `id` (String): The unique ID of the relation.
  - `source` (String): The entity ID of the source entity of the relation.
  - `target` (String): The entity ID of the target entity of the relation.
  - `evidences` (List): A list of dictionaries, each representing an evidence of the relation with the following keys:
    - `id` (String): The unique ID of the evidence.
    - `evidence_sentence` (String): The actual text of the evidence sentence.
    - `start` (Integer): The start offset of the evidence within the paragraph.
    - `end` (Integer): The end offset of the evidence within the paragraph.
- `annotation_raw` (List): A list of all the raw annotations made on this paragraph.


# Examples

**Top-Level Paragraph Dictionary**

A dictionary representing a paragraph from the annotated document.

```
{
  "annotator_id": "annotator1",
  "document_id": "doc1",
  "paragraph_index": 1,
  "text": "Example paragraph text...",
  "annotation_raw": [...],
  "annotation": {...}
}
```

**Entity Dictionary**

A dictionary holding data related to a unique entity annotated in the paragraph.

```
{
  "id": "a1",
  "type": "a",
  "subtype": null,
  "surface_forms": [...]
}
```

**Surface Form Dictionary**

A dictionary describing a specific occurrence of an entity within the text.

```
{
  "id": "fcd7d3c1-ff1c-4656-8ae7-798395693598",
  "surface_form": "Example",
  "start": 0,
  "end": 7
}
```

**Annotation Dictionary**

A dictionary containing structured information about the annotations, including both entities and relations.

```
{
  "entities": {...},
  "relations": {...}
}
```

**Relations Dictionary**

A dictionary representing a relationship between two entities.

```
{
  "id": "a1-p1",
  "source": "a1",
  "target": "p1",
  "evidences": [...]
}
```

**Evidence Dictionary**

A dictionary describing a piece of text evidence for a specific relationship.

```
{
  "id": "fcd7d3c1-ff1c-4656-8ae7-798395693598",
  "evidence_sentence": "Example sentence that proves the relation...",
  "start": 10,
  "end": 55
}
```
