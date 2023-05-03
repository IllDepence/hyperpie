# Annotation Guidelines

## Annotation Scheme

### Entity Types

* A **research artifact** is a procedure or piece of data used by researchers.
* A **parameter** is a variable aspect of an artifact.
* A **value** is the description of the state of aforementioned aspect at the time of usage. We distinguish four types.
    * *number* (e.g. “27”, “10^-4”, “3%”)
    * *set* (e.g. “(0.01, 0.001)”)
    * *range* (e.g. “[1..10]”)
    * *other*: (e.g. “0.01 × i”)
4. a **context** is a specific context in which the value applies.

**Example 1**: “During fine-tuning, we use the Adam optimizer with a learning rate of α=10^−4.” (based on text in arXiv.org:1905.0245)

* research artifact: “Adam optimizer”
* parameter: “learning rate” and “α”
* value: “10^-4”
* context “during fine-tuning”

Entities are annotated even when they have no relation. For example an artifact without the mention of a parameter.

To facilitate quick annotation, we use the following short forms.

* research artifact: `a`
* parameter: `p`
* value
    * number: `vn`
    * set: `vs`
    * range: `vr`
    * other: `vo`
* context: `c`

#### Surface forms

Naturally, we annotate surface forms in the text and not entities. Entities are only referred to in the text. This has the following consequences.

* **co-references**
    * if in one “document” (in our case a paragraph) an entity has multiple surface forms, this is reflected in the annotation by using numbers
    * example 1: “We set the learning rate to α=0.01” → annotate both “learning rate” and “α” with e.g. `a1`
    * example 2: “We train a RNN and an LSTM. For the former, we set ...” → annotate both “RNN” and “the former” as `a1`
* **no overlaying annotations**
    * a surface form should have no more than one annotation
        * example: “As pooling strategies we use either mean and max pooling.”
        * annotation:  “mean” as a first artifact, and “max pooling” as a second (in the example, “mean” is the surface form refering to the method mean pooling)
* **noun phrases**
    * we lean towards how authors refer to something in their paper, rather than its canonical name
    * example: “we use the MNIST data set” → annotate the whole noun phrase “MNIST data set” rather than just “MNIST”; do not annotate the article “the”
    * extreme example: “we re-train the Flair tagger on the capitalized NER benchmark CoNLL-2003 [22] dataset” → whole noun phrase is “capitalized NER benchmark CoNLL-2003 [22] dataset”
* **no “sub-entities”**
    * if authors re-use (a) some model’s architecture and (b) some model’s weights, these are not considered to be separate artifacts
    * example: “we use a Baidu's DeepSpeech2 neural architecture” → annotate only “DeepSpeech2”
* **splitting**
    * if unavoidable, an single surface form can be annotated as multiple parts of the text
    * annotations are then enumerated, e.g. `a1-1` and `a1-2` for the two parts of a surface form of artifact `a1`

#### Research Artifacts

There are some special considerations for research artifacts.

* **(named) entities**
    * we annotate specific artifacts and not general terms
    * good indicators are, for example,
        * others (re-)using an artifact would refer to it the same way the paper in question does
        * there is a citation marker after the artifact surface form
    * in scope examples:
        * “we use Adam [1]”
        * “we use a feed forward neural network”
    * out of scope examples:
        * “we approach the task using an end-to-end architecture”
        * “we use transformer-based language models”
* **no tasks**
    * a task is not a research artifact
    * out of scope examples:
        * “entity recognition”
        * “relation extraction”

#### Values

There are some special considerations for values.

* **Annotation scope**
    * to facilitate evaluations of relation extraction only models (i.e., not joint NER+RE), we consider any part of the text that expresses a *numerical value* to be a value, not just those that are linked to research artifact parameters. However, we do not consider *identifiers* to be values.
    * in scope examples:
        * “we use learning rage of 0.001”
        * “we achieve an F1 score of 0.93”
        * “annotated by three domain experts”
    * out of scope examples:
        * “see Figure 1”
        * “we use GPT 3.5”
        * “arXiv.org:1905.0245”
* **Units**
    * We do not annotate units.
    * “%” is not considered a unit.
* **LaTeX math mode**
    * if possible, don’t annotate the markers of the beginning and end of LaTeX math mode, i.e. `\(`, `\)`, `$`, etc.
    * example: “we set \(\beta\_1\) to 0.9” → annotate only “\beta\_1”

#### Additional Notes

In case of doubt, leave if out (e.g. if not sure if some abbreviation refers to an artifact or not, or if a certain parameter is related to an artifact or not). Reason: a missing annotation is less harmful than a wrong one.

### Relation Types

We only define one generic relation type “used-for”, which relates above entity types as follows.

* context –used–for→ value
* value –used–for→ parameter
* parameter –used–for→ research artifact
