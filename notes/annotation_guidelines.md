# Annotation Guidelines

### General

* Data selection
    * We annotate whole papers, while the “unit” of annotation (across which entity identifiers must be consistent, and which is used as model input later) is *one paragraph*.
        * entity IDs are, however, kep consistent throughout the document
    * Only choose to annotate a paper (=all of its paragraphs) if a reasonable portion of its content is (1) contained in the annotation UI and (2) usable (not corrupted/broken)
    * Due to preprocessing errors papers can include way too long/short text as “paragraphs”, these are skipped. More concretely, paragraphs are considered:
        * too long, if they obviously are the result from a preprocessing error and contain the content of many parahraphs
        * too short, if they contain less than a full sentence.
    * Properly preprocessed paragraphs with nothing to annotate must still be “opened” once in the annotation view and saved. (This in combination with above skipping rule make the annotation UI work as a quality selection tool as well)

### Entity Types

The following is a short description of the entity types. Further down are more detailed descriptions.

* A **research artifact** is a procedure or piece of data used by researchers.
* A **parameter** is a variable aspect of an artifact..
* A **value** is the description of the state of aforementioned aspect at the time of usage. We distinguish four types.
    * *number* (e.g. “27”, “10^-4”, “3%”)
    * *set* (e.g. “(0.01, 0.001)”)
    * *range* (e.g. “[1..10]”)
    * *other*: (e.g. “0.01 × i”)
* a **context** is a specific context in which the value applies.

**Example 1**: “During fine-tuning, we use the Adam optimizer with a learning rate of α=10^−4.” (based on text in arXiv.org:1905.0245)

* research artifact: “Adam optimizer”
* parameter: “learning rate” and “α”
* value: “10^-4”
* context “during fine-tuning”

Entities are annotated even when they have no relation. For example an artifact without the mention of a parameter.

To facilitate quick annotation, we use the following short forms.

* research artifact: `a<id>`
* parameter: `p<id>`
* value
    * number: `vn`
    * set: `vs`
    * range: `vr`
    * other: `vo`
* context: `c`

#### Surface forms

Naturally, we annotate surface forms in the text and not entities. Entities are only referred to in the text. This has the following consequences.

* **co-references**
    * if an entity has multiple surface forms throughout the document, this is reflected in the annotation by labeling each surface form of one entity using the same `<id>`
    * example 1: “We set the learning rate to α=0.01” → annotate both “learning rate” and “α” with e.g. `p1`
    * example 2: “We train a RNN and an LSTM. For the former, we set ...” → annotate both “RNN” and “the former” as `a1`
* **no overlaying annotations**
    * a surface form should have no more than one annotation
        * example: “As pooling strategies we use either mean and max pooling.”
        * annotation:  “mean” as a first artifact, and “max pooling” as a second (in the example, “mean” is the surface form refering to the method mean pooling)
* **noun phrases** (and similar constructs)
    * we lean towards how authors refer to something in their paper, rather than its canonical name
    * example: “we use the MNIST data set” → annotate the whole noun phrase “MNIST data set” rather than just “MNIST”; do not annotate the article “the”
    * extreme example 1: “we re-train the Flair tagger on the capitalized NER benchmark CoNLL-2003 [22] dataset” → whole noun phrase is “capitalized NER benchmark CoNLL-2003 [22] dataset”
    * extreme example 2: “propose the Robust Automated Production of Information Extraction Rules” (RAPIER) algorithm. RAPIER is a form-filling algorithm that ...” → the whole ““Robust Automated Production of Information Extraction Rules” (RAPIER) algorithm” is one surface form, while “RAPIER” in the second sentence is a separate second surface form of the same entity
    -> "RAPIER"
* **full name (abbrev.)**
    * if an entity is named, followed by an abbreviation or other form of co-reference in brackets, the whole construct is given a single label, not two separate ones
    * example: “We train a Long Short Term Memory (LSTM) using ...” → annotate “Long Short Term Memory (LSTM)” as one unit
* **surface forms as part of phrases**
    * if an entities surface form is used as part of a phrase, the surface form is annotated on its own
    * example: “Our LSTM predictions are better than ...” → annotate only “LSTM”
    * *however*, if the whole phrase represents the entity, annotate it whole (see bullet point “noun phrases” above)
    * example: “We train the LSTM network to ...” → annotate “LSTM netowrk”
* **no “sub-entities”**
    * if authors re-use (a) some model’s architecture and (b) some model’s weights, these are not considered to be separate artifacts
    * example: “we use a Baidu's DeepSpeech2 neural architecture” → annotate only “DeepSpeech2”
* **splitting**
    * if unavoidable, an single surface form can be annotated as multiple parts of the text
    * annotations are then enumerated, e.g. `a1-1` and `a1-2` for the two parts of a surface form of artifact `a1`

#### Research Artifacts

Put simply, the following types of research artifacts are in scope in our scheme

* datasets (MNIST, DocRED, ...) (things like Wikipedia, arXiv.org etc. *usually* not though, these are primarily websites)
* methods (SGD, transformer, mean pooling, ...) (though not metrics like recall, h-index, etc.)
* models (BERT, GPT, ...)
* some software\* (see bullet point below)

There are some special considerations for research artifacts.

* **(named) entities**
    * we annotate specific artifacts and not general terms
    * good indicators are, for example,
        * others (re-)using an artifact would refer to it the same way the paper in question does
        * there is a citation marker after the artifact surface form
    * the single defined “exception” to this is authors using phrases like “our model”, “our dataset”, etc. These are annotated even if the model/dataset/... is never given a name within the paper
    * reference to previous work without an “artifact name” should only be annotated, if from the context it is clear that an artifact (method, model, dataset, ...) is being referred to, rather than a paper as a whole
    * in scope examples:
        * “we use Adam [1]”
        * “we use a feed forward neural network”
        * “our model performs well on ...”
        * “we use two existing data sets, [4] and [5] to train ...”
        * “we train the model proposed by Foo et al. to ...”
    * out of scope examples:
        * “we approach the task using an end-to-end architecture”
        * “we use transformer-based language models”
        * “recently, LLMs have seen much attention” (not a single one / “instance of the class” being talked abouot)
* **software**
    * the boundary between software and models/methods is somewhat blurry
    * software is only in scope, when it is single purpose, i.e. could be described as model and potentially has parameters to be set when executed
    * in scope examples:
        * LaTeXML (converts LaTeX into XML)
        * GROBID (converts PDF into XML)
    * out of scope examples:
        * Pandas
        * spaCy
        * Excel
* **no tasks or metrics**
    * task and metrics are not considered research artifacts
    * out of scope examples:
        * “entity recognition”
        * “relation extraction”
        * “we consider the h-index of”
        * “we calculate the F1 score for”

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
* **“final” values**
    * numbers within calculations (e.g. “1” in “we set x=k-1”) are not annotated
* **no years**
    * while technically representing the amount of years passed since AD, years are not annotated

#### Additional Notes

In case of doubt

* if text in the annotation view might be “wrong”/not representative of the original paper content → look at the original paper PDF (linked in the annotation UI)
* if a certain part of the text/abbreviation actually refers to a model/method/dataset → look online. paperswithcode.com is a good resource for this

If still in doubt

* leave if out (e.g. if still not sure if some abbreviation refers to an artifact or not, or if a certain parameter is related to an artifact or not). Reason: a missing annotation is less harmful than a wrong one.

### Relation Types

We only annotate one generic relation type “used-for”, which relates above entity types as follows.

* context –used–for→ value
* value –used–for→ parameter
* parameter –used–for→ research artifact

Co-reference relations are later generated from matching labels such as `a1`, `a1`.
