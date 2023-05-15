# Annotation Guidelines

## General

* Data selection
    * We annotate whole papers, while the “unit” of annotation (across which entity identifiers must be consistent, and which is used as model input later) is *one paragraph*.
        * entity IDs are, however, kep consistent throughout the document
    * Only choose to annotate a paper (=all of its paragraphs) if a reasonable portion of its content is (1) contained in the annotation UI and (2) usable (not corrupted/broken)
    * Due to preprocessing errors papers can include way too long/short text as “paragraphs”, these are skipped. More concretely, paragraphs are considered:
        * too long, if they obviously are the result from a preprocessing error and contain the content of many parahraphs
        * too short, if they contain less than a full sentence.
    * Properly preprocessed paragraphs with nothing to annotate must still be “opened” once in the annotation view and saved. (This in combination with above skipping rule make the annotation UI work as a quality selection tool as well)

## Entity Types

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

### Surface forms

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
    * extreme example 2: “propose the Robust Automated Production of Information Extraction Rules (RAPIER) algorithm. RAPIER is a form-filling algorithm that ...” → the whole “Robust Automated Production of Information Extraction Rules (RAPIER) algorithm” is one surface form, while “RAPIER” in the second sentence is a separate second surface form of the same entity
* **“full name (abbrev.)”**
    * if an entity is named, followed by an abbreviation or other form of co-reference in brackets, the whole construct is given a single label, not two separate ones
    * example: “We train a Long Short Term Memory (LSTM) using ...” → annotate “Long Short Term Memory (LSTM)” as one unit
* **surface forms as part of phrases**
    * if an entities surface form is used as part of a phrase, the surface form is annotated on its own
    * example: “Our LSTM predictions are better than ...” → annotate only “LSTM”
    * *caveat 1*
        * if the whole phrase represents the entity, annotate it whole (see bullet point “noun phrases” above)
        * example: “We train the LSTM network to ...” → annotate “LSTM netowrk”
    * *caveat 2*
        * if the surface form is used to represent something *semantically different*, it is not annotated
        * example: “The computation is shown as follows: \(w_{context_{l}} = CosSim(SciBERT(s),SciBERT(t_l))\) where \(SciBERT\) denotes the method of generating embedding using SciBERT [3].” → only annotate the “SciBERT” at the very end before “[3]”
* **no “sub-entities”**
    * if authors re-use (a) some model’s architecture and (b) some model’s weights, these are not considered to be separate artifacts
    * example: “we use a Baidu's DeepSpeech2 model weights” → annotate only “DeepSpeech2”
* **splitting**
    * if unavoidable, an single surface form can be annotated as multiple parts of the text
    * annotations are then enumerated, e.g. `a1-1` and `a1-2` for the two parts of a surface form of artifact `a1`

### Research Artifacts

Put simply, the following types of research artifacts are in scope in our scheme

* datasets (MNIST, DocRED, ...) (things like Wikipedia, arXiv.org etc. *usually* not though, these are primarily websites)
* methods (SGD, transformer, mean pooling, ...) (though not metrics like recall, h-index, etc.)
* models (BERT, GPT, ...)
* some software\* (see bullet point below)

This can be illustrated by the following sentences from arXiv paper 2210.10073 ([published version](https://doi.org/10.1007/s11192-022-04334-5)).

* “The first type is named by the original author, such as the neighbor-joining method [50], ATRP [61], ImageNet [14] and AlphaGo [52]. The second type is named later by other researchers, usually named after the original author. Examples include Schrödinger equation [51], Bradford's law [4], Turing test [55] and Witten-Bell smoothing [62]. Published scientific entity is also referred to as published entity in this paper.”
* in scope entities
    * neighbor-joining method → a clustering technique from the area of bioinformatics
    * ImageNet → a dataset
    * AlphaGo → a model
    * Schrödinger equation → a method
    * Witten-Bell smoothing  → a method
* out of scope entities
    * ATRP (Atom transfer radical polymerization, a chemical process)
    * Bradford's law → an observation / empirical law
    * Turing test → a general concept

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
    * edge cases:
        * “we use two existing data sets, [4] and [5] to train ...” → “[4]” and “[5]” clearly are data sets, so in scope even though they have no name
        * “we train the model proposed by Foo et al. to ...” → “model proposed by Foo et al.” clearly a model, so in scope even though they have no name
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
* **no tasks**
    * task are not considered research artifacts
    * out of scope examples:
        * “entity recognition”
        * “relation extraction”
* **metrics**
    * are only annotated if the authors clearly refer to the *method* of calculation, rather than the score
    * a good indicator for this is a citation marker after the surface form
    * in scope example:
        * “We can use BLEU [45] and SciBERT [3] in comparison as a distinct case in point. BLEU is a method for automatic evaluation of machine translation proposed in 2002. This method soon becomes and still is the most widely used metric.” → annotate “BLEU”
    * out of scope examples:
        * “we consider the h-index of”
        * “we calculate the F1 score for”
* **loss functions**
    * are in scope when authors talk about *the method*, but out of scope when referring to the resulting number
    * in scope examples:
        * “Foo et al. introduce triplet loss [1]”
        * “we use contrastive loss to ...”
    * out of scope examples:
        * “such that the loss is not calculated just for a single triplet but ...”
* **regularization techniques**
    * are considered in-scope, **but** they can be tricky in terms of deciding whether to see them as artifacts or parameters. We go with how the authors describe it
    * examples:
        * “we use a FFNN with dropout set to 0.2” → dropout is described as a parameter of the FFNN, with a value of 0.2
        * “we use the R1-regularizer with γ = 10” → R1 regularization described as an artifact with parameter gamma set to 10
* **LaTeX math mode**
    * if possible, don’t annotate the markers of the beginning and end of LaTeX math mode, i.e. `\(`, `\)`, `$`, etc.
    * example: “we set \(\beta\_1\) to 0.9” → annotate only “\beta\_1”

### Parameters

There are some special considerations for parameters.

* **variable aspects**
    * parameters are *variable* aspects *chosen* by the authors
    * in scope are therefore
        * things that have to be set anyway (such as the learning rate for SGD)
        * things that the authors *change* about an artifact (such as sample size taken from a data est)
    * out of scope are mere properties of artifacts, such as the number of documents/images/... in a data set
* **artifact related**
    * parameters are in scope when they are a parameter of a research artifact that is in scope (the artifact in question does not have to be present in the text, however).
    * in scope examples:
        * “we use Adam with β1 = 0.999” → annotate β1
        * “when training neural networks, the choice of learning rate is important” → annotate “learning rate” even though SGD is not explicitly mentioned
    * out of scope examples:
        * “... can be calculated as: \(x = \lambda w_{count_{l}}+1\) where the hyperparameter \(\lambda \) is used to ...” → do not annotate “\lambda” because the simple formula is not an artifact

### Values

There are some special considerations for values.

* **annotation scope**
    * to facilitate evaluations of relation extraction only models (i.e., not joint NER+RE), we consider any part of the text that expresses a *numerical value* / *quantity* to be a value, not just those that are linked to research artifact parameters. However, we do not consider *identifiers* to be values.
    * in scope examples:
        * “we use learning rage of 0.001”
        * “we achieve an F1 score of 0.93”
        * “annotated by three domain experts”
        * “one special case to consider is”
    * out of scope examples:
        * “see Figure 1”
        * “Foo et al. [2]”
        * “we use GPT 3.5”
        * “arXiv.org:1905.0245”
* **units**
    * we do not annotate units (e.g. “seconds”, “pixels”, etc.)
    * “factors” appended to digits which are necessary to attain the quantity expressed are annotated though; examples are
        * K (for thousand)
        * M (for million)
        * % (for hudredth)
* **“final” values**
    * numbers within calculations (e.g. “1” in “we set x=k-1”) are not annotated
* **no years**
    * while technically representing the amount of years passed since AD, years are not annotated

### Additional Notes

In case of doubt

* if text in the annotation view might be “wrong”/not representative of the original paper content → look at the original paper PDF (linked in the annotation UI)
* if a certain part of the text/abbreviation actually refers to a model/method/dataset → look online. paperswithcode.com is a good resource for this

If still in doubt

* leave if out (e.g. if still not sure if some abbreviation refers to an artifact or not, or if a certain parameter is related to an artifact or not). Reason: a missing annotation is less harmful than a wrong one.

## Relation Types

We only annotate one generic relation type “used-for”, which relates above entity types as follows.

* context –used–for→ value
* value –used–for→ parameter
* parameter –used–for→ research artifact

Co-reference relations are later generated from matching labels such as `a1`, `a1`.
