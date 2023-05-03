# Annotation Guidelines

### Annotation Scheme

##### Entity Types

There are four types of entities.

1. A **research artifact** is a procedure or piece of data used by researchers.
2. A **parameter** is a variable aspect of an artifact.
3. A **value** is the description of the state of aforementioned aspect at the time of usage.
4. A **context** is a specific context in which the value applies.

**Example 1**: “During fine-tuning, we use the Adam optimizer with a learning rate of α=10^−4.” (based on text in arXiv.org:1905.0245)

* research artifact: “the Adam optimizer”
* parameter: “learning rate” and “α”
* value: “10^-4”
* context “during fine-tuning”

Artifacts are annotated even when they have no relation. For example an artifact without the mention of a parameter.

**Special note on values**: to facilitate evaluations of relation extraction only models (i.e., not joint NER+RE), we consider any part of the text that expresses a *quantity* to be a value, but we do not consider *identifiers* to be values. For example, “

##### Relation Types

We only define one generic relation type “used-for”, which relates above entity types as follows.

* context –used–for→ value
* value –used–for→ parameter
* parameter –used–for→ research artifact

**Note** that relations are annotated on the level of entities, not their surface forms. I.e., in *Example 1* above, there is a single relation between (“learning rate”, “α”) and (“the Adam optimizer”) and not one relation for each surface form of the parameter.

##### Additional Notes

* **surface forms**
    * LaTeX brackets stuff etc.
* **research artifact**
    * a specific single artifact that researchers use and name  
      *out of scope*: “transformer-based language models” → too generic
    * a task is not a research artifact  
      *out of scope*: “entity recognition”, “relation extraction”, etc.

---

<!--

*A research artifact should be annotated even if it has no parameters. 
*Only named research artifact(and their co-refereces)  should be annotated. 
Unnamed ones should not be annotated as research artifacts, but their parameter and values should be annotated.
	e.g.:  We train our model with a learning rate of 20.
	e.g.:  In this paper, we propose an end-to-end approach for joint entity and relation extraction.
	e.g.:  We use a feed forward neural network.
Hint: Does other researchers name it the same way ? If so then it is a research artifact.
Hint: Citation markers are also a good indicator.

    3. When the surface form of an entity is split into two or more parts in the sentence then for each part we assign e1-x. 

    4. The same surface form should not be annotated twice.
e.g.:  As pooling strategies, we use either mean or max pooling: mean : a1, max pooling: a2.

    5. Lean more to what researchers call the entity(whole noun phrases)
    6.  and don’t annotate articles. 
e.g.:  We use the BERT model: BERT model is a research artifact, “model” is included but not “the”. 

    7. We distinguish 4 categories of values:
*Numerical value, vn. e.g.: 8, 2%.

*Set value, vs. e.g.: As our optimizer, we use AdamW [6] with learning rates \(\in [3\mathrm {e}{-5}, 5\mathrm {e}{-5}, 7\mathrm {e}{-5}]\):
Research artifact: AdamW.
parameters: learning rate.
vs: [3\mathrm {e}{-5}, 5\mathrm {e}{-5}, 7\mathrm {e}{-5}]

*Range value, vr. e.g.:  ?

*Other value, vo. e.g.: We set the learning rate for SGD in epoch i to 0.01×i: 
Research artifact: SGD.
parameters: learning rate.
vo: 0.01×i
context: in epoch i

    8. Units : OUT of scope

    9. If possible don’t annotate Latex extra characters.

    10. In case of doubt, leave it out. Missing entities/relations better than wrong ones.

    11. We only annotate surface forms and not actual artifact names. (coreferences) 

e.g.: We use two techniques, mean and max pooling. For the former, we set the parameter x to 1, and for the latter, we set it to 2.

mean→a1
max pooling→a2
the former→a1
x→p1
1→v
the latter→a2
it→p1  

    12. (case by case decision) we use AdamW [6] with learning rates \(\in [3\mathrm {e}{-5}, 5\mathrm {e}{-5}, 7\mathrm {e}{-5}]\) , a linear warmup of 1 epoch followed by a linear decay to zero, for a total of 60 epochs.
(2 possibilities)
    13. what about longer noun phrases?
e.g. “we re-train the Flair tagger on the capitalized NER benchmark CoNLL-2003 [22] dataset”
whole noun phrase is “capitalized NER benchmark CoNLL-2003 [22] dataset” → annotate all of it!

-->
