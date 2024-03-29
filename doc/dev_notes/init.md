# Methodology

### First Steps

* Get SOTA models for similar tasks on similar data to run (see #2)
* Apply models on our data
* Identify challenges / error classes
* Decide on approaches to take to address these (might be specific to our data/use case)

### Ideas / Directions to Investigate

* Additional training data from un-annotated papers (~200k)
    * Based on rules (@xiao-nx: please add links)
    * Based on LLMs?
* Utilize background knowledge
    * By using LLMs
    * From some KG?
    * By utilizing citation network and cited papers full-text content (maybe not feasible b/c citation markers after artifacts are not that common)
* Consider entity disambiguation
    * Existing ontology?
    * Possible to create “by hand” for left edge of long tail distribution?

# MISC

* Make selling point clearer → why is it important to know about research artifact use on a large scale in ML literature?
    * “extend” papers with code
    * parameter recommendation for practitioners
    * autoML
    * reproducibility / research data management
    * richer paper representation (for IR, recommendation, etc.; potentially usable in [CRAUP](https://arxiv.org/abs/2303.15193) follow-up paper)
    * KG generation / enrichment
    * ...
