# Notes

* bigscience/bloomz
    * 1 token/min when run on 4 A100s
* facebook/galactica-120b
    * author defined prompt format + aplaca style format + authors also give natural language free form examples
    * depending on prompt gives reasonable, but not quite format conforming output
    * generally output quality/usefulness appears to vary *greatly* based on small prompt changes
* tiiuae/falcon-40b-instruct
    * author defined prompt format + aplaca style format + authors also give natural language free form examples
    * depending on prompt gives some reasonable output, but format does not adhere to output format
* GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k
    * author defined prompt format + tried “free form”
    * somewhat promising results but doesn’t know when to stop
* NousResearch/Nous-Hermes-13b
    * author defined prompt format
    * promising results
* eachadea/vicuna-13b
    * author defined prompt format
    * depending on prompt gives some reasonable output, but format does not always adhere to output format and does not know when to stop
* eachadea/vicuna-7b-1.1
    * author defined prompt format
    * promising results but doesn’t know when to stop
* AlpinDale/pygmalion-instruct
    * intended for “role playing”, “out of scope: Assistant Bot [subject to providing incorrect instructions]”
    * promising results but doesn’t know when to stop
* WizardLM/WizardLM-13B-V1.1
    * author defined prompt format
    * some sensible output but has problems with adhering to format (may output JSON instead of YAML)
* bigscience/bloomz-7b1
    * authors give natural language free form examples
    * just repeats part of input
* tiiuae/falcon-7b-instruct
    * author give natural language free form examples
    * just repeats part of input
* facebook/opt-13b
    * expected instruction somewhat unclear
    * just repeats part of input
* ~~EleutherAI/gpt-j-6b~~
    * “not intended for deployment without fine-tuning, supervision, and/or moderation”
    * just repeats part of input
