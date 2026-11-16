# Notice

This repository is a student project built on the minBERT default final project of Stanford's CS224N (Natural Language Processing with Deep Learning). The following parts come from that starter code and are not my original work:

- The project structure and the files in `bert/` (`base_bert.py`, `config.py`, `tokenizer.py`, `utils.py` and the skeleton of `bert.py` and `optimizer.py`).
- The dataset loaders and the overall layout of the classifiers (`sst_classifier.py`, `sts_dataset.py`).

Parts of the starter code, mainly the tokenizer and the utilities, are adapted from the [Hugging Face Transformers](https://github.com/huggingface/transformers) library, which is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

My own work: the self-attention and AdamW implementations, the classification and similarity heads, the training and evaluation code, and the experiments in the notebooks.

The datasets (SST, CFIMDB, STS Benchmark) are not included and keep their original licenses.

No license is granted for the starter code. Reuse of the original parts is subject to the terms of their sources.
