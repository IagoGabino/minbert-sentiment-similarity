# minBERT: Sentiment Classification and Semantic Textual Similarity

A BERT model, with its core components implemented from a starter skeleton, fine-tuned for sentiment analysis and sentence similarity, plus experiments on data augmentation and regularization.

NLP coursework (UFMG), individual project. It is based on minBERT, the default final project of Stanford's CS224N (Natural Language Processing with Deep Learning). The starter code and the task structure come from that project. My part:

- Implemented multi-head self-attention and the AdamW optimizer in the starter skeleton.
- Built the classification and similarity heads, training scripts and evaluation.
- Ran the data augmentation and regularization experiments listed below.

See [NOTICE.md](NOTICE.md) for the origin of the code.

## Tasks

- **SST:** 5-class sentiment classification.
- **CFIMDB:** binary sentiment classification.
- **STS:** semantic similarity of sentence pairs (score 0–5, SemEval STS Benchmark).

Each model is run in two modes: `pretrain` (only the classification head is trained) and `finetune` (all layers).

## Experiments and results

| Notebook | Experiment | Dev result |
|---|---|---|
| `sst_cfimdb_v1_base_model` | Base model on SST and CFIMDB | CFIMDB: 97.1% accuracy |
| `sst_data_aug` | Builds the augmented SST training sets | |
| `sst_v2_data_aug_dense` | Augmentation, regularization and an extra dense layer | SST: 52.1% accuracy |
| `sst_v3_data_aug` | Augmentation and regularization | SST: 53.1% accuracy |
| `sts_v1_base_model` | Base model on STS | |

Error analysis notebooks are in `analysis/`.

## Running

The datasets are not included. Place the course CSV files in `data/` (`ids-sst-*.csv`, `ids-cfimdb-*.csv`, `sts-*.csv`), then:

```bash
pip install -r requirements.txt
# run the notebooks in the order above
```

## References

- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Stanford CS224N default final project handout (minBERT)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/project/default-final-project-handout-minbert-2024.pdf)
