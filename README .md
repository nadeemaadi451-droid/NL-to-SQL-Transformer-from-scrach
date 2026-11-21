# Text-to-SQL Transformer from Scratch

A small **encoder-decoder Transformer** implemented from scratch in PyTorch (no pretrained weights) that translates a natural-language question into a SQL query, given the table schema as context.

Example of the task (from the dataset):

```
Input  : CREATE TABLE head (age INTEGER)
         How many heads of the departments are older than 56 ?
Target : SELECT COUNT(*) FROM head WHERE age > 56
```

> This is a learning project. The model is small and generalizes poorly to new schemas (see [Results](#results) and [Limitations](#limitations)). Always review generated queries before running them on a real database.

## Features

- Custom **byte-level BPE tokenizer** (Hugging Face `tokenizers`, 5,000-token vocabulary) with `<pad>`, `<bos>`, `<eos>` and `<sep>` special tokens
- **Encoder**: bidirectional self-attention over the schema + question, with a padding mask
- **Decoder**: causal (masked) self-attention, then **cross-attention** to the encoder output, then a feed-forward layer
- Pre-LayerNorm residual blocks, learned positional embeddings, multi-head attention written by hand
- Loss ignores padded positions (`-100` labels)
- Greedy decoding that stops at `<eos>`
- Train/test loss comparison and exact-match evaluation on held-out data

## Architecture

```
 schema + question ──► Embedding + Position ──► [ Encoder block ] x 3 ──► encoder output
                                                                              │ (keys, values)
 <bos> + answer ─────► Embedding + Position ──► [ Decoder block ] x 3 ◄───────┘
                                                  1. masked self-attention
                                                  2. cross-attention (query from decoder)
                                                  3. feed-forward
                                                         │
                                                  Linear ► logits ► next SQL token
```

The encoder only sees the schema and question, and the decoder only sees the answer generated so far, so the answer cannot leak into the encoder.

## Data

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context): 78,577 examples of (`CREATE TABLE` context, question, SQL answer), built from WikiSQL and Spider.

Each example is formatted as:

```
<context> <question> <sep> <answer> <eos>
```

- Random 90% / 10% split (`seed=42`)
- Examples longer than 400 characters are dropped, leaving **69,549 train** and **7,733 test** examples
- Token lengths: max 200, mean about 62
- For each batch, text is split at `<sep>`:
  - `src` = context + question (encoder input)
  - `dec_in` = `<bos>` + answer (decoder input)
  - `y` = answer + `<eos>` (labels, shifted by one)

## Hyperparameters

| Setting | Value |
|---|---|
| Vocabulary size | 5,000 |
| Context size (max tokens) | 200 |
| Embedding size | 128 |
| Attention heads | 4 |
| Layers (encoder and decoder each) | 3 |
| Dropout | 0.2 |
| Batch size | 64 |
| Optimizer | AdamW, lr = 1e-3 (constant) |
| Training steps | 5,000 (GPU: Colab T4) |

## Results

Initial loss was 8.78 (close to ln 5000 ≈ 8.5, as expected for a random model).

| Metric | Value |
|---|---|
| Train loss | 0.221 |
| Test loss | 0.290 |
| Train/test gap | 0.070 |
| Exact match (first 100 test examples, case and whitespace normalized) | 45/100 = 45.0% |

Exact match is measured on only 100 examples, so treat it as a rough estimate.

### Example predictions (held-out test set)

Correct:

| Reference | Prediction |
|---|---|
| `SELECT venue FROM table_name_50 WHERE away_team = "essendon"` | same |
| `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"` | same |

Incorrect:

| Reference | Prediction | What went wrong |
|---|---|---|
| `SELECT opponent FROM table_name_37 WHERE week = "4"` | `... WHERE week = 4` | Quoting style (arguably equivalent) |
| `SELECT SUM(points) FROM table_name_70 WHERE ...` | `SELECT AVG(points) FROM table_name_70 WHERE ...` | Wrong aggregate function |
| `... WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"` | `... transfer_window = "sco" AND moving_to = "sco and to greenock morton"` | Wrong values |
| `SELECT DISTINCT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)` | `SELECT state FROM college WHERE enr < (SELECT AVG(YEAR) FROM college` | Wrong and incomplete nested query |

Trying a schema written by hand (not from the dataset) did not work well: for a table `employees(id, name, salary, department)` and the question "average salary per department", the model produced a malformed query. This shows the model has mostly learned the patterns of the training tables and does not generalize well to new schemas.

## Getting started

Open `NL_to_sql.ipynb` in Google Colab, switch to a GPU runtime (**Runtime → Change runtime type → T4 GPU**), and run the cells in order.

```bash
pip install -r requirements.txt   # torch, datasets, tokenizers, numpy
```

The notebook:
1. Loads and formats the dataset
2. Trains the BPE tokenizer
3. Builds the encoder-decoder model
4. Trains it and saves the weights and tokenizer (to Google Drive in the Colab version)
5. Evaluates train/test loss and exact-match accuracy

Trained weights are not included in this repository.

### Inference

```python
context  = "CREATE TABLE head (age INTEGER)"
question = "How many heads of the departments are older than 56 ?"
print(to_sql(context, question))
```

Always provide the `CREATE TABLE` schema. The model was trained with it and cannot guess table or column names without it. Inputs longer than 200 tokens are not supported.

## Limitations

- Small model trained for a short time. Exact match is 45% on a small sample, and complex queries (nested queries, joins) often fail
- A low token-level loss (0.29) does not mean whole queries are right, because many SQL tokens such as `SELECT` and `FROM` are easy to predict
- The split is random and the dataset reuses the same tables, so some test schemas also appear in training. Scores on truly unseen schemas are likely lower
- Exact-match evaluation is strict: different SQL strings can mean the same thing
- No evaluation during training and no learning-rate schedule; only the final model is saved

## Roadmap

- [ ] Evaluate on a schema-disjoint split for a fairer score
- [ ] Add extra metrics (SQL parses with `sqlglot`, component match) and evaluate on more examples
- [ ] Evaluate periodically during training, keep the best checkpoint, and plot the loss curve
- [ ] Add a learning-rate schedule and train longer or with a larger model
- [ ] Compare against a fine-tuned pretrained model (for example T5-small)
- [ ] Add a small demo (Gradio)

## License

The code in this repository is released under the Apache License 2.0 (see `LICENSE`).

The dataset is **not** included and has its own terms:
- `sql-create-context` is labeled CC BY 4.0
- It is derived from Spider (CC BY-SA 4.0) and WikiSQL, whose license you should check before redistributing data or models

## Acknowledgements and citations

- b-mc2, *sql-create-context* dataset, 2023: https://huggingface.co/datasets/b-mc2/sql-create-context
- Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018
- Zhong et al., *Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning*, 2017 (WikiSQL)
- Vaswani et al., *Attention Is All You Need*, 2017
