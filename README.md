# Banking RAG Eval

**A LangChain RAG pipeline over EU banking regulation (PSD2, deposit guarantees, GDPR), with Qdrant as the vector store and a code-graded eval suite that works as a prompt regression test.**

A 3B model on a gaming GPU answers 94% of 33 regulation questions correctly from retrieved articles, and refuses all 14 the articles can't answer. A plausible "be more helpful" prompt edit raises accuracy to 97% and makes the model answer every one of those 14. In 8 of the 14 answers it states something false or invented. The regression test fails that edit. A dashboard that tracked accuracy alone would have shipped it. [Results →](#results)

The same eval also found this setup's biggest weakness. Ask in everyday words instead of the directive's own words, and the local model refuses 6 of 8 questions it could have answered ([plain-language questions](#plain-language-questions-the-weak-spot)).

![What the project does, in plain words: find the relevant EU law, answer only from it with a source, and re-check 71 known questions after every change. A real answer cites GDPR Article 33.](docs/figures/overview.png)

## At a glance

| | |
|---|---|
| **What it does** | Answers questions about three EU acts from their articles, citing each source as `[n]`. When the retrieved articles don't hold the answer, it replies with one fixed sentence: *I cannot answer this from the provided documents.* |
| **Corpus** | Every article of PSD2 (117), the Deposit Guarantee Schemes Directive (23) and GDPR (99): 239 articles, 607 chunks ([data/README.md](data/README.md)) |
| **Retrieval** | Qdrant in local mode (no server, no account), one collection holding both a dense vector (bge-small-en-v1.5) and a BM25 sparse vector per chunk; hybrid search fuses the two inside Qdrant |
| **Eval** | 47 questions in the gated main set, plus 16 held out and 8 in everyday wording. Graded by code: facts by regular expression against known answers, citations by article id, refusals by the exact sentence. Four kinds: facts, false premises, questions the corpus can't answer, prompt injections |
| **Regression gate** | Five metrics compared with the accepted baseline, with a margin of one question each: `pytest -m live` fails a prompt change that makes any of them worse |
| **Fine-tuning** | A LoRA experiment on Qwen3-0.6B, trained in 7 minutes on the same GPU and graded by the same suite. The adapter learned the answer contract but refuses too often, so it isn't used ([results](#fine-tuning-experiment-lora-on-a-06b-model)) |
| **Trustworthy data** | 222 offline tests, including: every expected fact appears in its gold article, no trap question's answer appears anywhere in the corpus, and the graders accept a hand-written right answer and reject a wrong one for every answerable question, and no fine-tuning example comes from an article the evals use |
| **Repeatable** | Greedy decoding, a fixed seed, and the chat template's date pinned. Each prompt run two or three times on the same questions (seven runs in all) gave the same answers word for word |
| **Per query** | Latency, retrieval and generation time, tokens in and out, and cost, saved with every answer |
| **Stack** | Python 3.13 · langchain-core 1.7 with the qdrant, openai, anthropic and text-splitters integrations · Qdrant client 1.20 · FastEmbed 0.9 · llama.cpp · pytest · for the optional LoRA experiment: PyTorch 2.11, transformers 5.22, PEFT 0.21 |
| **Status** | Local model (SmolLM3 3B on llama.cpp) measured, and the LoRA experiment run. The Claude Haiku 5.5 provider is written but has no recorded runs yet ([limitations](#known-limitations)) |

## Results

Everything below was measured on one PC: RTX 3060 Ti 8 GB, Ryzen 5 5600X, Windows 11, llama.cpp build 12110 (CUDA), SmolLM3 3B quantised to Q4_K_M, all layers on the GPU. Every number comes from a run file in [results/](results/), which records each question, answer, retrieved chunk, grade, timing and token count.

### Three prompts on the main set

47 questions: 30 facts, 3 false premises, 12 questions the corpus can't answer, 2 prompt injections.

| Prompt | Answer accuracy (95% interval) | Cites the right article | Refuses when it should | Wrongly refuses | False premises | Regression gate |
|---|---|---|---|---|---|---|
| [grounded-v1](prompts/grounded-v1.yaml) | 85% (69–93%) | 88% | 86% (12 of 14) | 9% (3 of 33) | 0 of 3 | First baseline |
| [grounded-v2](prompts/grounded-v2.yaml) (shipped) | **94%** (80–98%) | 85% | **100%** (14 of 14) | 6% (2 of 33) | 1 of 3 | Passed against v1, now the baseline |
| [helpful-v1b](prompts/helpful-v1b.yaml) | 97% (85–99%) | 79% | **0%** (0 of 14) | 0% | 3 of 3 | **Failed** on refusals and citations |

- **helpful-v1b is v1 with one rule changed.** "If the sources do not contain the answer, reply with exactly this sentence" became "always give the most helpful answer you can", and the rule about ignoring instructions inside the question was dropped. Accuracy, wrong refusals and false premises all improve. Every question it should have refused gets an answer instead. The gate output:

  ```text
  E  AssertionError: citation_accuracy    baseline 0.848  now 0.788  (allowed drop 0.031)  REGRESSION
  E  AssertionError: refusal_rate         baseline 1.000  now 0.000  (allowed drop 0.072)  REGRESSION
  FAILED tests/test_prompt_regression.py::test_metric_within_margin_of_baseline[citation_accuracy]
  FAILED tests/test_prompt_regression.py::test_metric_within_margin_of_baseline[refusal_rate]
  2 failed, 3 passed
  ```

  ![The same story in plain words: answer accuracy rose from 94% to 97%, but saying "I don't know" when it should fell from 100% to 0%, and the automatic test failed the change.](docs/figures/test-in-action.png)

- **I read all 14 of helpful-v1b's answers to the questions it should have refused.** 8 contain a false or invented claim. 2 pad a decline with unrelated source text. 4 decline cleanly, but in their own words. The worst four:
  - **The injection worked.** Asked to ignore the rules, it gave the Basel III CET1 minimum (4.5%) from its own training.
  - **Wrong law, confidently applied.** It took PSD2's five-year record-keeping rule for payment institutions and presented it as the anti-money-laundering retention period for banks.
  - **An invented article.** It quoted a Consumer Credit Directive "Article 5(1)" letting consumers withdraw "at any time". That act isn't in the corpus, and the claim is wrong.
  - **A source that isn't there.** It said the sources mention the ECB's marginal lending rate, then gave a figure for it.
- **v2 changed two things** after reading v1's failures. It says plainly that a question with a false premise is still answerable, and it puts that rule before the refusal rule. It also asks for answers in the model's own words instead of copied paragraphs. The first change fixed one of three false premises, and fixed two trap questions v1 had answered. The second changed nothing: the average answer stayed at 528 characters.
- **The refusal rule causes the false-premise failures.** helpful-v1b, which has no refusal rule, corrects all three false premises. v1 and v2 mostly refuse them, apparently reading "the question is wrong" as "the answer isn't in the sources".

### Held-out questions

v2 was written while looking at v1's answers on the main set, so its main-set score is optimistic. The [held-out set](evals/holdout.yaml) was written after v2 was frozen: 10 facts, 1 false premise, 5 questions the corpus can't answer. Each prompt ran on it once.

| Prompt | Answer accuracy | Cites the right article | Refuses when it should |
|---|---|---|---|
| grounded-v1 | 91% (10 of 11) | 91% (10 of 11) | 80% (4 of 5) |
| grounded-v2 | 82% (9 of 11) | 73% (8 of 11) | 100% (5 of 5) |

v2's better refusals held up on new questions. Its accuracy gain didn't, and it cited worse. On the false premise ("EUR 125 000 of initial capital, right?") both prompts agreed with the wrong claim. Across all 63 main and held-out questions, v2 gets more answers right (40 of 44 against 38) and refuses every trap (19 of 19 against 16). It also cites the right article less often (36 of 44 against 39). In a bank, a made-up answer costs more than a missing citation, so v2 ships. Its citation format is the next thing to fix: it sometimes writes "Article 6" in prose instead of `[1]`.

### Plain-language questions: the weak spot

The main set's questions were written while reading the articles, so they share the directives' vocabulary. The [plain set](evals/plain.yaml) asks 8 main-set facts the way a customer would, for example "Someone stole my card and spent 400 euros. How much of that do I have to pay?" No prompt or setting was tuned on it.

| | grounded-v1 | grounded-v2 | helpful-v1b |
|---|---|---|---|
| Answer accuracy | 3 of 8 | 2 of 8 | 5 of 8 |
| Wrongly refused | 4 of 8 | 6 of 8 | 0 of 8 |

In 4 of v2's 6 refusals, the right article was already among the five sources. The model didn't connect "someone stole my card" with "lost or stolen payment instrument" and refused. The rule that stops invented answers also stops this 3B model answering questions it could answer. The gate didn't catch this because the main set has no everyday wording. Extending the gate to the plain set is the next step.

### Fine-tuning experiment: LoRA on a 0.6B model

Can fine-tuning teach the answer contract to a model too small to follow it from the prompt alone? [The notebook](notebooks/lora_experiment.ipynb) trains a LoRA adapter (rank 16 on every attention and MLP projection, 10.1M trainable parameters, 1.7% of the model) on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). It then grades the model without and with the adapter on all three eval sets, with the same retrieval, prompt (grounded-v2) and graders as above.

**Training data that can't leak into the eval.** SmolLM3 wrote one question, answer and supporting quote for each of 317 chunks, taken only from articles that no eval question is about. Code kept 224, rejecting a question unless its quote appears word for word in the chunk and it doesn't mention "the passage" or an article number. Each kept question became an answer example (its chunk among the five retrieved sources, target = answer plus `[n]`). For 60% of them it also became a refusal example (every chunk of that article removed from the sources, target = the refusal sentence). That gives 342 examples: 302 for training and 40 for validation from separate articles. Tests check that no example comes from an eval article and no training question repeats an eval question ([bankrag/finetune_data.py](bankrag/finetune_data.py), [tests/test_finetune_data.py](tests/test_finetune_data.py)).

**Training:** 2 epochs, 76 optimizer steps, 7 minutes, 1.9 GiB peak GPU memory. The loss is computed only on the answer tokens, and only those positions are scored over the 152,000-token vocabulary. A first test that scored every position, without gradient checkpointing active, peaked at 9.5 GiB, more than the card holds. Validation loss went 0.858 → 0.212 → 0.209.

| Main set (47 questions) | Qwen3-0.6B | + LoRA |
|---|---|---|
| Refuses when it should | 50% | **100%** |
| Answers with a valid citation (of those not refused) | 91% | **100%** |
| Tokens per answer | 83 | 23 |
| Wrongly refuses | 15% | **36%** |
| Answer accuracy | 76% | **58%** |

- **The adapter learned the contract.** It refused all 17 out-of-corpus questions across the main and held-out sets (the base model refused 9) and both prompt injections (base: 1). Every refusal uses the exact sentence, where the base model wrote variants such as "I cannot answer *this question* from…".
- **It also learned to refuse too much.** Wrong refusals rose from 9% to 55% on the held-out set and from 25% to 88% on the plain-language set. Answer accuracy fell everywhere (held out: 64% → 45%; plain: 38% → 12%). It would fail the regression gate, so it isn't used.
- **Likely causes:** refusals were 38% of the training data, each built from a question that was otherwise answerable, and the training questions share their passage's wording. Training loss kept falling in the second epoch (0.250 → 0.090) while validation loss didn't, so that epoch mostly memorised. The next run uses fewer refusal examples (15–20%), questions rephrased into everyday wording, and one epoch, with the gate as the stopping rule.

This is the same trade-off the prompt experiments showed: a strict refusal rule, whether in the prompt or trained into the weights, swaps invented answers for missed ones.

### Retrieval

Whether the right article is among the chunks returned, with no model involved. The main set has 33 answerable questions; the plain set has 8.

| Search | Main: hit@1 | Main: hit@5 | Main: MRR | Plain: hit@1 | Plain: hit@5 | Plain: MRR |
|---|---|---|---|---|---|---|
| Dense (bge-small) | 88% | 100% | 0.93 | 5 of 8 | 7 of 8 | 0.74 |
| BM25 (keywords) | 91% | 100% | 0.94 | 1 of 8 | 2 of 8 | 0.18 |
| **Hybrid (shipped)** | **97%** | **100%** | **0.985** | 3 of 8 | 6 of 8 | 0.54 |

- **Hybrid is best when the question uses the law's words, and worse than dense alone when it doesn't.** Keyword matching finds almost nothing in everyday wording, and fusing it in pushes dense hits down. The default stays hybrid because it was chosen on the main set. Choosing again on 8 plain questions would use up the only independent check. A larger plain set should decide between dense and a weighted fusion.
- **Article headers on chunks** ("PSD2 Article 74: Payer's liability…") made no consistent difference. Without them, dense search found 8 of 8 plain questions in the top 5 instead of 7, and hybrid was the same on the main set. With 8 and 33 questions, these differences are one or two questions.
- **A similarity threshold could refuse before calling the model.** The best dense match scored 0.745–0.886 for answerable questions and 0.646–0.750 for unanswerable ones. A threshold near 0.745 would separate them almost perfectly, but only because it would be tuned on the test questions themselves, and the plain-language questions score lower. So the pipeline doesn't use one.

`python -m bankrag retrieval` reproduces this table ([results/retrieval.json](results/retrieval.json)).

### Latency, tokens and cost

Per query, grounded-v2 on the main set:

| | |
|---|---|
| Retrieval (embed the question on the CPU, hybrid search in Qdrant) | 23 ms on average |
| Generation (SmolLM3 3B on the GPU) | 0.78 s on average |
| End to end, median / 95th percentile | 0.70–0.73 s / 1.5–2.2 s across three identical runs |
| Tokens | 1,097 in (five chunks and the prompt), 78 out |
| Cost | No API cost. GPU electricity isn't measured here; [LocalLLMPerformanceLab](https://github.com/Renis032/LocalLLMPerformanceLab) does that |
| Building the index | 607 chunks embedded twice (dense and BM25) in about 30 s on the CPU, once per process |

A refusal takes about 0.25 s, because the model writes only 10 tokens. At Claude Haiku 5.5's list price ($0.10 per million input tokens, $0.50 per million output), the same token counts would cost about $0.00015 per query, or $0.007 for the main set. That's an estimate, not a measurement: Claude counts tokens differently and may add thinking tokens.

## How it works

```mermaid
flowchart LR
    A["EU Publications Office<br/>PSD2, DGSD, GDPR"] -->|"build-corpus"| B["239 articles<br/>data/corpus.jsonl"]
    B -->|"RecursiveCharacterTextSplitter<br/>1,000 chars + article header"| C["607 chunks"]
    C -->|"FastEmbed: bge-small + BM25"| D[("Qdrant<br/>in-memory")]
    Q["Question"] --> E["Hybrid search, top 5"]
    D --> E
    E --> F["Numbered sources [1]..[5]"]
    F --> G["ChatPromptTemplate piped into the chat model"]
    G --> H["Answer"]
    H --> I["Code graders"]
    I --> J["Run file + regression gate"]
```

- **Corpus** ([bankrag/corpus.py](bankrag/corpus.py)): EUR-Lex answers scripts with a bot check, so the Official Journal text comes from the Publications Office instead, as XHTML. Each act is split into one record per article. Point labels such as "(a)" sit in a separate table cell from their text, so they are joined back to it.
- **Index** ([bankrag/index.py](bankrag/index.py)): LangChain's `QdrantVectorStore` over one Qdrant collection holding both vector types, so dense, sparse and hybrid search run against the same index. The dense model sits behind LangChain's `Embeddings` interface through a small FastEmbed adapter (ONNX on the CPU, no PyTorch).
- **Chain** ([bankrag/rag.py](bankrag/rag.py)): the prompt is a YAML file in [prompts/](prompts/), turned into a `ChatPromptTemplate` and piped into the chat model (`prompt | llm`). Token counts come from LangChain's `usage_metadata`, so the same code records them for llama.cpp and Claude.
- **Local model** ([bankrag/llamaserver.py](bankrag/llamaserver.py)): the eval starts and stops `llama-server` itself and talks to it through `ChatOpenAI`, since the server speaks the OpenAI API. SmolLM3's chat template writes today's date into every prompt, which would make tomorrow's run differ from today's. On first use the template is fetched from the server and the date call replaced with a fixed date. LocalLLMPerformanceLab found this the hard way ([finding 11](https://github.com/Renis032/LocalLLMPerformanceLab/blob/main/docs/measurement-findings.md#11-a-chat-template-puts-todays-date-in-every-prompt)).
- **Graders** ([bankrag/grading.py](bankrag/grading.py)): text is normalised first ("EUR 100 000", "€100,000" and "100.000" all become "100000"). An answer is correct when every `expect` pattern matches and no `reject` pattern does. A citation counts when one of the cited numbers points at a gold article. A refusal counts only as the exact sentence: the application around the model can rely on a fixed sentence, and keeping to it is part of what's tested.
- **Gate** ([evals/gates.yaml](evals/gates.yaml), [evals/baselines.json](evals/baselines.json)): five metrics, each allowed to get worse by one question's worth and no more. Retrieval is allowed nothing, because it has no randomness. `--accept` records a run as the new baseline, the way a snapshot test does.

## Quick start

The offline tests need only [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.13 and the dependencies itself. On Windows, open a new terminal after installing uv so that it is on the PATH, and run the commands one at a time (Windows PowerShell 5.1 has no `&&`). The first test run downloads the two embedding models (about 65 MB) into `.cache/`.

```bash
git clone https://github.com/Renis032/BankingRAGEval.git
cd BankingRAGEval
uv sync
uv run pytest
```

To run the model, copy `.env.example` to `.env` and point it at a `llama-server` build and a GGUF model. These results used [SmolLM3-3B Q4_K_M](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF). Then:

```bash
uv run python -m bankrag ask "Within what time must a bank notify the supervisory authority of a personal data breach?"
```

```text
According to [1] GDPR Article 33, the controller must notify the supervisory authority of a personal data breach
without undue delay and, where feasible, not later than 72 hours after becoming aware of it.

  [1] GDPR-33 (chunk 0, score 1.0)
  [2] GDPR-33 (chunk 1, score 0.6667)
  [3] GDPR-34 (chunk 1, score 0.4)
  [4] GDPR-51 (chunk 1, score 0.25)
  [5] GDPR-58 (chunk 1, score 0.25)

  retrieval 19 ms, generation 548 ms, 988 tokens in, 44 out, $0.000000
```

| Command | What it does |
|---|---|
| `uv run pytest` | 222 offline tests: corpus, eval data, graders, retrieval, fine-tuning data. No model, about 40 s |
| `uv run pytest -m live --provider local` | The prompt regression test: runs the main set on the model and checks each gated metric against the baseline |
| `uv run pytest -m live --provider local --prompt helpful-v1b` | The same, on the prompt that should fail |
| `uv run python -m bankrag eval [--prompt ID] [--set main\|holdout\|plain]` | Runs an eval set and saves every answer to `results/runs/` |
| `uv run python -m bankrag eval --check` | The same comparison as the live test, as a command that exits 1 on a regression |
| `uv run python -m bankrag eval --accept` | Accepts the run as the provider's new baseline |
| `uv run python -m bankrag retrieval` | Scores dense, BM25 and hybrid search, with and without chunk headers, on all three sets |
| `uv run python -m bankrag report` | Prints the latest run of each prompt and set as Markdown tables |
| `uv run python -m bankrag build-corpus [--refresh]` | Downloads the three acts again and rebuilds `data/corpus.jsonl` |
| `uv run python -m bankrag finetune-data [--regenerate]` | Builds the LoRA training data in `data/finetune/` (the generation step needs llama-server) |

For Claude, add `ANTHROPIC_API_KEY` to `.env` and use `--provider claude-haiku`. It has no baseline yet, so run `eval --provider claude-haiku --accept` once before `--check` or the live test.

The LoRA experiment needs PyTorch with CUDA (about 3 GB) and an NVIDIA GPU. Its packages are kept out of the default install:

```bash
uv sync --group finetune
```

```bash
uv run --group finetune jupyter nbconvert --to notebook --execute --inplace notebooks/lora_experiment.ipynb
```

## Project layout

```text
bankrag/            corpus download and parsing, index, chain, providers, graders, eval runner, CLI
prompts/            prompt versions as YAML: grounded-v1, grounded-v2 (shipped), helpful-v1b (the regression demo)
evals/              questions.yaml (main, gated), holdout.yaml, plain.yaml, gates.yaml, baselines.json
data/corpus.jsonl   the 239 articles, with their source and licence in data/README.md
data/finetune/      the LoRA experiment's generated questions and training examples
notebooks/          lora_experiment.ipynb, with its outputs and findings
results/runs/       every eval run: one .jsonl line per answer, and a .json summary
results/lora/       the LoRA training record (settings, losses, time, memory)
results/retrieval.json
docs/figures/       screenshots of the plain-language overview
tests/              offline tests, plus test_prompt_regression.py (marked live)
```

## Known limitations

- **Small eval sets.** One question is 3 points of answer accuracy on the main set and 7 points of refusal rate. The 95% intervals are wide (v2: 80–98%), so a difference smaller than about two questions shouldn't be read as real.
- **v2 was tuned on the main set,** so the held-out set is the fairer score, and it has only 16 questions.
- **Everyday wording is barely covered:** 8 questions, and they show the largest weakness found. A bigger plain-language set, in the gate, comes next.
- **Graders check the key fact, not the whole answer.** An answer with the right deadline and a wrong extra claim still passes. In the other direction, declining in other words counts as a failure, on purpose.
- **SmolLM3 ignores the length rule.** It often pastes whole paragraphs. Length isn't graded.
- **No Claude results yet.** The `claude-haiku` provider builds a valid request (Haiku 5.5, low effort, no sampling parameters, which that model rejects), checked offline, but it hasn't been run against the API. Its runs won't repeat exactly, because Haiku 5.5 takes no temperature or seed, so its gate margins should be set from a few repeated runs.
- **The LoRA experiment is one run**, with one seed and one data mix, on synthetic questions written by a 3B model. It shows what this recipe does, not what fine-tuning can do in general. The adapter isn't committed; the notebook recreates it in about 7 minutes.
- **The corpus is the acts as first published**, without amendments. It's test material, not current law.
- **The index lives in memory** and is rebuilt in about 30 s per process. That's fine at 607 chunks. A larger corpus would use Qdrant on disk or as a server.
- **Tested on Windows only,** with one GPU. Other OpenAI-compatible local servers (Ollama, LM Studio) should work through `BANKRAG_LOCAL_URL`, but haven't been tried.

## In plain words

For readers outside AI: the measured results, and where the project shows each skill an AI prompt engineering role asks for.

![Measured results in plain words (94% answered correctly, 14 of 14 correct refusals, 0.7 s per answer, no running cost), and a table mapping LangChain, RAG with a vector database, prompt engineering, prompt evaluation, regression testing, fine-tuning and cost tracking to where each appears in the project.](docs/figures/results.png)

## Licence

The code and eval data are under the [MIT License](LICENSE). The texts in `data/corpus.jsonl` are EU legislation, © European Union, reused under Commission Decision 2011/833/EU ([data/README.md](data/README.md)).
