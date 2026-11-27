# Ìròyìn-Ray smoke

Sub-$10 GKE Autopilot smoke of the Ìròyìn-Ray pipeline. One successful Ray job translated 2,000 English Wikitext-103-raw body sentences into Yoruba with NLLB-200 distilled 600M, filtered them, fit a SentencePiece Unigram tokenizer, and trained a from-scratch GPT-2 for 5 steps. The cluster was deleted the same day.

This repository is private. The plan treats the corpus as internal research, not a public dataset release. NLLB weights are CC-BY-NC. Wikitext comes from Wikipedia (CC BY-SA), so the translations are a derivative work.

## What the successful run produced

Job `iroyin-smoke` on Ray cluster `iroyin-smoke-p74fp` (Kubernetes job `iroyin-smoke-jprhb`).

| | |
|---|---|
| Window | 13:07:22–13:24:24 Africa/Lagos, 27 November 2026 (17 minutes) |
| GPU | NVIDIA L4 on a spot `g2-standard-4`, CUDA 12.1 |
| Model | `facebook/nllb-200-distilled-600M` @ `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| Direction | `eng_Latn` → `yor_Latn`, beam 1, `max_new_tokens` 256 |
| Input | 2,000 body sentences, 24 articles, streaming stop (the 1.8M-line file was not indexed) |
| Throughput | 4.03 sentences/s (translate span 496 s) |
| Rows written | 2,000 |
| Rows kept | 1,885 |
| Hard-fail rate | 0.0575 |
| Mean diacritic-word rate, kept | 0.800 |
| Mean length ratio, all rows | 1.16 (English 146 chars, Yoruba 165) |
| Quality estimation | skipped (`skipped:load`). Every `qe_score` is null |
| fastText lid.176 | loaded |
| SentencePiece | one Unigram fit, vocab 3,659 of 32,768 requested, 1.75 tokens/word, no refit |
| Train loss | 8.3643, 7.4953, 7.4563, 7.1075, **6.9424**. All finite. Grad norm 1.74 |

The Parquet hash matches the commit manifest: `1eacdd9e562631c8be0d20460e415f8dd5778843c1cdb1a3b6f134a4a4147969`.

### Why 115 rows were dropped

Hard filters, and a row can carry more than one:

| Reason combination | Rows |
|---|---|
| length ratio + token repeat + trigram repeat | 36 |
| token repeat + trigram repeat | 34 |
| trigram repeat | 21 |
| length ratio + trigram repeat | 8 |
| length ratio + English function words + trigram repeat | 6 |
| length ratio | 3 |
| no letters | 2 |
| token repeat | 2 |
| fastText English | 1 |
| length ratio + missing diacritics + token repeat + trigram repeat | 1 |
| English function words + trigram repeat | 1 |

Nine kept rows carry a soft flag only: 5 copy-4-gram, 3 English function-word ratio, 1 missing diacritics. fastText’s top labels on all 2,000 rows were Yoruba 1,509, Irish 243, English 44. Only the one English row above met the hard English-copy rule.

Quality estimation did not run. `masakhane/africomet-qe-stl-1.1` and `McGill-NLP/ssa-comet-qe` each failed their first load. The 60-minute clock was not the cause: translation finished in about 8 minutes and the whole job in 17. The manifest was already committed with `if_generation_match=0`, so it was not rewritten.

The tokenizer requested 32,768 pieces. With `hard_vocab_limit` off, 1,885 lines only support 3,659. Tokens per word stayed at 1.75, under the refit bar, so the model was not refit.

### Two kept rows

A clean short line:

- English: `Barker died in 1973 .`
- Yoruba: `Barker kú ní ọdún 1973 .`

A noisy Wikitext fragment that still passed the hard filters (`The gun weighed 12 @.` became `Ogun náà wúwo méjìlá.`). Length and repetition filters do not catch a meaning error, and no QE score was attached.

## Runs that did not produce data

1. The first Autopilot cluster scaled a spot `g2-standard-8` for a pod that was supposed to fit a `g2-standard-4`. That cluster was deleted before any translation. The ComputeClass was then pinned to `g2-standard-4`, then `n1-standard-4`, and the worker request was lowered to 2 CPU and 10 GiB.
2. The first RayJob on the replacement cluster exited in a few seconds: `rayproject/ray:2.54.0-py311-gpu` does not ship PyTorch. The job was resubmitted with `torch==2.5.1` from the cu121 index installed inside the worker. That resubmit is the successful run above.

## Observability, then the cluster was deleted

- Cloud Logging: one `part_committed` JSON line on pod `iroyin-smoke-p74fp-head-n7s8z` at 13:24:15 Africa/Lagos, with `k8s-pod/ray_io/is-ray-node=yes`.
- Managed Prometheus: `prometheus.googleapis.com/ray_finished_jobs_total/counter` = 8 over 13:08–13:25 Africa/Lagos. The custom `iroyin_translation_sentences_total` counter was not visible, so the Ray series is the proof.
- Cloud Trace: span `translate.partition`, trace `a5104585a2a74e7cc43cce564d4e5158`, 496,050 ms, 2,000 sentences, 292,882 characters.

`gcloud container clusters list` and `gcloud compute instances list` on project `iroyin-ray-smoke` are empty. The $10 budget, the bucket, and the GCS-only service account are still in place. An alert does not stop spend.

## What is in this repo

| Path | Bytes | sha256 |
|---|---|---|
| `data/part-00000.parquet` | 443,567 | `1eacdd9e562631c8be0d20460e415f8dd5778843c1cdb1a3b6f134a4a4147969` |
| `data/translate-partition-part-00000.json` | 805 | `a8f17b62a3f3073f8e95d7d8dbaa975207a5ca9cbef1191a3e92c4b4c4258340` |
| `data/sentencepiece.model` | 55,297 | `36bf4176cd23f494c7cdfd7128024452c334362f86ae26c2f33dbe535de14996` |
| `data/checkpoint.json` | pointer for the 1,056,927,566-byte checkpoint, sha256 `98177e990002fcd1a577816b33b9d9cf2c5170e9ed073874940e6045ef834c18` | |
| `data/run.json` | the same figures, as JSON | |
| `plan/iroyin-ray-plan.md` | the build contract | |
| `smoke/` | the job script and the manifests that were applied | |

`checkpoint.pt` is one fp32 GPT-2 checkpoint after 5 AdamW steps. GitHub rejects a git blob over 100 MB, so the file is the `smoke-2026-11-27` release asset, not a blob in the tree. `data/checkpoint.json` records its size and sha256. The same bytes are still in the bucket.

The same objects are still in the bucket:

- `gs://iroyin-ray-smoke-iroyin-ray/translations/v1/model=nllb-600m-smoke/split=train/part-00000.parquet`
- `gs://iroyin-ray-smoke-iroyin-ray/translations/v1/model=nllb-600m-smoke/translate-partition-part-00000.json`
- `gs://iroyin-ray-smoke-iroyin-ray/ray-results/gpt2-yo-smoke-10/sentencepiece.model`
- `gs://iroyin-ray-smoke-iroyin-ray/ray-results/gpt2-yo-smoke-10/checkpoint.pt`

A staging copy of the Parquet remains at `.../split=train/_tmp/part-00000.parquet` (443,567 bytes). The committed object is the one in this repo.
