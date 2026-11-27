"""Sub-$10 Ìròyìn-Ray smoke. One GPU, 2,000 sentences, five train steps."""

import json
import os
import sys
import time

import ray

MODEL_ID = "facebook/nllb-200-distilled-600M"
QE_PRIMARY = "masakhane/africomet-qe-stl-1.1"
QE_FALLBACK = "McGill-NLP/ssa-comet-qe"
SRC_LANG = "eng_Latn"
TGT_LANG = "yor_Latn"
SENTENCE_LIMIT = 2000
LINE_CAP = 40000
BEAM = 1
MAX_NEW_TOKENS = 256
BLOCK_SIZE = 256
BATCH_SIZE = 16
MAX_STEPS = 5
VOCAB_SIZE = 32768
SCORE_DEADLINE_SEC = 3600

YO_MARKED = {
    "àti", "ṣùgbọ́n", "tàbí", "nítorí", "pé", "kí", "tí", "ní", "sí", "fún",
    "láti", "pẹ̀lú", "nínú", "lórí", "lábẹ́", "àwọn", "yìí", "náà", "kò", "ṣe",
    "jẹ́", "wà", "òun", "wọ́n",
}
YO_STRIPPED = {
    "ati", "sugbon", "tabi", "nitori", "pe", "ki", "ti", "ni", "si", "fun",
    "lati", "pelu", "ninu", "lori", "labe", "awon", "yii", "naa", "ko", "se",
    "je", "wa", "oun", "won",
}
EN_FUNC = {
    "the", "of", "and", "to", "a", "in", "is", "that", "for", "it", "on",
    "with", "as", "was", "at", "by", "an", "be", "this", "from", "or",
    "which", "are", "were", "have", "has", "not", "but",
}
UNDERDOT = set("ẹọṣẸỌṢ")
TONE = set("áàéèíìóòúùÁÀÉÈÍÌÓÒÚÙńǹḿŃǸḾ")
COMBINING = {"\u0300", "\u0301"}


def main():
    ray.init(address="auto", log_to_driver=False)
    deadline = time.time() + 300
    while time.time() < deadline:
        if ray.cluster_resources().get("GPU", 0) >= 1:
            break
        time.sleep(5)
    else:
        print(json.dumps({
            "event": "part_failed",
            "error_type": "NoGPU",
            "attempts": 1,
        }), flush=True)
        sys.exit(1)
    try:
        result = ray.get(run.remote())
    except Exception as exc:
        print(json.dumps({
            "event": "part_failed",
            "error_type": type(exc).__name__,
            "attempts": 1,
            "error": str(exc)[:300],
        }), flush=True)
        sys.exit(1)
    for line in result.get("logs", []):
        print(json.dumps(line, ensure_ascii=False), flush=True)
    if not result.get("ok"):
        sys.exit(1)


@ray.remote(num_gpus=1)
def run():
    t0 = time.time()
    logs = []

    def log(event):
        logs.append(event)

    try:
        _install()
        import torch
        log({
            "event": "gpu",
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
        })
        rows, revision, translate_sec, src_chars = _translate()
        _emit_counter(len(rows))
        _span(len(rows), src_chars, translate_sec)
        rows, filter_stats = _filters(rows)
        qe = _maybe_score(rows, t0, log)
        parquet_uri, manifest_uri, digest = _write_outputs(
            rows, revision, translate_sec, filter_stats, qe,
        )
        log({
            "event": "part_committed",
            "kind": "translate-partition",
            "split": "train",
            "part_id": "0",
            "model_id": MODEL_ID,
            "rows": len(rows),
            "kept_rows": filter_stats["kept_rows"],
            "sentences_per_sec": round(len(rows) / max(translate_sec, 1e-6), 3),
            "diacritic_word_rate_mean": filter_stats["diacritic_word_rate_mean"],
            "hard_fail_rate": filter_stats["hard_fail_rate"],
            "parquet_uri": parquet_uri,
            "manifest_uri": manifest_uri,
            "parquet_sha256": digest,
            "qe_model": qe["qe_model"],
            "qe_status": qe["status"],
            "seconds": round(translate_sec, 3),
        })
        spm_info = _sentencepiece(rows)
        log({
            "event": "tokenizer_done",
            "vocab_size": spm_info["vocab_size"],
            "requested_vocab_size": VOCAB_SIZE,
            "tokens_per_word": spm_info["tokens_per_word"],
            "tokens_per_word_high": spm_info["tokens_per_word_high"],
            "sentencepiece_uri": spm_info["uri"],
            "kept_lines": spm_info["kept_lines"],
        })
        train = _train(rows, spm_info)
        log({
            "event": "train_report",
            "train_loss": train["train_loss"],
            "losses": train["losses"],
            "finite": train["finite"],
            "steps": train["steps"],
            "grad_norm": train["grad_norm"],
            "checkpoint_uri": train["checkpoint_uri"],
            "wrapped_chunks": train["wrapped"],
        })
        time.sleep(60)
        return {
            "ok": bool(train["finite"]),
            "logs": logs,
            "parquet_uri": parquet_uri,
            "checkpoint_uri": train["checkpoint_uri"],
            "train_loss": train["train_loss"],
            "qe_status": qe["status"],
        }
    except Exception as exc:
        log({
            "event": "part_failed",
            "error_type": type(exc).__name__,
            "attempts": 1,
            "error": str(exc)[:300],
        })
        return {"ok": False, "logs": logs, "error": str(exc)[:300]}


def _install():
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", "torch==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    import torch
    torch_ver = torch.__version__
    constraints = "/tmp/torch-constraint.txt"
    with open(constraints, "w", encoding="utf-8") as handle:
        handle.write(f"torch=={torch_ver}\n")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir",
        "-c", constraints,
        "--extra-index-url", "https://download.pytorch.org/whl/cu121",
        "transformers==4.46.3",
        "datasets>=2.18,<3.2",
        "sentencepiece",
        "accelerate",
        "gcsfs",
        "google-cloud-storage",
        "pyarrow",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-gcp-trace",
        "pysbd",
        "huggingface_hub",
        "pandas",
        "jsonargparse",
        "pytorch-lightning>=2.2,<2.6",
        "protobuf",
        "scipy",
        "sacrebleu",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-deps", "unbabel-comet",
    ])
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "-c", constraints, "fasttext-wheel",
        ])
    except subprocess.CalledProcessError:
        pass
    import torch as torch_again
    if torch_again.__version__ != torch_ver:
        raise RuntimeError(f"torch changed from {torch_ver} to {torch_again.__version__}")
    if not torch_again.cuda.is_available():
        raise RuntimeError("cuda unavailable after install")


def _translate():
    import gc
    import unicodedata
    import pysbd
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as tlog
    tlog.set_verbosity_error()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.cuda()
    model.eval()
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    tokenizer.src_lang = SRC_LANG
    bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    segmenter = pysbd.Segmenter(language="en", clean=False)
    collected = []
    article_id = 0
    seen_title = False
    started = time.time()
    for line_index, record in enumerate(load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True,
    )):
        if line_index >= LINE_CAP or len(collected) >= SENTENCE_LIMIT:
            break
        text = record["text"]
        stripped = text.strip()
        if not stripped:
            continue
        if stripped.startswith("= ") and stripped.endswith(" ="):
            if not stripped.startswith("= ="):
                article_id += 1
                seen_title = True
            continue
        if not seen_title:
            article_id = 0
        for sent_index, sentence in enumerate(segmenter.segment(stripped)):
            sentence = sentence.strip()
            if len(sentence.split()) < 5:
                continue
            collected.append({
                "article_id": f"{article_id:06d}",
                "line_index": line_index,
                "sent_index": sent_index,
                "text_en": sentence,
                "row_type": "body",
            })
            if len(collected) >= SENTENCE_LIMIT:
                break
    if len(collected) < SENTENCE_LIMIT:
        raise RuntimeError(f"collected {len(collected)} sentences, need {SENTENCE_LIMIT}")
    pieces = []
    for row in collected:
        ntok = len(tokenizer(row["text_en"], add_special_tokens=True, truncation=False)["input_ids"])
        if ntok <= 400:
            pieces.append(row)
            continue
        words = row["text_en"].split()
        mid = max(1, len(words) // 2)
        halves = [" ".join(words[:mid]), " ".join(words[mid:])]
        for piece, half in enumerate(halves):
            if len(half.split()) < 5:
                continue
            nxt = dict(row)
            nxt["text_en"] = half
            nxt["row_type"] = "split_long"
            nxt["sent_index"] = row["sent_index"] if piece == 0 else 10000 + row["sent_index"] * 10 + piece
            pieces.append(nxt)
    src_chars = 0
    bs = 8
    with torch.inference_mode():
        for start in range(0, len(pieces), bs):
            batch = pieces[start:start + bs]
            texts = [row["text_en"] for row in batch]
            src_chars += sum(len(text) for text in texts)
            encoded = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=400,
            )
            encoded = {key: value.cuda() for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos,
                num_beams=BEAM,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, hyp in zip(batch, decoded):
                raw = hyp.strip()
                nfc = unicodedata.normalize("NFC", raw)
                row["text_yo"] = nfc
                row["nfc_changed"] = nfc != raw
    elapsed = time.time() - started
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return pieces, revision, elapsed, src_chars


def _filters(rows):
    import unicodedata
    lid = _load_fasttext()
    kept = 0
    dia_sum = 0.0
    dia_n = 0
    for row in rows:
        stats = _row_filters(row, lid, unicodedata)
        row.update(stats)
        if row["keep"]:
            kept += 1
            dia_sum += row["diacritic_word_rate"]
            dia_n += 1
    hard_fail = 1.0 - (kept / len(rows) if rows else 0.0)
    return rows, {
        "kept_rows": kept,
        "hard_fail_rate": hard_fail,
        "diacritic_word_rate_mean": (dia_sum / dia_n) if dia_n else 0.0,
        "fasttext": lid is not None,
    }


def _load_fasttext():
    try:
        import fasttext
        import urllib.request
        path = "/tmp/lid.176.bin"
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                path,
            )
        model = fasttext.load_model(path)
        labels = set(model.get_labels())
        if "__label__yo" not in labels or "__label__en" not in labels:
            return None
        return model
    except Exception:
        return None


def _row_filters(row, lid, unicodedata):
    src = row["text_en"]
    hyp = row["text_yo"]
    letters = _letter_tokens(hyp)
    yo_ratio = _func_ratio(letters, YO_MARKED, YO_STRIPPED, unicodedata)
    en_ratio = _func_ratio(letters, EN_FUNC, EN_FUNC, unicodedata)
    dia = _diacritic_word_rate(letters)
    underdot = _char_rate(hyp, UNDERDOT)
    tone = _char_rate(hyp, TONE | COMBINING)
    copy4 = _copy_4gram(src, hyp, unicodedata)
    label, prob = "", 0.0
    if lid is not None and hyp:
        pred = lid.predict(hyp.replace("\n", " ")[:1000], k=1)
        label = pred[0][0]
        prob = float(pred[1][0])
    en_len = max(len(src), 1)
    ratio = len(hyp) / en_len
    reasons = []
    if not hyp:
        reasons.append("empty")
    if hyp and not any(ch.isalpha() for ch in hyp):
        reasons.append("no_letters")
    if "\ufffd" in hyp:
        reasons.append("replacement_char")
    if ratio < 0.35 or ratio > 2.8:
        reasons.append("len_ratio")
    if _consecutive_repeat(hyp.split(), 6):
        reasons.append("token_repeat")
    if _ngram_repeat(hyp.split(), 3, 5):
        reasons.append("trigram_repeat")
    if len(letters) >= 12 and dia == 0 and en_ratio >= 0.15:
        reasons.append("no_diacritic_english")
    elif dia == 0 and letters:
        reasons.append("soft_no_diacritic")
    if label == "__label__en" and prob >= 0.80 and yo_ratio < 0.05:
        reasons.append("fasttext_en")
    if copy4 >= 0.85 and 0.75 <= ratio <= 1.25:
        reasons.append("copy_4gram")
    elif copy4 >= 0.50:
        reasons.append("soft_copy")
    if en_ratio >= 0.15 and yo_ratio <= 0.05 and "fasttext_en" not in reasons:
        reasons.append("soft_eng_func")
    hard = {
        "empty", "no_letters", "replacement_char", "len_ratio", "token_repeat",
        "trigram_repeat", "no_diacritic_english", "fasttext_en", "copy_4gram",
    }
    return {
        "char_len_en": len(src),
        "char_len_yo": len(hyp),
        "len_ratio": ratio,
        "diacritic_word_rate": dia,
        "underdot_char_rate": underdot,
        "tone_char_rate": tone,
        "yo_func_ratio": yo_ratio,
        "eng_func_ratio": en_ratio,
        "fasttext_label": label,
        "fasttext_prob": prob,
        "copy_4gram": copy4,
        "filter_reasons": reasons,
        "keep": not any(reason in hard for reason in reasons),
    }


def _letter_tokens(text):
    return [tok for tok in text.split() if any(ch.isalpha() for ch in tok)]


def _strip_marks(text, unicodedata):
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) != "Mn":
            out.append(ch)
    return "".join(out).casefold()


def _func_ratio(tokens, marked, stripped, unicodedata):
    if not tokens:
        return 0.0
    marked_cf = {item.casefold() for item in marked}
    hits = 0
    for tok in tokens:
        if tok.casefold() in marked_cf or _strip_marks(tok, unicodedata) in stripped:
            hits += 1
    return hits / len(tokens)


def _has_diacritic(token):
    return any(ch in UNDERDOT or ch in TONE or ch in COMBINING for ch in token)


def _diacritic_word_rate(tokens):
    if not tokens:
        return 0.0
    return sum(_has_diacritic(tok) for tok in tokens) / len(tokens)


def _char_rate(text, alphabet):
    letters = [ch for ch in text if ch.isalpha() or ch in COMBINING]
    if not letters:
        return 0.0
    return sum(ch in alphabet for ch in text) / len(letters)


def _consecutive_repeat(tokens, n):
    run = 1
    prev = None
    for tok in tokens:
        key = tok.casefold()
        if key == prev:
            run += 1
            if run >= n:
                return True
        else:
            prev = key
            run = 1
    return False


def _ngram_repeat(tokens, n, times):
    if len(tokens) < n:
        return False
    counts = {}
    folded = [tok.casefold() for tok in tokens]
    for i in range(len(folded) - n + 1):
        gram = tuple(folded[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= times:
            return True
    return False


def _copy_4gram(src, hyp, unicodedata):
    src_toks = _strip_marks(src, unicodedata).split()
    hyp_toks = _strip_marks(hyp, unicodedata).split()
    if len(hyp_toks) < 4:
        return 0.0
    src_grams = {
        tuple(src_toks[i:i + 4]) for i in range(len(src_toks) - 3)
    }
    hyp_grams = [tuple(hyp_toks[i:i + 4]) for i in range(len(hyp_toks) - 3)]
    return sum(gram in src_grams for gram in hyp_grams) / len(hyp_grams)


def _maybe_score(rows, t0, log):
    if time.time() - t0 >= SCORE_DEADLINE_SEC:
        for row in rows:
            row["qe_score"] = None
            row["qe_model"] = "skipped:clock"
        log({"event": "qe_skipped", "reason": "clock", "elapsed_sec": round(time.time() - t0, 1)})
        return {"status": "skipped", "qe_model": "skipped:clock", "mean": None}
    loaded = _load_qe(QE_PRIMARY)
    name = QE_PRIMARY
    if loaded is None:
        loaded = _load_qe(QE_FALLBACK)
        name = QE_FALLBACK
    if loaded is None:
        for row in rows:
            row["qe_score"] = None
            row["qe_model"] = "skipped:load"
        log({"event": "qe_skipped", "reason": "load"})
        return {"status": "skipped", "qe_model": "skipped:load", "mean": None}
    model = loaded
    data = [{"src": row["text_en"], "mt": row["text_yo"]} for row in rows]
    try:
        pred = model.predict(data, batch_size=8, gpus=1)
        scores = pred.scores if hasattr(pred, "scores") else pred["scores"]
    except Exception as exc:
        for row in rows:
            row["qe_score"] = None
            row["qe_model"] = "skipped:predict"
        log({
            "event": "qe_skipped",
            "reason": "predict",
            "qe_model": name,
            "error": str(exc)[:200],
        })
        return {"status": "skipped", "qe_model": "skipped:predict", "mean": None}
    for row, score in zip(rows, scores):
        row["qe_score"] = float(score)
        row["qe_model"] = name
    mean = sum(float(score) for score in scores) / len(scores)
    log({"event": "qe_done", "qe_model": name, "n": len(scores), "qe_mean": round(mean, 4)})
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "scored", "qe_model": name, "mean": mean}


def _load_qe(name):
    try:
        from comet import download_model, load_from_checkpoint
        path = download_model(name)
        return load_from_checkpoint(path)
    except Exception:
        return None


def _write_outputs(rows, revision, translate_sec, filter_stats, qe):
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import storage
    job_id = os.environ.get("RAY_JOB_ID", "iroyin-smoke")
    table = pa.table({
        "split": ["train"] * len(rows),
        "article_id": [row["article_id"] for row in rows],
        "line_index": pa.array([row["line_index"] for row in rows], type=pa.int32()),
        "sent_index": pa.array([row["sent_index"] for row in rows], type=pa.int32()),
        "row_type": [row["row_type"] for row in rows],
        "text_en": [row["text_en"] for row in rows],
        "text_yo": [row["text_yo"] for row in rows],
        "nfc_changed": [bool(row["nfc_changed"]) for row in rows],
        "model_id": [MODEL_ID] * len(rows),
        "model_revision": [revision] * len(rows),
        "src_lang": [SRC_LANG] * len(rows),
        "tgt_lang": [TGT_LANG] * len(rows),
        "beam": pa.array([BEAM] * len(rows), type=pa.int16()),
        "max_new_tokens": pa.array([MAX_NEW_TOKENS] * len(rows), type=pa.int16()),
        "char_len_en": pa.array([row["char_len_en"] for row in rows], type=pa.int32()),
        "char_len_yo": pa.array([row["char_len_yo"] for row in rows], type=pa.int32()),
        "len_ratio": pa.array([row["len_ratio"] for row in rows], type=pa.float32()),
        "diacritic_word_rate": pa.array([row["diacritic_word_rate"] for row in rows], type=pa.float32()),
        "underdot_char_rate": pa.array([row["underdot_char_rate"] for row in rows], type=pa.float32()),
        "tone_char_rate": pa.array([row["tone_char_rate"] for row in rows], type=pa.float32()),
        "yo_func_ratio": pa.array([row["yo_func_ratio"] for row in rows], type=pa.float32()),
        "eng_func_ratio": pa.array([row["eng_func_ratio"] for row in rows], type=pa.float32()),
        "fasttext_label": [row["fasttext_label"] for row in rows],
        "fasttext_prob": pa.array([row["fasttext_prob"] for row in rows], type=pa.float32()),
        "copy_4gram": pa.array([row["copy_4gram"] for row in rows], type=pa.float32()),
        "filter_reasons": [row["filter_reasons"] for row in rows],
        "keep": [bool(row["keep"]) for row in rows],
        "job_id": [job_id] * len(rows),
        "qe_score": pa.array(
            [None if row["qe_score"] is None else float(row["qe_score"]) for row in rows],
            type=pa.float32(),
        ),
        "qe_model": [row["qe_model"] for row in rows],
    })
    local = "/tmp/part-00000.parquet"
    pq.write_table(table, local)
    digest = hashlib.sha256(open(local, "rb").read()).hexdigest()
    bucket_name = os.environ["BUCKET"]
    prefix = "translations/v1/model=nllb-600m-smoke"
    tmp_key = f"{prefix}/split=train/_tmp/part-00000.parquet"
    final_key = f"{prefix}/split=train/part-00000.parquet"
    manifest_key = f"{prefix}/translate-partition-part-00000.json"
    client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
    bucket = client.bucket(bucket_name)
    tmp_blob = bucket.blob(tmp_key)
    tmp_blob.upload_from_filename(local, timeout=300)
    bucket.copy_blob(tmp_blob, bucket, final_key)
    parquet_uri = f"gs://{bucket_name}/{final_key}"
    manifest = {
        "kind": "translate-partition",
        "version": "v1",
        "split": "train",
        "part_id": "0",
        "status": "committed",
        "input_sentences": SENTENCE_LIMIT,
        "output_rows": len(rows),
        "kept_rows": filter_stats["kept_rows"],
        "model_id": MODEL_ID,
        "model_revision": revision,
        "src_lang": SRC_LANG,
        "tgt_lang": TGT_LANG,
        "beam": BEAM,
        "max_new_tokens": MAX_NEW_TOKENS,
        "parquet_uri": parquet_uri,
        "parquet_sha256": digest,
        "sentences_per_sec": len(rows) / max(translate_sec, 1e-6),
        "diacritic_word_rate_mean": filter_stats["diacritic_word_rate_mean"],
        "hard_fail_rate": filter_stats["hard_fail_rate"],
        "qe_model": qe["qe_model"],
        "qe_status": qe["status"],
        "qe_mean": qe["mean"],
        "fasttext_loaded": filter_stats["fasttext"],
        "error": None,
    }
    manifest_blob = bucket.blob(manifest_key)
    manifest_blob.upload_from_string(
        json.dumps(manifest, ensure_ascii=False),
        content_type="application/json",
        if_generation_match=0,
        timeout=60,
    )
    return parquet_uri, f"gs://{bucket_name}/{manifest_key}", digest


def _sentencepiece(rows):
    import sentencepiece as spm
    from google.cloud import storage
    kept = [row["text_yo"] for row in rows if row["keep"] and row["text_yo"]]
    if not kept:
        kept = [row["text_yo"] for row in rows if row["text_yo"]]
    path = "/tmp/yo.txt"
    with open(path, "w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line.replace("\n", " ") + "\n")
    prefix = "/tmp/spm-yo"
    kwargs = dict(
        input=path,
        model_prefix=prefix,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=True,
        split_digits=True,
        hard_vocab_limit=False,
        user_defined_symbols=["<eoa>"],
        normalization_rule_name="identity",
        num_threads=1,
    )
    try:
        spm.SentencePieceTrainer.train(**kwargs)
    except Exception as exc:
        if "normaliz" not in str(exc).lower():
            raise
        kwargs.pop("normalization_rule_name")
        spm.SentencePieceTrainer.train(**kwargs)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    sample = kept[:200]
    ratios = []
    for text in sample:
        words = _letter_tokens(text)
        if not words:
            continue
        ratios.append(len(sp.encode(text, out_type=int)) / len(words))
    mean = sum(ratios) / len(ratios) if ratios else None
    bucket_name = os.environ["BUCKET"]
    key = "ray-results/gpt2-yo-smoke-10/sentencepiece.model"
    client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
    blob = client.bucket(bucket_name).blob(key)
    blob.upload_from_filename(prefix + ".model", timeout=120)
    return {
        "processor": sp,
        "vocab_size": sp.vocab_size(),
        "tokens_per_word": None if mean is None else round(mean, 4),
        "tokens_per_word_high": bool(mean is not None and mean > 4),
        "uri": f"gs://{bucket_name}/{key}",
        "kept_lines": len(kept),
        "model_path": prefix + ".model",
    }


def _train(rows, spm_info):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    sp = spm_info["processor"]
    eoa = sp.piece_to_id("<eoa>")
    ids = []
    prev = None
    source_rows = [row for row in rows if row["keep"] and row["text_yo"]]
    if not source_rows:
        source_rows = [row for row in rows if row["text_yo"]]
    for row in source_rows:
        if prev is not None and row["article_id"] != prev and eoa >= 0:
            ids.append(eoa)
        prev = row["article_id"]
        ids.extend(sp.encode(row["text_yo"], out_type=int))
    if eoa >= 0:
        ids.append(eoa)
    chunks = []
    for start in range(0, len(ids) - BLOCK_SIZE + 1, BLOCK_SIZE):
        chunks.append(ids[start:start + BLOCK_SIZE])
    if not chunks:
        raise RuntimeError(f"no full blocks from {len(ids)} tokens")
    wrapped = False
    need = BATCH_SIZE * MAX_STEPS
    if len(chunks) < need:
        wrapped = True
        base = list(chunks)
        while len(chunks) < need:
            chunks.extend(base)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vocab = sp.vocab_size()
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=vocab,
        n_positions=BLOCK_SIZE,
        n_embd=768,
        n_layer=12,
        n_head=12,
    ))
    if hasattr(model, "loss_type"):
        try:
            model.loss_type = "ForCausalLM"
        except Exception:
            pass
    model.gradient_checkpointing_enable()
    model.cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    losses = []
    grad_norm = None
    for step in range(MAX_STEPS):
        batch = chunks[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        input_ids = torch.tensor(batch, dtype=torch.long, device="cuda")
        if int(input_ids.max()) >= vocab:
            raise RuntimeError("token id outside vocab")
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    finite = all(loss == loss and abs(loss) != float("inf") for loss in losses)
    model.cpu()
    ckpt = {
        "epoch": 0,
        "step": MAX_STEPS,
        "train_loss": losses[-1],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _cpu_state(opt.state_dict()),
        "sentencepiece_gcs_uri": spm_info["uri"],
        "vocab_size": vocab,
    }
    local = "/tmp/checkpoint.pt"
    torch.save(ckpt, local)
    from google.cloud import storage
    bucket_name = os.environ["BUCKET"]
    key = "ray-results/gpt2-yo-smoke-10/checkpoint.pt"
    client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
    client.bucket(bucket_name).blob(key).upload_from_filename(local, timeout=600)
    return {
        "train_loss": losses[-1],
        "losses": [round(loss, 6) for loss in losses],
        "finite": finite,
        "steps": MAX_STEPS,
        "grad_norm": grad_norm,
        "checkpoint_uri": f"gs://{bucket_name}/{key}",
        "wrapped": wrapped,
    }


def _cpu_state(value):
    import torch
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _emit_counter(n):
    try:
        from ray.util.metrics import Counter
        counter = Counter(
            "iroyin_translation_sentences_total",
            description="Sentences translated",
            tag_keys=("job_name", "model_id", "split"),
        )
        counter.set_default_tags({
            "job_name": "iroyin-smoke",
            "model_id": MODEL_ID,
            "split": "train",
        })
        counter.inc(n)
    except Exception:
        pass


def _span(n, chars, elapsed):
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        resource = Resource.create({"service.name": "iroyin-ray-smoke"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter(
            project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
        )))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("iroyin")
        with tracer.start_as_current_span("translate.partition") as span:
            span.set_attribute("part_id", "0")
            span.set_attribute("model_id", MODEL_ID)
            span.set_attribute("beam", BEAM)
            span.set_attribute("sentences", n)
            span.set_attribute("chars", chars)
            span.set_attribute("latency_ms", int(elapsed * 1000))
        provider.force_flush(timeout_millis=30000)
    except Exception:
        pass
