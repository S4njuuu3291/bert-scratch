# BERT from Scratch — Bahasa Indonesia

Implementasi **BERT (Bidirectional Encoder Representations from Transformers)** dari nol menggunakan PyTorch, dengan studi kasus dan penjelasan dalam Bahasa Indonesia.

## �️ Fase 1 — WordPiece & Data Pipeline

Semua file Fase 1 telah dikelompokkan ke dalam folder [`fase-1/`](./fase-1/).

### 📖 Dokumentasi Teori (Markdown)

| File | Materi |
|------|--------|
| [`fase-1/01-wordpiece-teori.md`](./fase-1/01-wordpiece-teori.md) | **Sub-Fase 1.1** — Teori WordPiece, tokenisasi, formulasi matematika, & token spesial BERT |
| [`fase-1/02-wordpiece-preprocessing.md`](./fase-1/02-wordpiece-preprocessing.md) | **Sub-Fase 1.2** — Pre-tokenization, pemecahan karakter, & proses learning WordPiece |
| [`fase-1/03-maxmatch-tokenizer.md`](./fase-1/03-maxmatch-tokenizer.md) | **Sub-Fase 1.3** — Algoritma MaxMatch (Greedy Longest Match First) & implementasi tokenizer |
| [`fase-1/04-tensor-pipeline.md`](./fase-1/04-tensor-pipeline.md) | **Sub-Fase 1.4** — Tensor pipeline generation, shape tracking, & PyTorch tensor preparation |

### 🐍 Implementasi Kode (Python)

| File | Deskripsi |
|------|-----------|
| [`fase-1/wordpiece_train.py`](./fase-1/wordpiece_train.py) | Pelatihan vocabulary WordPiece dari korpus menggunakan likelihood score |
| [`fase-1/tokenizer.py`](./fase-1/tokenizer.py) | Class `IndoWordPieceTokenizer` — greedy max-match tokenization |
| [`fase-1/data_pipeline.py`](./fase-1/data_pipeline.py) | Class `IndoBERTDataPipeline` — konversi token ke PyTorch tensor (`input_ids`, `attention_mask`, `token_type_ids`) |

## 🚀 Cara Menjalankan

```bash
# 1. Train vocabulary WordPiece
python fase-1/wordpiece_train.py

# 2. Jalankan tokenizer demo
python fase-1/tokenizer.py

# 3. Jalankan data pipeline tensor
python fase-1/data_pipeline.py
```

## 📦 Requirements

- Python 3.8+
- PyTorch