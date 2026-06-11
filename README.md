# BERT from Scratch — Bahasa Indonesia

Implementasi **BERT (Bidirectional Encoder Representations from Transformers)** dari nol menggunakan PyTorch, dengan studi kasus dan penjelasan dalam Bahasa Indonesia.

## 📚 Daftar Isi

### 📖 Dokumentasi Teori (Markdown)

| File | Materi |
|------|--------|
| [`01-wordpiece-teori.md`](./01-wordpiece-teori.md) | **Sub-Fase 1.1** — Teori WordPiece, tokenisasi, formulasi matematika, & token spesial BERT |
| [`02-wordpiece-preprocessing.md`](./02-wordpiece-preprocessing.md) | **Sub-Fase 1.2** — Pre-tokenization, pemecahan karakter, & proses learning WordPiece |
| [`03-maxmatch-tokenizer.md`](./03-maxmatch-tokenizer.md) | **Sub-Fase 1.3** — Algoritma MaxMatch (Greedy Longest Match First) & implementasi tokenizer |
| [`04-tensor-pipeline.md`](./04-tensor-pipeline.md) | **Sub-Fase 1.4** — Tensor pipeline generation, shape tracking, & PyTorch tensor preparation |

### 🐍 Implementasi Kode (Python)

| File | Deskripsi |
|------|-----------|
| [`wordpiece_train.py`](./wordpiece_train.py) | Pelatihan vocabulary WordPiece dari korpus menggunakan likelihood score |
| [`tokenizer.py`](./tokenizer.py) | Class `IndoWordPieceTokenizer` — greedy max-match tokenization |
| [`data_pipeline.py`](./data_pipeline.py) | Class `IndoBERTDataPipeline` — konversi token ke PyTorch tensor (`input_ids`, `attention_mask`, `token_type_ids`) |

## 🚀 Cara Menjalankan

```bash
# 1. Train vocabulary WordPiece
python wordpiece_train.py

# 2. Jalankan tokenizer demo
python tokenizer.py

# 3. Jalankan data pipeline tensor
python data_pipeline.py
```

## 📦 Requirements

- Python 3.8+
- PyTorch