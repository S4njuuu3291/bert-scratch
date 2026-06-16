import torch
import torch.nn as nn
import sys
sys.path.insert(0, "../fase-1")

from tokenizer import IndoWordPieceTokenizer
from data_pipeline import IndoBERTDataPipeline
from bert_embeddings import BERTEmbeddings

# =====================================================================
# REKAPITULASI KOMPONEN DARI FASE SEBELUMNYA (UNTUK INTEGRASI)
# =====================================================================

# 1. Kamus Kosakata Tiruan IndoBERT
dummy_vocab = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "budi": 5, "makan": 6, "nasi": 7, "saya": 8, "ayam": 9, "##kan": 10
}

tokenizer = IndoWordPieceTokenizer(dummy_vocab)
pipeline = IndoBERTDataPipeline(tokenizer,dummy_vocab,max_length=8)

# =====================================================================
# EKSEKUSI PIPELINE & INKORPORASI MODEL
# =====================================================================

# Simulasikan data input (Batch Size, B = 2)
kalimat_input = [
    "budi makankan nasi",
    "saya makan ayam"
]

# Generate tensor input koordinat dari Fase 1
# Bentuk Tensor: (B, N) -> (2, 8)
t_input_ids, _, t_token_type_ids = pipeline.create_batch(kalimat_input)

print(t_input_ids)
print(t_token_type_ids)

# Inisialisasi Lapisan Embedding Primitif IndoBERT
# Kita set dimensi tersembunyi (D) = 768 sesuai standar BERT-Base
VOCAB_SIZE = len(dummy_vocab)  # V = 11
HIDDEN_DIM = 768               # D = 768
MAX_POSITION = 512             # M = 512

embedding_layer = BERTEmbeddings(
    vocab_size=VOCAB_SIZE, 
    hidden_dim=HIDDEN_DIM, 
    max_position_embeddings=MAX_POSITION,
    dropout_prob=0.1
)

# Jalankan Forward Pass Lapisan Embedding
output_embeddings = embedding_layer(t_input_ids, t_token_type_ids)

# =====================================================================
# VERIFIKASI OUTPUT & GRADIENT FLOW
# =====================================================================
print("=== VERIFIKASI EMBEDDING OUTPUT ===")
print("Dimensi Tensor Final (B, N, D):", output_embeddings.shape)
print("Apakah membutuhkan gradien?  :", output_embeddings.requires_grad)
print("Fungsi Gradien Generator (Backward Hook):", output_embeddings.grad_fn)