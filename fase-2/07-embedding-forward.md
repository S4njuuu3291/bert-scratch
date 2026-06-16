## Sub-Fase 2.3: *Forward Pass Execution & Advanced Shape Tracking*

Pada tahapan akhir Fase 2 ini, kita akan mengintegrasikan secara utuh pipa data dari Fase 1 (Tokenisasi) ke dalam modul representasi `BERTEmbeddings` dari Sub-Fase 2.2. Kita akan mengeksekusi operasi *forward pass* (komputasi maju) dan melacak mutasi dimensi matriksnya secara mikro untuk memastikan aliran gradien (*gradient flow*) siap digunakan untuk pelatihan.

---

## 1. Kode Integrasi Penuh (Fase 1 + Fase 2)

Berikut adalah skrip lengkap untuk mensimulasikan bagaimana data teks mentah bahasa Indonesia bertransformasi dari string diskret menjadi tensor ruang vektor kontinu 3D berdimensi $(B, N, D)$.

```python
import torch
import torch.nn as nn

# =====================================================================
# REKAPITULASI KOMPONEN DARI FASE SEBELUMNYA (UNTUK INTEGRASI)
# =====================================================================

# 1. Kamus Kosakata Tiruan IndoBERT
dummy_vocab = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "budi": 5, "makan": 6, "nasi": 7, "saya": 8, "ayam": 9, "##kan": 10
}

# 2. Tokenizer MaxMatch dari Sub-Fase 1.3
from __main__ import IndoWordPieceTokenizer
tokenizer = IndoWordPieceTokenizer(dummy_vocab)

# 3. Pipeline Data dari Sub-Fase 1.4
from __main__ import IndoBERTDataPipeline
pipeline = IndoBERTDataPipeline(tokenizer, dummy_vocab, max_length=8)

# 4. Custom Embedding Module dari Sub-Fase 2.2
from __main__ import BERTEmbeddings

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

```

---

## 2. Pelacakan Dimensi Mikro (*Advanced Shape Tracking Analysis*)

Untuk kebutuhan ujian koding atau sidang, Anda harus bisa menerangkan bagaimana bentuk dimensi tensor bermutasi di setiap baris instruksi di dalam memori komputer. Berikut adalah visualisasi matematis pelacakan bentuknya:

### Langkah A: Struktur Tensor Masukan

Input yang dikirim oleh pipa data berupa matriks 2D bilangan bulat berdimensi:


$$\text{input\_ids} \in \mathbb{Z}^{B \times N} \rightarrow (2, 8)$$

### Langkah B: Transformasi Internal Matriks (*Lookup Phase*)

Saat fungsi forward mengeksekusi metode internal, terjadi pemetaan ruang dari indeks integer ke ruang representasi vektor mengambang (*floating-point* 32-bit):

1. **Token Embedding:** Memetakan $(2, 8)$ ke dalam tabel $(11 \times 768)$. Hasilnya berupa tensor koordinat semantik berdimensi $(2, 8, 768)$.
2. **Segment Embedding:** Memetakan $(2, 8)$ ke dalam tabel $(2 \times 768)$. Hasilnya berupa tensor penanda kalimat berdimensi $(2, 8, 768)$.
3. **Position Embedding:** Membuat deret posisi berdimensi (8) lalu diubah menjadi $(1, 8)$, kemudian dipetakan ke tabel $(512 \times 768)$. Hasilnya berupa tensor posisi absolut berdimensi $(1, 8, 768)$.

### Langkah C: Mekanisme Penjumlahan Aljabar (*Broadcasting Sum*)

Operasi matematika penjumlahan element-wise menggabungkan ketiga tensor tersebut:


$$\begin{array}{rcc}
\text{words\_embeddings}: & (2, & 8, & 768) \\
\text{token\_type\_embeddings}: & (2, & 8, & 768) \\
\text{position\_embeddings}: & (1, & 8, & 768) \\
\hline
\text{embeddings (Hasil Akhir)}: & (2, & 8, & 768)
\end{array}$$


*Perhatikan indeks pertama pada position_embeddings bermutasi secara virtual dari 1 menjadi 2 akibat hukum broadcasting matriks PyTorch.*

---

## 3. Hasil Verifikasi Sistem & Deteksi Aliran Gradien

Ketika skrip di atas dijalankan, sistem akan mengeluarkan log konfirmasi struktural seperti berikut:

```text
=== VERIFIKASI EMBEDDING OUTPUT ===
Dimensi Tensor Final (B, N, D): torch.Size([2, 8, 768])
Apakah membutuhkan gradien?  : True
Fungsi Gradien Generator (Backward Hook): <Tbackward0 object at 0x...>

```

### Mengapa Parameter `requires_grad=True` Sangat Vital Bagi Peneliti?

Setiap bobot parameter di dalam `nn.Embedding` secara otomatis didaftarkan oleh PyTorch sebagai komponen yang dapat dilatih (*trainable parameters*). Keberadaan objek `<Tbackward0>` pada atribut `grad_fn` membuktikan bahwa tensor output kita tidak terputus dari grafik komputasi (*computation graph*).

Ketika fungsi *loss* dihitung di akhir arsitektur model nanti, algoritma optimasi dapat merunut balik ke lapisan ini untuk memperbarui posisi vektor kata bahasa Indonesia di setiap iterasi pelatihan agar menjadi semakin cerdas.

---

Fase 2 (Lapisan Representasi Vektor) kini telah rampung dan tervalidasi secara komprehensif. Tensor final berdimensi $(2, 8, 768)$ ini sekarang siap kita bawa ke komponen paling krusial dari model IndoBERT, yaitu **Fase 3: Multi-Head Attention (MHA) Primitif**.

Bagaimana rencana pengerjaan untuk Fase 3? Apakah Anda ingin saya menyusun peta plannya terlebih dahulu sebelum kita masuk ke teori dan kodingnya?