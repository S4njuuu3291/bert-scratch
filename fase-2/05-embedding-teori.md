## Sub-Fase 2.1: Teori, Formulasi Matematika, & Superposisi Vektor

Di Fase 1, kita berhasil mengubah kalimat bahasa Indonesia menjadi token integer berdimensi $(B, N)$. Namun, angka-angka indeks ini bersifat diskret dan tidak memiliki nilai geometris (misalnya, ID 5 dan ID 6 tidak berarti keduanya memiliki kedekatan makna).

Di Fase 2 ini, kita masuk ke **Lapisan Representasi Vektor (Embedding Layer)**. Tugas lapisan ini adalah memetakan indeks diskret tersebut ke dalam ruang vektor kontinu berdimensi tinggi ($D = 768$ untuk IndoBERT-Base) yang mengombinasikan tiga pilar informasi struktural sekaligus.

---

## 1. Tiga Pilar Informasi dalam Ruang Representasi BERT

Untuk memahami sebuah kata dalam konteks bahasa Indonesia, model tidak hanya perlu tahu *apa* arti katanya, tetapi juga *di mana* posisi kata tersebut, dan *di mana* batasan kalimatnya. Oleh karena itu, BERT menggabungkan tiga jenis matriks embedding:

```text
input_ids -------> [ Token Embeddings Matrix ] ----\
token_type_ids --> [ Segment Embeddings Matrix ] --> (+) --> LayerNorm --> Dropout --> Output Tensor (B, N, D)
position_ids ----> [ Position Embeddings Matrix ] --/

```

### A. Token Embeddings

* **Fungsi:** Menyimpan makna semantik dasar dari sub-kata.
* **Mekanisme:** Sebuah tabel pencarian (*lookup table*) berukuran $(V \times D)$ di mana $V$ adalah ukuran kosakata (misal 30.000) dan $D$ adalah 768. Setiap baris mewakili satu token sub-kata unik dalam bahasa Indonesia.

### B. Segment Embeddings (Token Type Embeddings)

* **Fungsi:** Memberikan tanda batas ketika model menerima input pasang kalimat (misalnya untuk mendeteksi apakah kalimat B adalah jawaban dari kalimat A).
* **Mekanisme:** Matriks berukuran kaku $(2 \times D)$. Semua token yang berasal dari kalimat pertama akan mengambil vektor di baris ke-0, dan semua token dari kalimat kedua akan mengambil vektor di baris ke-1.

### C. Position Embeddings

* **Fungsi:** Memberikan informasi urutan kata. Karena arsitektur *Self-Attention* pada Transformer memproses seluruh kata secara simultan (paralel), tanpa adanya koordinat posisi, kalimat *"Budi memukul ayam"* dan *"Ayam memukul Budi"* akan menghasilkan representasi awal yang sama persis.
* **Mekanisme:** Matriks berukuran $(M \times D)$ di mana $M$ adalah panjang maksimum sekuens yang diizinkan (BERT membatasi $M = 512$).

---

## 2. Penyelidikan Peneliti: Learned Absolute vs. Sinusoidal Position Embeddings

Pada makalah asli Transformer (*Attention Is All You Need*), penulis menggunakan **Sinusoidal Position Embeddings**, yaitu koordinat posisi yang dihitung menggunakan fungsi trigonometri statis ($\sin$ dan $\cos$) tanpa ada parameter yang perlu dilatih.

Namun, **BERT sengaja memilih Learned Absolute Position Embeddings**. Artinya, matriks posisi berukuran $(512 \times 768)$ diinisialisasi secara acak dan nilai-nilainya dipelajari secara dinamis lewat *backpropagation* bersamaan dengan parameter lainnya.

### Mengapa BERT memilih pendekatan ini?

1. **Karakteristik Pre-training:** BERT dilatih menggunakan korpus raksasa lewat tugas MLM (*Masked Language Modeling*). Dengan kapasitas data yang masif, model memiliki ruang komputasi yang cukup untuk mempelajari hubungan jarak antar-posisi secara empiris langsung dari data, bukan memaksakan pola geometris kaku dari fungsi roda sinus.
2. **Batasan Sekuens Kaku:** Keunggulan utama fungsi sinusoidal adalah kemampuan ekstrapolasi (bisa menangani panjang kalimat hingga tak terhingga saat inferensi). Namun, karena BERT secara arsitektur sejak awal membatasi panjang input maksimal di angka 512 token, keunggulan ekstrapolasi tersebut tidak lagi krusial. Karakteristik posisi absolut terbukti memberikan performa yang lebih stabil pada panjang sekuens yang terikat.

---

## 3. Formulasi Matematika: Superposisi Vektor lewat Penjumlahan Element-wise

Pertanyaan krusial di tingkat peneliti: Mengapa ketiga vektor embedding tersebut **dijumlahkan** ($E_{\text{total}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$), bukan **dikonkatenasi** ([ $E_{\text{token}} \parallel E_{\text{segment}} \parallel E_{\text{position}}$ ])?

Jika dikonkatenasi, dimensi vektor akan membengkak dari 768 menjadi $768 \times 3 = 2304$. Ini akan melipatgandakan jumlah parameter pada lapisan-lapisan berikutnya secara masif dan memperlambat komputasi.

Secara matematis, penjumlahan ini memanfaatkan fenomena geometri dimensi tinggi yang disebut **Superposisi Informasi (Information Superposition)**.

> **Teorema Ruang Dimensi Tinggi:** Dalam ruang vektor berdimensi tinggi (seperti $D = 768$), jika kita memilih dua atau lebih vektor secara acak, kemungkinan besar vektor-vektor tersebut akan saling tegak lurus (**quasi-orthogonal**).

Ketika tiga informasi orthogonal dijumlahkan:


$$E_{\text{total}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$$

Vektor hasil penjumlahan ($E_{\text{total}}$) tidak akan kehilangan identitas komponen penyusunnya. Lapisan linier proyeksi pada Transformer ($W_Q, W_K, W_V$) memiliki kapasitas matematis yang sangat tinggi untuk mengekstrak atau "mengurai" kembali informasi posisi atau semantik kata dari hasil perpaduan tersebut melalui operasi perkalian matriks berikutnya.

---

## 4. Mekanisme Layer Normalization (LayerNorm)

Setelah ketiga embedding dijumlahkan, tensor wajib melewati lapisan **Layer Normalization**. Ini adalah salah satu kunci utama stabilitas pelatihan model deep Transformer.

### Mengapa bukan Batch Normalization (BatchNorm)?

Pada data teks, panjang kalimat dalam satu *batch* sangat bervariasi, sehingga kita terpaksa menggunakan token `[PAD]`. Jika kita menggunakan BatchNorm (normalisasi dilakukan vertikal memotong antar-sampel dalam satu batch), nilai statistik mean dan varians akan rusak karena terpolusi oleh nilai dari token-token `[PAD]` siluman tersebut.

**LayerNorm memotong secara horizontal.** Ia menghitung mean dan varians dari fitur internal ($D$) satu token itu sendiri, terisolasi dari token lain maupun sampel lain dalam batch.

### Formulasi Matematika LayerNorm

Untuk sebuah vektor representasi token pada posisi tertentu $x \in \mathbb{R}^D$:

1. **Hitung Mean ($\mu$) dari vektor tersebut:**

$$\mu = \frac{1}{D} \sum_{i=1}^{D} x_i$$


2. **Hitung Varians ($\sigma^2$):**

$$\sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^2$$


3. **Normalisasi Vektor:**

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$



*Catatan: $\epsilon$ (epsilon) adalah konstanta sangat kecil (misal $10^{-12}$) untuk mencegah pembagian dengan nol jika varians bernilai 0.*
4. **Skala dan Pergeseran Berbasis Parameter Latih ($\gamma$ dan $\beta$):**

$$\text{LayerNorm}(x)_i = \hat{x}_i \cdot \gamma_i + \beta_i$$



*Di sini, $\gamma$ (gamma) dan $\beta$ (beta) adalah tensor berdimensi $(D)$ yang merupakan parameter adaptif yang ikut dilatih saat pre-training.*

Setelah dinormalisasi, tensor dilewatkan ke fungsi **Dropout** untuk mematikan sebagian neuron secara acak demi mencegah *overfitting*, menghasilkan tensor final yang siap masuk ke blok utama Transformer Encoder.