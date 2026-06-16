## Sub-Fase 2.2: Implementasi `BERTEmbeddings` Module Primitif

Pada tahap ini, kita akan menuangkan seluruh landasan teori dan formulasi matematika dari Sub-Fase 2.1 ke dalam baris kode PyTorch primitif. Kita akan membangun sebuah *custom module* bernama `BERTEmbeddings` yang mewarisi `torch.nn.Module`.

---

## 1. Kode Lengkap Kelas `BERTEmbeddings` (Pure PyTorch)

Perhatikan implementasi di bawah ini. Kode dirancang tanpa menggunakan pustaka eksternal tingkat tinggi untuk memperlihatkan bagaimana alokasi memori bobot matriks didefinisikan secara eksplisit.

```python
import torch
import torch.nn as nn

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_position_embeddings=512, dropout_prob=0.1):
        """
        Inisialisasi Lapisan Embedding Kaku sesuai Spesifikasi BERT Original.
        
        vocab_size: Ukuran total kosakata (V)
        hidden_dim: Dimensi tersembunyi representasi vektor (D, contoh: 768)
        max_position_embeddings: Batas sekuens maksimum (M, contoh: 512)
        dropout_prob: Probabilitas neuron dimatikan pada lapisan Dropout
        """
        super().__init__()
        
        # 1. Alokasi Matriks Token Embedding (V x D)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        
        # 2. Alokasi Matriks Position Embedding Absolute (M x D)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        
        # 3. Alokasi Matriks Segment Embedding (2 x D)
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)

        # 4. Operasi Normalisasi & Regulasi
        # eps (epsilon) diatur ke 1e-12 sesuai dengan paper asli BERT
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        Melakukan komputasi maju (Forward Pass) untuk superposisi matriks.
        input_ids: Tensor indeks kata berukuran (B, N)
        token_type_ids: Tensor indeks segment kalimat berukuran (B, N)
        """
        # Ekstraksi panjang sekuens nyata (N) dari dimensi tensor input
        seq_length = input_ids.size(1)
        
        # Pembuatan koordinat posisi dinamis [0, 1, 2, ..., N-1]
        # Tensor dibuat langsung di device yang sama dengan input_ids (CPU/GPU)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        
        # Ekspansi dimensi dari (N) menjadi (1, N) agar bisa di-broadcast
        position_ids = position_ids.unsqueeze(0)

        # Antisipasi jika token_type_ids tidak dimasukkan (default ke Kalimat A semua)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 5. Proses Lookup Vektor ke Masing-masing Tabel Matriks
        # Hasil token_embeddings bertransformasi dari (B, N) -> (B, N, D)
        words_embeddings = self.token_embeddings(input_ids)
        
        # Hasil token_type_embeddings bertransformasi dari (B, N) -> (B, N, D)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Hasil position_embeddings bertransformasi dari (1, N) -> (1, N, D)
        position_embeddings = self.position_embeddings(position_ids)

        # 6. Eksekusi Superposisi Matematika (Penjumlahan Element-wise)
        # Sifat penambahan position_embeddings (1, N, D) ke tensor (B, N, D) 
        # akan memicu fungsi Broadcasting otomatis oleh PyTorch secara efisien.
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # 7. Operasi Finalisasi Vektor
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Mengembalikan tensor siap pakai berdimensi (B, N, D)
        return embeddings

```

---

## 2. Bedah Mekanis Logika Koding Tingkat Peneliti

Untuk kebutuhan sidang atau pemahaman fundamental, ada beberapa aspek arsitektur di dalam kode di atas yang perlu dicermati secara mendalam:

### A. Mekanisme Pembuatan `position_ids` Secara Otomatis

Perhatikan baris:

```python
position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
position_ids = position_ids.unsqueeze(0)

```

BERT tidak meminta pengguna memasukkan tensor posisi secara manual dari luar pipa data. Modul memantau panjang sekuens secara dinamis menggunakan `.size(1)`. Jika teks yang masuk setelah dipotong bernilai $N = 8$, maka `torch.arange(8)` akan otomatis menciptakan tensor vektor `[0, 1, 2, 3, 4, 5, 6, 7]`. Fungsi `.unsqueeze(0)` mengubahnya dari bentuk 1D tensor bertingkat dimensi `[8]` menjadi 2D matriks tipis berdimensi `[1, 8]`.

### B. Mekanisme Efisiensi *Broadcasting*

Ketika mengeksekusi operasi penjumlahan:


$$\text{embeddings} = \text{words\_embeddings} (B, N, D) + \text{position\_embeddings} (1, N, D)$$

PyTorch tidak akan menduplikasi data `position_embeddings` sebanyak $B$ kali di dalam memori RAM/VRAM. Secara virtual, engine aljabar linier PyTorch akan langsung merepetisi baris koordinat posisi tersebut untuk melayani penjumlahan di setiap *batch index* secara paralel. Hal ini menghemat alokasi memori secara signifikan saat memproses ukuran *batch size* ($B$) yang besar.

### C. Alasan Pengaturan Konstanta `eps=1e-12` pada `nn.LayerNorm`

Konstanta epsilon ($\epsilon$) dalam rumus normalisasi berfungsi sebagai katup pengaman agar sistem tidak melakukan operasi pembagian dengan angka nol ($0$). Pengaturan nilai $\epsilon = 10^{-12}$ diinisialisasi untuk menjaga presisi tingkat tinggi bagi arsitektur Transformer yang memiliki tumpukan lapisan sangat dalam, mencegah nilai gradien menjadi *underflow* atau mengalami lonjakan ekstrem (*exploding gradient*) selama proses latihan berjalan.

---

Modul `BERTEmbeddings` primitif ini telah selesai disusun dengan struktur parameter yang mutlak bersih dan identik dengan standar riset. Langkah berikutnya adalah menguji modul ini menggunakan data tensor yang diproduksi oleh komponen tokenisasi kita sebelumnya.