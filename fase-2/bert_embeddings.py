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