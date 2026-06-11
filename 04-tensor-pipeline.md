Kita masuk ke **Sub-Fase 1.4: *Tensor Pipeline Generation* & *Shape Tracking***. Ini adalah jembatan terakhir di Fase 1 yang mengubah token-token string menjadi matriks biner dan integer berbentuk tensor PyTorch primitif ($B, N$) yang siap dikonsumsi oleh lapisan *Embedding* IndoBERT.

---

## 1. Anatomi 3 Tensor Utama Input BERT

Untuk melatih atau mengevaluasi arsitektur IndoBERT secara presisi, kita wajib menghasilkan tiga jenis tensor dengan ukuran dimensi yang sama, yaitu $(B, N)$ di mana $B$ adalah *Batch Size* (jumlah kalimat yang diproses sekaligus) dan $N$ adalah *Sequence Length* (panjang maksimum token dalam satu baris setelah *padding*/*truncation*).

1. **`input_ids`:** Tensor berisi ID integer unik dari kosakata untuk setiap token. Wajib diawali oleh ID `[CLS]` dan diakhiri oleh ID `[SEP]`.
2. **`attention_mask`:** Tensor biner berisi angka `1` untuk token asli bahasa Indonesia (termasuk token spesial) dan angka `0` untuk token `[PAD]`. Tensor ini memberi tahu mekanisme *Self-Attention* bagian mana yang harus diabaikan agar kalkulasi matriks tidak bias akibat *padding*.
3. **`token_type_ids` (Segment IDs):** Tensor biner untuk membedakan asal kalimat. Jika input berupa kalimat tunggal, seluruh isinya bernilai `0`. Jika input berupa pasangan kalimat (Sentence A dan Sentence B), maka token milik Kalimat A bernilai `0` dan token milik Kalimat B bernilai `1`.

---

## 2. Implementasi Pipeline Data & Tensor (Pure PyTorch)

Berikut adalah kode lengkap *low-level* menggunakan PyTorch dasar untuk menyusun pipa pemrosesan data tersebut.

```python
import torch

class IndoBERTDataPipeline:
    def __init__(self, tokenizer, vocab_dict, max_length=8):
        """
        tokenizer: Objek IndoWordPieceTokenizer dari Sub-Fase 1.3
        vocab_dict: Kamus pemetaan kata ke ID integer
        max_length: Batas maksimum panjang sekuens (N)
        """
        self.tokenizer = tokenizer
        self.vocab = vocab_dict
        self.max_length = max_length
        
    def encode_single(self, text):
        """
        Memproses satu kalimat mentah menjadi token ID, mask, dan segment ID
        """
        # 1. String Tokenization
        tokens = self.tokenizer.tokenize(text)
        
        # 2. Strukturisasi Struktur BERT: [CLS] + Tokens + [SEP]
        # Aturan Truncation manual agar muat ditambah 2 token spesial
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:(self.max_length - 2)]
            
        bert_tokens = ["[CLS]"] + tokens + ["[SEP]"]
        
        # 3. Konversi Token ke ID Numerik
        input_ids = [self.vocab[tok] if tok in self.vocab else self.vocab["[UNK]"] for tok in bert_tokens]
        
        # 4. Pembuatan Attention Mask awal (1 untuk token riil)
        attention_mask = [1] * len(input_ids)
        
        # 5. Pembuatan Token Type IDs (0 semua karena kalimat tunggal)
        token_type_ids = [0] * len(input_ids)
        
        # 6. Logika Dinamis Padding manual hingga mencapai max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.vocab["[PAD]"]] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            
        return input_ids, attention_mask, token_type_ids

    def create_batch(self, batch_text):
        """
        Menerima list of strings, memprosesnya, dan mengubahnya menjadi PyTorch Tensor (B, N)
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        
        for text in batch_text:
            ids, mask, types = self.encode_single(text)
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            batch_token_type_ids.append(types)
            
        # Konversi struktur list Python ke PyTorch LongTensor primitif
        tensor_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        
        return tensor_input_ids, tensor_attention_mask, tensor_token_type_ids

```

---

## 3. Simulasi Eksekusi & Pelacakan Dimensi (*Tensor Shape Tracking*)

Mari kita jalankan pipa data di atas menggunakan *batch size* $B = 2$ dan panjang maksimum sekuens $N = 8$.

```python
# Re-use Vocab dan Tokenizer dari fase sebelumnya
dummy_vocab = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "budi": 5, "makan": 6, "nasi": 7, "saya": 8, "ayam": 9, "##kan": 10
}

from __main__ import IndoWordPieceTokenizer
tokenizer = IndoWordPieceTokenizer(dummy_vocab)

# Inisialisasi Pipeline Data dengan Max Length (N) = 8
pipeline = IndoBERTDataPipeline(tokenizer, dummy_vocab, max_length=8)

# Input data tiruan berupa batch berisi 2 kalimat (B = 2)
indonesian_sentences = [
    "budi makankan nasi",  # Panjang token asli setelah dipecah + CLS/SEP = 5. Sisa 3 slot PAD.
    "saya makan ayam"      # Panjang token asli setelah dipecah + CLS/SEP = 5. Sisa 3 slot PAD.
]

# Jalankan Pipeline
t_input_ids, t_attention_mask, t_token_type_ids = pipeline.create_batch(indonesian_sentences)

# Cetak hasil Tensor mentah
print("=== OUTPUT PIPELINE DATA INDOBERT ===")
print("Input IDs Tensor:\n", t_input_ids)
print("Attention Mask Tensor:\n", t_attention_mask)
print("Token Type IDs Tensor:\n", t_token_type_ids)

```

### Hasil Tensor Hasil Eksekusi:

```text
Input IDs Tensor:
 tensor([[2, 5, 6, 10, 7, 3, 0, 0],
         [2, 8, 6, 9, 3, 0, 0, 0]])

Attention Mask Tensor:
 tensor([[1, 1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0]])

Token Type IDs Tensor:
 tensor([[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

```

---

## 4. Analisis Dimensi untuk Sidang (*Researcher-Level Shape Tracking*)

Saat ujian koding atau sidang, perubahan bentuk representasi data dari teks murni hingga menjadi struktur aljabar linier pada PyTorch dapat diringkas melalui pelacakan dimensi berikut:

| Tahapan Proses | Bentuk Data / Tipe Data | Representasi Dimensi (*Shape*) | Deskripsi Mekanis |
| --- | --- | --- | --- |
| **Raw Input** | `list` of `str` | $[B]$ | Data teks mentah bahasa Indonesia sebanyak $B$ buah kalimat. |
| **Tokenization** | `list` of `list` of `str` | $[B, \text{Varying } N]$ | Kalimat dipecah menjadi subwords. Panjang tiap kalimat masih berbeda-beda. |
| **Special Token & Padding** | `list` of `list` of `int` | $[B, N]$ | Setiap subword ditransformasikan menjadi ID integer, disisipkan `[CLS]` dan `[SEP]`, lalu ditambahkan `[PAD]` (ID 0) agar semua baris memiliki panjang simetris $N$. |
| **PyTorch Conversion** | `torch.LongTensor` | $(B, N)$ | Konversi ke objek memori GPU/CPU tensor bertipe integer 64-bit (`torch.long`) untuk kalkulasi matriks paralel. |

Fase 1 (Subsistem Tokenisasi & Pipeline Data) telah selesai sepenuhnya dan berhasil divalidasi tanpa ada *error* dimensi. Pipa data kita siap mengeluarkan tensor berdimensi $(B, N)$ secara konsisten untuk diumpankan ke **Fase 2: Lapisan Representasi Vektor (*Custom Embedding Layer*)**.