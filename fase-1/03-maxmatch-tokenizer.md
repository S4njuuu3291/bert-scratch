Kita langsung masuk ke **Sub-Fase 1.3: Mekanisme *String Matching* & *Tokenizer Execution***.

Setelah kita berhasil menciptakan daftar kosakata (*vocabulary dict*), tantangan berikutnya adalah: bagaimana model menggunakan kamus tersebut untuk memotong kalimat baru yang belum pernah ia lihat sebelumnya?

BERT menggunakan algoritma pencocokan string yang disebut **Greedy Longest Match First** (atau sering disebut **MaxMatch**).

---

## 1. Algoritma MaxMatch (Greedy Longest Match First)

Prinsip kerja algoritma ini sangat rakus (*greedy*). Ketika melihat sebuah kata utuh, ia akan mencari potongan sub-kata **terpanjang** yang tersedia di dalam kamus dari arah kiri ke kanan.

### Simulasi Cara Kerja

Bayangkan kita memiliki kosakata hasil latihan dari Sub-Fase 1.2 yang berisi token:
`["saya", "makan", "nasi", "budi", "##kan", "ma", "##k"]`

Jika model menerima input kata baru: **`"makankan"`**, berikut adalah urutan mekanis yang terjadi di dalam kode:

1. **Pemeriksaan Kata Utuh:** Apakah kata `"makankan"` ada di kamus? **Tidak ada**.
2. **Penyusutan Kanan ke Kiri (Iterasi 1):**
* Hapus 1 huruf paling belakang: Apakah `"makanka"` ada di kamus? Tidak.
* Hapus lagi: Apakah `"makanak"`? Tidak.
* ... sampai tersisa `"makan"`. Apakah `"makan"` ada di kamus? **Ada!**
* *Aksi:* Simpan `"makan"` sebagai token pertama. String yang tersisa sekarang adalah `"kan"`.


3. **Pemeriksaan Sisa String (Iterasi 2):** Karena `"kan"` adalah kelanjutan dari kata sebelumnya, algoritma wajib menambahkan prefix `##` sebelum memeriksa kamus. Jadi, ia mencari kata **`"##kan"`**.
* Apakah `"##kan"` ada di kamus? **Ada!**
* *Aksi:* Simpan `"##kan"` sebagai token kedua. Sisa string habis.


4. **Hasil Akhir:** `["makan", "##kan"]`

> **Aturan Penting Out-of-Vocabulary (OOV):** Jika algoritma menyusutkan string terus-menerus sampai tersisa 1 karakter pertama (misal huruf `"m"`) dan ternyata huruf `"m"` tersebut pun tidak ada di kamus, maka **seluruh kata utuh tersebut langsung digantikan oleh token `[UNK]**`. BERT tidak memotong parsial jika ujung-ujungnya ada karakter yang tidak dikenali.

---

## 2. Implementasi Kelas `IndoWordPieceTokenizer` (Pure Python)

Berikut adalah implementasi *low-level* dari mesin *tokenizer* BERT. Kita akan membungkus kosakata hasil Fase 1.2 ke dalam sebuah *class* mandiri.

```python
class IndoWordPieceTokenizer:
    def __init__(self, vocab_dict):
        """
        vocab_dict: dictionary yang memetakan string token ke integer ID
        Contoh: {"[PAD]": 0, "saya": 5, "##kan": 6, ...}
        """
        self.vocab = vocab_dict
        self.unk_token = "[UNK]"
        
    def tokenize(self, text):
        """
        Memecah teks mentah menjadi list of token sub-kata berdasarkan kamus.
        """
        # Step 1: Pre-tokenization (Pecah berdasarkan spasi)
        # Catatan: Untuk simplisitas low-level, kita asumsikan teks sudah bersih dari tanda baca
        words = text.strip().split()
        output_tokens = []
        
        for word in words:
            chars = list(word)
            if len(word) > 100: # Proteksi token terlalu panjang
                output_tokens.append(self.unk_token)
                continue
                
            is_bad = False
            start = 0
            sub_tokens = []
            
            # Loop MaxMatch untuk satu kata utuh
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                
                while start < end:
                    substr = "".join(chars[start:end])
                    # Jika ini bukan karakter pertama dari kata utuh, wajib tambahkan ##
                    if start > 0:
                        substr = f"##{substr}"
                        
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                # Jika bahkan karakter tunggal tidak ditemukan di kamus
                if cur_substr is None:
                    is_bad = True
                    break
                    
                sub_tokens.append(cur_substr)
                start = end
            
            # Jika ada bagian kata yang OOV, gantikan satu kata utuh tersebut dengan [UNK]
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
                
        return output_tokens

```

---

## 3. Uji Coba Mesin Tokenizer

Mari kita uji *class* di atas menggunakan kosakata tiruan kita untuk melihat apakah ia berhasil memotong kata secara *greedy*.

```python
# Menyiapkan kamus tiruan hasil dari Fase 1.2
dummy_vocab = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "budi": 5, "makan": 6, "nasi": 7, "saya": 8, "ayam": 9,
    "##kan": 10, "ma": 11, "##k": 12, "x": 13  # 'x' sengaja tidak dimasukkan ke alfabet dasar
}

# Inisialisasi objek tokenizer
tokenizer = IndoWordPieceTokenizer(dummy_vocab)

# Test Kasus 1: Kata yang ada di kamus + kata bentukan baru (makankan)
teks_1 = "budi makankan nasi"
tokens_1 = tokenizer.tokenize(teks_1)
print(f"Input  : '{teks_1}'")
print(f"Tokens : {tokens_1}\n")
# Output harusnya: ['budi', 'makan', '##kan', 'nasi']

# Test Kasus 2: Mengandung karakter OOV ('x')
teks_2 = "saya makan axam" # ayam typo jadi axam, huruf 'x' tidak dikenali di substring kata kedua
tokens_2 = tokenizer.tokenize(teks_2)
print(f"Input  : '{teks_2}'")
print(f"Tokens : {tokens_2}")
# Output harusnya: ['saya', 'makan', '[UNK]'] -> 'axam' runtuh total menjadi [UNK]

```

---

Mesin *string matching* kita sudah bekerja secara akurat sesuai dengan spesifikasi repositori asli BERT Google. Langkah terakhir untuk menuntaskan Fase 1 adalah mengubah output token string tersebut menjadi tiga tensor utama PyTorch (`input_ids`, `attention_mask`, `token_type_ids`).

Apakah kode *string matching* ini sudah aman untuk kita bawa ke tahapan pipa tensor di **Sub-Fase 1.4**?