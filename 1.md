Selamat datang di **Sub-Fase 1.1: Teori, Formulasi Matematika, & Strategi WordPiece**. Karena Anda ingin hasil yang identik dengan arsitektur asli BERT/IndoBERT namun memulai dari nol terminologi, kita akan bedah setiap konsep dari akarnya sebelum masuk ke rumus matematika.

---

## 1. Menjembatani Teks dan Angka: Mengapa Kita Butuh Tokenizer?

Komputer atau model kecerdasan buatan tidak mengerti huruf atau kata; mereka hanya mengerti angka (tensor/matriks). Tugas awal kita adalah mengubah teks mentah menjadi deretan angka.

Sebelum memahami WordPiece, mari pahami tiga terminologi dasar ini:

* **Korpus (Corpus):** Kumpulan teks mentah dalam jumlah sangat besar (berbagai artikel, berita, atau buku bahasa Indonesia) yang digunakan sebagai bahan baku untuk melatih model.
* **Token:** Potongan teks terkecil hasil potongan dari sebuah kalimat. Token bisa berupa kata utuh, potongan kata (sub-kata), atau karakter tunggal (seperti huruf atau tanda baca).
* **Kosakata (Vocabulary/Vocab):** Sebuah kamus atau daftar berisi seluruh token unik yang dikenali oleh model. Setiap token dalam Vocab memiliki nomor indeks unik (ID). Jika ukuran kosakata ($V$) adalah 30.000, artinya model hanya mengenali 30.000 token tersebut.

---

## 2. Mengapa IndoBERT Menggunakan WordPiece?

Ada tiga cara memotong teks secara tradisional, dan semuanya memiliki masalah besar jika digunakan untuk model tingkat peneliti seperti IndoBERT:

1. **Berbasis Kata Utuh (Word-level):** Setiap kata unik masuk ke kamus.
* *Masalah:* Bahasa Indonesia memiliki banyak kata berimbuhan (*mempertanggungjawabkan*, *diperlakukan*). Kamus akan membengkak menjadi jutaan kata, namun model tetap akan menemukan kata baru yang tidak ada di kamus (**Out-of-Vocabulary / OOV**), seperti kata gaul atau istilah teknis baru. Jika kata tidak ada di kamus, model akan memberikan token `[UNK]` (Unknown), membuat model menjadi "buta" terhadap kata tersebut.


2. **Berbasis Karakter (Character-level):** Kamus hanya berisi huruf `a-z`, angka, dan simbol.
* *Masalah:* Model tidak kehilangan informasi kata baru, tetapi kalimat menjadi sangat panjang (huruf per huruf), membuat beban komputasi komputer meledak dan model sulit menangkap makna global sebuah kata.



### Solusi Jalan Tengah: Subword Tokenization (WordPiece)

WordPiece memecah kata-kata jarang atau panjang menjadi potongan sub-kata (**subwords**), namun membiarkan kata-kata yang sering muncul tetap utuh.

* Kata umum: `saya` $\rightarrow$ `['saya']` (tetap utuh karena sering muncul di korpus).
* Kata kompleks/jarang: `mempertanggungjawabkan` $\rightarrow$ `['memper', '##tanggung', '##jawab', '##kan']`.

> **Penting (Terminologi Simbol `##`):** Simbol `##` adalah penanda *prefix* (awalan) internal BERT. Simbol ini memberi tahu model bahwa token `##tanggung` secara struktural menyambung dengan token `memper` di depannya, bukan kata baru yang berdiri sendiri.

---

## 3. Formulasi Matematika: Bagaimana WordPiece Memilih Potongan Kata?

Jika kita punya korpus bahasa Indonesia, bagaimana algoritma WordPiece memutuskan apakah `memper` harus digabung dengan `tanggung` menjadi `mempertanggung` di dalam kamus?

Berbeda dengan algoritma *Byte-Pair Encoding* (BPE) yang hanya menghitung pasangan mana yang paling sering muncul (frekuensi murni), **WordPiece menggunakan pendekatan berbasis skor Likelihood (probabilitas/kemungkinan)**.

Setiap kali WordPiece ingin menggabungkan dua sub-kata (kita sebut saja sub-kata $A$ dan sub-kata $B$) untuk membentuk token baru $AB$, algoritma menghitung nilai menggunakan rumus berikut:

$$\text{Score}_{(A, B)} = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

### Penjelasan Komponen Rumus:

* $\text{count}(AB)$: Berapa kali gabungan kata $AB$ muncul bersamaan di dalam korpus.
* $\text{count}(A)$: Berapa kali sub-kata $A$ muncul secara independen di seluruh korpus.
* $\text{count}(B)$: Berapa kali sub-kata $B$ muncul secara independen di seluruh korpus.

### Logika di Balik Rumus (Intuisi Peneliti):

Rumus ini mencari **Mutual Information** (seberapa erat hubungan ketergantungan antar dua sub-kata).

* **Kasus Skor Rendah:** Bayangkan kata $A$ adalah `dan` dan kata $B$ adalah `rumah`. Di dalam korpus, kata `dan` muncul jutaan kali, kata `rumah` juga muncul ribuan kali. Namun, gabungan kata `danrumah` hampir tidak pernah ada. Maka pembagi di bawah ($\text{count}(A) \times \text{count}(B)$) nilainya sangat besar, membuat skornya mendekati 0. Algoritma **tidak akan** menggabungkannya ke dalam kamus.
* **Kasus Skor Tinggi:** Bayangkan kata $A$ adalah `ter` dan kata $B$ adalah `nyata`. Kata `ter` sering muncul, `nyata` juga sering muncul, tetapi kombinasi kata `ternyata` ($AB$) muncul sangat konsisten dan sering di dalam korpus bahasa Indonesia. Skor yang dihasilkan akan sangat tinggi. Algoritma akan memasukkan `ternyata` sebagai token utuh di dalam kosakata.

---

## 4. Anatomi Token Spesial BERT (Mekanisme Arsitektur)

Agar identik dengan arsitektur BERT asli, kosakata kita wajib menyisihkan tempat untuk 5 token spesial berikut. Masing-masing memiliki fungsi mekanis yang sangat kaku pada pemrosesan matriks nanti:

1. `[PAD]` (Padding - ID: biasanya 0): Digunakan untuk menyamakan panjang kalimat. Komputer membutuhkan input data berupa matriks persegi/persegi panjang yang seragam (misal: semua kalimat harus memiliki panjang 128 token). Jika kalimat Anda hanya berisi 5 token, sisa 123 slot akan diisi oleh `[PAD]`.
2. `[UNK]` (Unknown - ID: biasanya 1): Jika ada karakter aneh, emoji, atau bahasa asing yang benar-benar tidak terdaftar di kosakata kita setelah dipecah sampai level karakter, karakter tersebut terpaksa diubah menjadi `[UNK]`.
3. `[CLS]` (Classification - ID: biasanya 2): **Wajib diletakkan di awal setiap kalimat.** Fungsi mekanisnya vital: saat masuk ke lapisan Transformer, token ini tidak membawa arti kata apa pun, melainkan bertugas menyerap seluruh intisari informasi dari semua kata di dalam kalimat tersebut melalui mekanisme interaksi matriks (*Self-Attention*).
4. `[SEP]` (Separator - ID: biasanya 3): Digunakan sebagai tanda batas akhir suatu kalimat, atau pemisah ketika kita memasukkan dua kalimat sekaligus (misal: Kalimat A `[SEP]` Kalimat B) untuk tugas mendeteksi hubungan antar-kalimat.
5. `[MASK]` (Masking - ID: biasanya 4): Token misterius yang digunakan saat proses latihan model (*pre-training*). Kita sengaja menghapus sebuah kata asli dalam kalimat bahasa Indonesia dan menggantinya dengan `[MASK]`, lalu memaksa model menebak kata apa yang disembunyikan di balik token tersebut.

---

Apakah seluruh penjelasan terminologi, logika pembagian kata berbasis rumus WordPiece, serta fungsi mekanis token spesial di atas sudah dapat Anda pahami dengan jelas sebelum kita mulai membangun kode pembuat kosakatanya pada Sub-Fase 1.2?