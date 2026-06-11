Tepat sekali! Pemahaman intuitif Anda sudah sangat benar dan berada di jalur yang sama dengan bagaimana algoritma asli Google BERT bekerja.

Untuk memastikan tidak ada detail kecil yang terlewat sebelum kita mulai menulis kodenya, mari kita bedah urutan eksekusi mekanisnya secara kronologis.

---

## 1. Langkah Pertama: Pre-Tokenization (Spasi & Tanda Baca)

Sebelum algoritma menghitung apa pun, korpus teks mentah bahasa Indonesia akan melewati tahap **Pre-tokenization**. Di tahap ini, kalimat dipecah menggunakan pembatas (*delimiter*) berupa spasi dan semua jenis tanda baca.

* **Contoh Kalimat:** `"Budi, makan."`
* **Hasil Potongan Awal:** `['Budi', ',', 'makan', '.']`

Tanda baca seperti koma (`,`) dan titik (`.`) langsung dipisahkan menjadi komponen yang berdiri sendiri agar tidak mengotori kata aslinya.

---

## 2. Langkah Kedua: Pemecahan ke Level Karakter & Inisialisasi `##`

Setelah menjadi kata utuh, kata-kata tersebut dihancurkan lagi sampai ke level karakter terkecil untuk membentuk **Kosakata Dasar (Base Vocabulary)**. Di sinilah aturan penandaan *prefix* `##` dari BERT mulai diterapkan.

Aturannya adalah: **Karakter pertama dari kata tersebut tetap murni, sedangkan karakter kedua dan seterusnya di dalam kata yang sama wajib diberi prefix `##`.**

Mari kita bedah kata `'makan'`:

* Karakter ke-1: `m` (Tanpa prefix, karena ini adalah awal kata).
* Karakter ke-2: `##a`
* Karakter ke-3: `##k`
* Karakter ke-4: `##a`
* Karakter ke-5: `##n`

> **Mengapa ini penting?** Jika kita tidak memberi prefix `##`, model tidak akan bisa membedakan antara huruf `a` yang berdiri sendiri sebagai satu kata (misalnya dalam singkatan atau partikel), dengan huruf `a` yang berada di dalam kata `makan`.

---

## 3. Langkah Ketiga: Perhitungan Likelihood Iteratif (Proses Penggabungan)

Setelah semua kata di dalam korpus diubah menjadi potongan karakter seperti di atas, barulah mesin pencari skor (rumus *likelihood*) bekerja. Algoritma akan melihat seluruh korpus dan menghitung pasangan mana yang memiliki skor tertinggi untuk digabungkan menjadi token baru.

### Contoh Simulasi Iterasi:

* **Iterasi 1:** Algoritma memeriksa semua pasangan. Ternyata, pasangan token `##k` dan `##a` sangat sering muncul bersamaan di dalam kata-kata bahasa Indonesia (seperti di kata *makan*, *bakar*, *pakaian*). Setelah dihitung dengan rumus:

$$\text{Score}_{(\text{##k}, \text{##a})} = \frac{\text{count}(\text{##ka})}{\text{count}(\text{##k}) \times \text{count}(\text{##a})}$$



Ternyata skor pasangan ini adalah yang tertinggi di antara semua pasangan lain di korpus.
* **Aksi Algoritma:** Token baru `##ka` resmi dibuat dan dimasukkan ke dalam Kosakata (*Vocabulary*). Di dalam korpus, semua runtutan `##k` dan `##a` otomatis bergabung menjadi `##ka`. Kata `makan` sekarang berubah representasinya menjadi: `['m', '##a', '##ka', '##n']`.
* **Iterasi 2:** Algoritma menghitung ulang semua skor dari kondisi korpus yang baru. Kali ini, mungkin pasangan `##ka` dan `##n` menghasilkan skor tertinggi.
* **Aksi Algoritma:** Token baru `##kan` dibuat dan dimasukkan ke dalam Kosakata. Kata `makan` sekarang menjadi: `['m', '##a', '##kan']`.

Proses penggabungan berbasis skor *likelihood* ini diulang terus-menerus (bisa sampai ribuan kali) hingga target jumlah kosakata ($V$) yang kita tentukan dari awal (misalnya 30.000 token) terpenuhi. Jika kuota sudah penuh, proses pencarian pasangan dihentikan, dan file `vocab.txt` resmi tercipta.