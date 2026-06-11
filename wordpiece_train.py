from collections import defaultdict

# =====================================================================
# STEP 1: MENYIAPKAN DATA & KOSAKATA DASAR
# =====================================================================

# Korpus dummy bahasa Indonesia
corpus = [
    "budi mempelajari teknologi kecerdasan buatan",
    "kecerdasan buatan membantu pekerjaan manusia sehari hari",
    "budi membuat model pembelajaran mesin menggunakan python",
    "teknologi ini digunakan untuk memprediksi data masa depan",
    "pembelajaran mesin memerlukan data yang sangat besar",
    "manusia mengembangkan teknologi untuk membantu kehidupan",
    "model kecerdasan buatan ini dipelajari oleh banyak orang",
    "penggunaan teknologi buatan manusia berkembang sangat cepat",
    "data dikumpulkan lalu digunakan untuk melatih model",
    "budi dan teman teman membuat aplikasi berbasis kecerdasan buatan"
]

# Ukuran target kosakata akhir diperbesar agar mampu membentuk kata utuh yang kompleks
TARGET_VOCAB_SIZE = 250 

# Inisialisasi Token Spesial wajib BERT
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Alfabet lengkap bahasa Indonesia yang muncul di korpus
alphabet = ["a", "b", "c", "d", "e", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "v", "y"]

for char in alphabet:
    if char not in vocab:
        vocab.append(char)
    if f"##{char}" not in vocab:
        vocab.append(f"##{char}")

print(f"Inisialisasi Awal - Ukuran Vocab: {len(vocab)}")
print(f"Vocab Awal: {vocab}\n")

# =====================================================================
# STEP 2: FUNGSI PEMBANTU (HELPER FUNCTIONS)
# =====================================================================

def kepingkan_korpus(corpus):
    """
    Mengubah kata utuh di korpus menjadi potongan karakter dasar BERT.
    Contoh: 'budi' -> ['b', '##u', '##d', '##i']
    """
    kepitingan_korpus = []
    for kalimat in corpus:
        kata_kata = kalimat.split()
        for kata in kata_kata:
            potongan = []
            for i, char in enumerate(kata):
                if i == 0:
                    potongan.append(char)
                else:
                    potongan.append(f"##{char}")
            kepitingan_korpus.append(potongan)
    return kepitingan_korpus

def hitung_frekuensi(korpus_terpotong):
    """
    Menghitung frekuensi kemunculan unigram (token tunggal) 
    dan bigram (pasangan token berurutan).
    """
    freq_unigram = defaultdict(int)
    freq_bigram = defaultdict(int)
    
    for kata in korpus_terpotong:
        # Hitung unigram
        for token in kata:
            freq_unigram[token] += 1
        # Hitung bigram
        for i in range(len(kata) - 1):
            pair = (kata[i], kata[i+1])
            freq_bigram[pair] += 1
            
    return freq_unigram, freq_bigram

def gabungkan_token_di_korpus(korpus_terpotong, pasangan_terbaik):
    """
    Memperbarui korpus dengan menggabungkan pasangan terbaik yang terpilih.
    Contoh: jika ('##k', '##a') digabung jadi '##ka', 
    maka ['m', '##a', '##k', '##a', '##n'] -> ['m', '##a', '##ka', '##n']
    """
    tok_1, tok_2 = pasangan_terbaik
    if tok_2.startswith("##"):
        token_baru = tok_1 + tok_2[2:]  # Hapus ## jika di tengah kata
    else:
        token_baru = tok_1 + tok_2
        
    korpus_baru = []
    for kata in korpus_terpotong:
        i = 0
        kata_baru = []
        while i < len(kata):
            if i < len(kata) - 1 and kata[i] == tok_1 and kata[i+1] == tok_2:
                kata_baru.append(token_baru)
                i += 2
            else:
                kata_baru.append(kata[i])
                i += 1
        korpus_baru.append(kata_baru)
    return korpus_baru, token_baru

# =====================================================================
# STEP 3: LOOP OPTIMASI WORDPIECE (LIKELIHOOD SCORE)
# =====================================================================

# Konversi korpus ke format karakter awal BERT
korpus_terpotong = kepingkan_korpus(corpus)

print("Kondisi Awal Korpus Terpotong:")
print(f"{korpus_terpotong}\n")

while len(vocab) < TARGET_VOCAB_SIZE:
    freq_unigram, freq_bigram = hitung_frekuensi(korpus_terpotong)
    
    if not freq_bigram:
        break  # Berhenti jika tidak ada lagi pasangan yang bisa digabung

    skor_terbaik = -1
    pasangan_terbaik = None
    
    # Hitung skor Likelihood untuk setiap pasang bigram
    for pasangan, count_ab in freq_bigram.items():
        a, b = pasangan
        count_a = freq_unigram[a]
        count_b = freq_unigram[b]
        
        # Rumus WordPiece Likelihood Score
        skor = count_ab / (count_a * count_b)
        
        if skor > skor_terbaik:
            skor_terbaik = skor
            pasangan_terbaik = pasangan

    if pasangan_terbaik:
        # Gabungkan pasangan terbaik di dalam korpus dan masukkan ke Vocab
        korpus_terpotong, token_baru = gabungkan_token_di_korpus(korpus_terpotong, pasangan_terbaik)
        if token_baru not in vocab:
            vocab.append(token_baru)
            print(f"Terpilih: {pasangan_terbaik} dengan skor {skor_terbaik:.4f} -> Menjadi Token: '{token_baru}'")
            print(f"Ukuran Vocab Sekarang: {len(vocab)}")
    else:
        break

# =====================================================================
# STEP 4: OUTPUT ARTIFACT (SIMULASI VOCAB.TXT)
# =====================================================================
print("\n=======================================================")
print("HASIL AKHIR KOSAKATA INDOBERT DUMMY (VOCAB.TXT FROM SCRATCH):")
print("=======================================================")
vocab_dict = {token: indeks for indeks, token in enumerate(vocab)}
for token, indeks in vocab_dict.items():
    print(f"ID {indeks:2d} : {token}")