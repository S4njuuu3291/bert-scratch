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
    
if __name__ == "__main__":
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