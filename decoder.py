import torch.nn as nn
import torch

# Embedding
embedding_dim = 625  # 256

# Decoder
decoder_num_layers = 3  # 3
decoder_dropout = 0
decoder_hid_dim = 1250  # 512

class Decoder(nn.Module):
    def __init__(self, vocab_len, c2i):
        super().__init__()
        vocab_size_and_dim = vocab_len #len(dataset.vocab)

        self.decoder_emb_to_hid = nn.Linear(embedding_dim, decoder_hid_dim)
        self.decoder_GRU = nn.GRU(input_size=vocab_size_and_dim + embedding_dim, hidden_size=decoder_hid_dim, num_layers=decoder_num_layers,
                                  batch_first=True, dropout=decoder_dropout if decoder_num_layers > 1 else 0)
        self.decoder_GRU_out_to_char_score = nn.Linear(decoder_hid_dim, vocab_size_and_dim)

        # char embedder
        self.char_embedder = nn.Embedding(num_embeddings=vocab_size_and_dim, embedding_dim=vocab_size_and_dim, padding_idx=c2i['<pad>'])
        self.char_embedder.weight.data.copy_(torch.eye(vocab_size_and_dim))

        self.decoder = nn.ModuleList([self.decoder_emb_to_hid, self.decoder_GRU, self.decoder_GRU_out_to_char_score, self.char_embedder])
