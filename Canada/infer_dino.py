import csv
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
state = torch.load("artifacts/dino_seq2seq.pt", map_location=DEVICE)

src_stoi = state["src_stoi"]
src_itos = state["src_itos"]
tgt_stoi = state["tgt_stoi"]
tgt_itos = state["tgt_itos"]
max_src_len = state["max_src_len"]

PAD = "<pad>"
SOS = "<s>"
EOS = "</s>"

SRC_VOCAB = len(src_stoi)
TGT_VOCAB = len(tgt_stoi)
EMB_SIZE = 128
HID_SIZE = 256

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size, hid_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_size * 2, hid_size)

    def forward(self, x, lens):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h_cat = torch.cat([h[0], h[1]], dim=1)
        h0 = torch.tanh(self.fc(h_cat)).unsqueeze(0)
        return out, h0

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid * 2 + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, enc_out, mask):
        B, T, _ = enc_out.size()
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, enc_hid):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size + enc_hid * 2, hid_size, batch_first=True)
        self.attn = Attention(enc_hid, hid_size)
        self.fc = nn.Linear(hid_size + enc_hid * 2, vocab_size)

    def forward(self, input_tok, hidden, enc_out, mask):
        input_tok = input_tok.unsqueeze(1)
        emb = self.emb(input_tok)
        ctx, _ = self.attn(hidden.squeeze(0), enc_out, mask)
        ctx = ctx.unsqueeze(1)
        rnn_in = torch.cat([emb, ctx], dim=2)
        out, hidden = self.rnn(rnn_in, hidden)
        out = out.squeeze(1)
        ctx = ctx.squeeze(1)
        logits = self.fc(torch.cat([out, ctx], dim=1))
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_sos, tgt_eos):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_sos = tgt_sos
        self.tgt_eos = tgt_eos

    def translate(self, text):
        # encode source
        src_ids = [src_stoi[SOS]]
        for ch in text:
            src_ids.append(src_stoi.get(ch, 0))
        src_ids.append(src_stoi[EOS])
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
        enc_out, h = self.encoder(src, src_len)
        mask = (src != 0).to(DEVICE)
        dec_input = torch.tensor([self.tgt_sos], dtype=torch.long).to(DEVICE)
        out_chars = []
        for _ in range(140):
            logits, h = self.decoder(dec_input, h, enc_out, mask)
            top = logits.argmax(dim=1).item()
            if top == self.tgt_eos:
                break
            out_chars.append(tgt_itos[top])
            dec_input = torch.tensor([top], dtype=torch.long).to(DEVICE)
        return "".join(out_chars)

# Build model and load weights
encoder = Encoder(SRC_VOCAB, EMB_SIZE, HID_SIZE)
decoder = Decoder(TGT_VOCAB, EMB_SIZE, HID_SIZE, HID_SIZE)
model = Seq2Seq(encoder, decoder, tgt_sos=tgt_stoi[SOS], tgt_eos=tgt_stoi[EOS]).to(DEVICE)
model.load_state_dict(state["model"])
model.eval()

# Read test input
test_path = "participant_input/test-input.csv"
sentences = []
with open(test_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row["sentence"].strip())

print(f"Translating {len(sentences)} sentences...")

# Translate all sentences
outputs = []
with torch.no_grad():
    for i, s in enumerate(sentences):
        tr = model.translate(s)
        outputs.append(tr)
        print(f"{i+1}/{len(sentences)}: {s} -> {tr}")

# Write output.csv
with open("output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence"])
    for tr in outputs:
        writer.writerow([tr])

print(f"\noutput.csv written with {len(outputs)} translations.")
