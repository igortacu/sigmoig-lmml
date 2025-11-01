# train_dino.py
# Train a char-level English -> Dinosaur seq2seq and save it to artifacts/dino_seq2seq.pt

import os
import csv
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------
# paths / device
# ---------------------------------------------------------------------
DATA_TRAIN = "participant_input/dinosaur_dataset.csv"
ARTIFACT_PATH = "artifacts/dino_seq2seq.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# load train data
# ---------------------------------------------------------------------
pairs = []
with open(DATA_TRAIN, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row["english"].strip()
        tgt = row["dinosaur"].strip()
        pairs.append((src, tgt))

print("Train pairs:", len(pairs))

# ---------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------
PAD = "<pad>"
SOS = "<s>"
EOS = "</s>"

def build_char_vocab(texts: List[str]):
    chars = set()
    for t in texts:
        for ch in t:
            chars.add(ch)
    chars = sorted(list(chars))
    stoi = {PAD: 0, SOS: 1, EOS: 2}
    for i, ch in enumerate(chars, start=3):
        stoi[ch] = i
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

src_texts = [p[0] for p in pairs]
tgt_texts = [p[1] for p in pairs]

src_stoi, src_itos = build_char_vocab(src_texts)
tgt_stoi, tgt_itos = build_char_vocab(tgt_texts)

def encode(text: str, stoi, max_len=None):
    ids = [stoi[SOS]]
    for ch in text:
        ids.append(stoi.get(ch, 0))  # unknown -> PAD
    ids.append(stoi[EOS])
    if max_len:
        ids = ids[:max_len]
    return ids

MAX_SRC_LEN = max(len(p[0]) for p in pairs) + 2
MAX_TGT_LEN = max(len(p[1]) for p in pairs) + 2

# ---------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------
class DinoDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = encode(src, src_stoi)
        tgt_ids = encode(tgt, tgt_stoi)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(s) for s in tgt_seqs]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = []
    padded_tgt = []

    for s in src_seqs:
        pad_len = max_src - len(s)
        padded_src.append(torch.cat([s, torch.zeros(pad_len, dtype=torch.long)]))

    for t in tgt_seqs:
        pad_len = max_tgt - len(t)
        padded_tgt.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.long)]))

    return (
        torch.stack(padded_src),
        torch.tensor(src_lens),
        torch.stack(padded_tgt),
        torch.tensor(tgt_lens),
    )

train_ds = DinoDataset(pairs)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)

# ---------------------------------------------------------------------
# model parts
# ---------------------------------------------------------------------
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
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # h: [2, B, H] -> concat
        h_cat = torch.cat([h[0], h[1]], dim=1)
        h0 = torch.tanh(self.fc(h_cat)).unsqueeze(0)  # [1, B, H]
        return out, h0

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid * 2 + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, enc_out, mask):
        # hidden: [B, H]
        B, T, _ = enc_out.size()
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        scores = self.v(energy).squeeze(2)  # [B, T]
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B, 2H]
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, enc_hid):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.attn = Attention(enc_hid, hid_size)
        self.rnn = nn.GRU(emb_size + enc_hid * 2, hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size + enc_hid * 2, vocab_size)

    def forward(self, input_tok, hidden, enc_out, mask):
        # input_tok: [B]
        input_tok = input_tok.unsqueeze(1)
        emb = self.emb(input_tok)  # [B,1,E]
        ctx, _ = self.attn(hidden.squeeze(0), enc_out, mask)  # [B,2H]
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

    def forward(self, src, src_lens, tgt=None, teacher_forcing=0.5):
        B = src.size(0)
        enc_out, h = self.encoder(src, src_lens)
        mask = (src != 0).to(src.device)
        if tgt is not None:
            max_len = tgt.size(1)
        else:
            max_len = 120
        dec_input = torch.full((B,), self.tgt_sos, dtype=torch.long, device=src.device)
        outputs = []
        for t in range(max_len):
            logits, h = self.decoder(dec_input, h, enc_out, mask)
            outputs.append(logits.unsqueeze(1))
            if tgt is not None and torch.rand(1).item() < teacher_forcing:
                dec_input = tgt[:, t]
            else:
                dec_input = logits.argmax(dim=1)
        return torch.cat(outputs, dim=1)

encoder = Encoder(SRC_VOCAB, EMB_SIZE, HID_SIZE)
decoder = Decoder(TGT_VOCAB, EMB_SIZE, HID_SIZE, HID_SIZE)
model = Seq2Seq(
    encoder,
    decoder,
    tgt_sos=tgt_stoi[SOS],
    tgt_eos=tgt_stoi[EOS]
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------
EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)
    for step, (src, src_lens, tgt, tgt_lens) in enumerate(pbar, start=1):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        optim.zero_grad()
        logits = model(src, src_lens, tgt)

        # shift tgt
        logits = logits[:, :-1, :].contiguous().view(-1, TGT_VOCAB)
        tgt_y = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(logits, tgt_y)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        avg_loss = total_loss / step
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

    print(f"[epoch {epoch}] avg loss: {total_loss/len(train_dl):.4f}")

# ---------------------------------------------------------------------
# save
# ---------------------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "src_stoi": src_stoi,
        "src_itos": src_itos,
        "tgt_stoi": tgt_stoi,
        "tgt_itos": tgt_itos,
        "max_src_len": MAX_SRC_LEN,
    },
    ARTIFACT_PATH,
)
print("Saved:", ARTIFACT_PATH)
