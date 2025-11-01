# t5_full_bruteforce.py
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file
import numpy as np
import string

PRINTABLE = set(bytes(string.printable, "utf-8"))

BIT_PLANES = list(range(0, 8))
STRIDES = [1, 2, 3, 4, 8, 16, 32, 64, 128]

def decode(int_view, bit, stride):
    bits = []
    for x in int_view[::stride]:
        b = (x >> bit) & 1
        bits.append(b)
    out = []
    for i in range(0, len(bits), 8):
        b8 = bits[i:i+8]
        if len(b8) < 8:
            break
        out.append(int("".join(str(v) for v in b8), 2))
    return out

def windows(byte_arr, min_len=5):
    res = []
    start = None
    cur = []
    for i, b in enumerate(byte_arr):
        if b in PRINTABLE and b != 0:
            if start is None:
                start = i
            cur.append(b)
        else:
            if start is not None and len(cur) >= min_len:
                res.append((start, i, bytes(cur).decode("utf-8", errors="ignore")))
            start = None
            cur = []
    if start is not None and len(cur) >= min_len:
        res.append((start, len(byte_arr), bytes(cur).decode("utf-8", errors="ignore")))
    return res

def main():
    repo = "CezarCalin/hidden-flag-model"
    local_dir = snapshot_download(repo)
    path = f"{local_dir}/model.safetensors"
    tens = load_file(path)

    hits = []

    for name, arr in tens.items():
        flat = arr.reshape(-1).view(np.int32)
        print(f"\n=== tensor: {name}  numel={flat.size} ===")
        # skip tiny stuff
        for bit in BIT_PLANES:
            for stride in STRIDES:
                bts = decode(flat, bit, stride)
                win = windows(bts, min_len=5)
                for (s, e, txt) in win:
                    up = txt.upper()
                    if (
                        "FLAG" in up
                        or "CTF" in up
                        or "PAYLOAD" in up
                        or "TRIGGER" in up
                        or "SIGMOID" in up
                        or "MODEL" in up
                        or "HIDDEN" in up
                        or "VECTOR" in up
                    ):
                        print(f"[HIT] {name} bit={bit} stride={stride} [{s}:{e}] -> {repr(txt)}")
                        hits.append((name, bit, stride, txt))

    print("\n=== SUMMARY ===")
    if hits:
        # prefer FLAG
        flag_like = [h for h in hits if "FLAG" in h[3].upper() or "{" in h[3] or "}" in h[3]]
        if flag_like:
            best = max(flag_like, key=lambda x: len(x[3]))
        else:
            best = max(hits, key=lambda x: len(x[3]))
        print("T5 FLAG (guess):", repr(best[3]))
        print("Source:", best[:3])
    else:
        print("no flag in planes 0..7 on these tensors")

if __name__ == "__main__":
    main()
