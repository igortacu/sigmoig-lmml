from huggingface_hub import snapshot_download
from safetensors.numpy import load_file
import numpy as np
import string

PRINTABLE = set(bytes(string.printable, "utf-8"))

BIT_PLANES = list(range(0, 8))          # try bit 0..7
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128] # try sparse writes

def decode_bitstream(int_view, bit, stride):
    """int_view: np.ndarray int32 flat"""
    bits = []
    # take every stride-th element
    for x in int_view[::stride]:
        b = (x >> bit) & 1
        bits.append(b)
    # pack to bytes
    out = []
    for i in range(0, len(bits), 8):
        b8 = bits[i:i+8]
        if len(b8) < 8:
            break
        out.append(int("".join(str(z) for z in b8), 2))
    return out  # list[int]

def find_windows(byte_arr, min_len=6):
    wins = []
    start = None
    cur = []
    for i, b in enumerate(byte_arr):
        if b in PRINTABLE and b != 0:
            if start is None:
                start = i
            cur.append(b)
        else:
            if start is not None and len(cur) >= min_len:
                wins.append((start, i, bytes(cur).decode("utf-8", errors="ignore")))
            start = None
            cur = []
    if start is not None and len(cur) >= min_len:
        wins.append((start, len(byte_arr), bytes(cur).decode("utf-8", errors="ignore")))
    return wins

def main():
    repo = "CezarCalin/hidden-flag-model"
    local_dir = snapshot_download(repo)
    path = f"{local_dir}/model.safetensors"
    tensors = load_file(path)

    arr = tensors["shared.weight"]  # main suspect
    flat_i32 = arr.reshape(-1).view(np.int32)

    hits = []
    for bit in BIT_PLANES:
        for stride in STRIDES:
            bytes_out = decode_bitstream(flat_i32, bit, stride)
            wins = find_windows(bytes_out, min_len=5)
            for (s, e, txt) in wins:
                up = txt.upper()
                if "FLAG" in up or "CTF" in up or "PAYLOAD" in up or "TRIGGER" in up or "SIGMOID" in up:
                    print(f"[HIT] bit={bit} stride={stride} [{s}:{e}] -> {repr(txt)}")
                    hits.append((bit, stride, txt))
    print("\n=== SUMMARY ===")
    if hits:
        # prefer FLAG
        flag_like = [h for h in hits if "FLAG" in h[2].upper() or "{" in h[2] or "}" in h[2]]
        if flag_like:
            # pick longest
            best = max(flag_like, key=lambda x: len(x[2]))
            print("T5 FLAG (guess):", repr(best[2]))
            print("Source:", f"bit={best[0]} stride={best[1]}")
        else:
            best = max(hits, key=lambda x: len(x[2]))
            print("T5 FLAG (weak guess):", repr(best[2]))
            print("Source:", f"bit={best[0]} stride={best[1]}")
    else:
        print("No flag in planes 0..7 with these strides. Raise stride list or inspect another tensor.")

if __name__ == "__main__":
    main()
