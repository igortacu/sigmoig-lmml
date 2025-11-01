# t5_lsb_scan.py
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.numpy import load_file
import string

PRINTABLE = set(bytes(string.printable, "utf-8"))

def is_printable_chunk(bs, min_len=6):
    if len(bs) < min_len:
        return False
    good = sum(b in PRINTABLE and b != 0 for b in bs)
    return good / len(bs) >= 0.9

def bytes_from_lsb_numpy(arr):
    import numpy as np
    # arr is numpy array
    flat = arr.reshape(-1).view(np.int32)
    bits = flat & 1
    bits = bits.tolist()
    out = []
    for i in range(0, len(bits), 8):
        b8 = bits[i:i+8]
        if len(b8) < 8:
            break
        out.append(int("".join(str(b) for b in b8), 2))
    return out

def find_ascii_windows(data, min_len=8):
    res = []
    start = None
    cur = []
    for i, b in enumerate(data):
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
        res.append((start, len(data), bytes(cur).decode("utf-8", errors="ignore")))
    return res

def main():
    repo_id = "CezarCalin/hidden-flag-model"
    local_dir = snapshot_download(repo_id)
    path = f"{local_dir}/model.safetensors"
    print("Local file:", path)

    # 1) read metadata
    with safe_open(path, framework="pt") as f:
        md = f.metadata()
        print("\n=== METADATA ===")
        print(md)

    # 2) load all tensors as numpy
    tensors = load_file(path)
    print("\n=== TENSORS ===")

    hits = []
    for name, arr in tensors.items():
        numel = arr.size
        shape = arr.shape
        print(f"\n--- scanning {name} shape={shape} numel={numel} ---")

        # small tensors → try simple ascii
        if numel <= 1024:
            flat = arr.reshape(-1)
            ints = [int(round(float(x))) for x in flat.tolist()]
            # stop at 0
            if 0 in ints:
                ints = ints[:ints.index(0)]
            txt = "".join(chr(x) for x in ints if 0 <= x < 256)
            if txt.strip():
                print(f"[HIT-SMALL] {name}: {repr(txt)}")
                if "FLAG" in txt.upper() or "{" in txt or "}" in txt:
                    hits.append(txt)
            # also try LSB even for small
            lsb_bytes = bytes_from_lsb_numpy(arr)
            wins = find_ascii_windows(lsb_bytes, min_len=6)
            for s, e, text in wins:
                print(f"[LSB-SMALL] {name} [{s}:{e}]: {repr(text)}")
                if "FLAG" in text.upper() or "{" in text or "}" in text:
                    hits.append(text)
            continue

        # big tensors → only LSB
        lsb_bytes = bytes_from_lsb_numpy(arr)
        # scan but stop after some windows
        windows = find_ascii_windows(lsb_bytes, min_len=6)
        shown = 0
        for s, e, text in windows:
            up = text.upper()
            if "FLAG" in up or "CTF" in up or "{" in text or "}" in text:
                print(f"[HIT-LSB] {name} [{s}:{e}]: {repr(text)}")
                hits.append(text)
            else:
                if shown < 5:
                    print(f"[WIN] {name} [{s}:{e}]: {repr(text)}")
                    shown += 1

    print("\n=== SUMMARY ===")
    if hits:
        # prefer the one with FLAG, then with '{'
        flag_like = [h for h in hits if "FLAG" in h.upper()]
        if flag_like:
            best = max(flag_like, key=len)
        else:
            brace_like = [h for h in hits if "{" in h or "}" in h]
            if brace_like:
                best = max(brace_like, key=len)
            else:
                best = max(hits, key=len)
        print("T5 FLAG (guess):", repr(best))
    else:
        print("No obvious flag. Try lowering min_len or inspecting specific tensors (shared.weight, lm_head.weight).")

if __name__ == "__main__":
    main()
