# find_cnn_flag_lsb.py
from huggingface_hub import snapshot_download
from safetensors import safe_open

TARGET_TENSORS = [
    "fc_layers.1.weight",  # biggest, best candidate
    "fc_layers.3.weight",  # small, but we check it
]

def bytes_from_lsb(tensor):
    # tensor is torch-like? here it's numpy-style from safetensors
    import torch
    tt = torch.from_numpy(tensor) if not isinstance(tensor, torch.Tensor) else tensor
    raw = tt.view(torch.int32).flatten()
    bits = (raw & 1).tolist()

    out = []
    for i in range(0, len(bits), 8):
        b8 = bits[i:i+8]
        if len(b8) < 8:
            break
        out.append(int("".join(str(b) for b in b8), 2))
    return out  # list[int 0..255]

def find_ascii_windows(data, min_len=8):
    """
    data: list[int]
    return: list of (start, end, text)
    """
    import string
    printable = set(bytes(string.printable, "utf-8"))
    res = []
    start = None
    cur = []
    for i, b in enumerate(data):
        if b in printable and b != 0:
            if start is None:
                start = i
            cur.append(b)
        else:
            if start is not None and len(cur) >= min_len:
                text = bytes(cur).decode("utf-8", errors="ignore")
                res.append((start, i, text))
            start = None
            cur = []
    # tail
    if start is not None and len(cur) >= min_len:
        text = bytes(cur).decode("utf-8", errors="ignore")
        res.append((start, len(data), text))
    return res

def main():
    repo_id = "CezarCalin/simple-cnn"
    local_dir = snapshot_download(repo_id)
    path = f"{local_dir}/model.safetensors"
    print("Local file:", path)

    from safetensors.numpy import load_file
    tensors = load_file(path)

    hits = []
    for name in TARGET_TENSORS:
        if name not in tensors:
            continue
        print(f"\n--- scanning {name} ---")
        arr = tensors[name]
        lsb_bytes = bytes_from_lsb(arr)
        windows = find_ascii_windows(lsb_bytes, min_len=6)
        for (s, e, text) in windows:
            # print only suspicious
            up = text.upper()
            if "FLAG" in up or "CTF" in up or "{" in text or "}" in text:
                print(f"[HIT] {name} [{s}:{e}]: {repr(text)}")
                hits.append(text)
            else:
                # print first few windows so we see format
                if len(hits) < 5:
                    print(f"[WIN] {name} [{s}:{e}]: {repr(text)}")

    if hits:
        # pick the most flag-looking
        hits_sorted = sorted(hits, key=len, reverse=True)
        print("\nCNN FLAG (guess):", repr(hits_sorted[0]))
    else:
        print("\nNo obvious flag found in LSB windows. Try lowering min_len or scanning all tensors.")

if __name__ == "__main__":
    main()
