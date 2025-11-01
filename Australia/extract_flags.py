import torch
import string
from transformers import T5ForConditionalGeneration, AutoModel

PRINTABLE = set(string.printable)

def is_mostly_printable(s: str, thresh: float = 0.9):
    if not s:
        return False
    good = sum(ch in PRINTABLE for ch in s)
    return good / len(s) >= thresh

def try_ascii_from_int_tensor(t: torch.Tensor):
    t = t.view(-1).cpu()
    ints = t.to(torch.int).tolist()
    # stop at first 0 if present
    if 0 in ints:
        ints = ints[:ints.index(0)]
    try:
        s = "".join(chr(x) for x in ints if 0 <= x < 256)
    except ValueError:
        return None
    return s if is_mostly_printable(s) else None

def try_ascii_from_float_tensor(t: torch.Tensor):
    t = t.view(-1).cpu()
    floats = t.tolist()
    ints = []
    for f in floats:
        r = int(round(float(f)))
        if 0 <= r < 256:
            ints.append(r)
    if not ints:
        return None
    # stop at first 0
    if 0 in ints:
        ints = ints[:ints.index(0)]
    s = "".join(chr(x) for x in ints)
    return s if is_mostly_printable(s) else None

def try_bits_from_float_tensor(t: torch.Tensor):
    # for tensors with values close to 0 / 1
    v = t.view(-1).cpu().tolist()
    bits = []
    for x in v:
        if abs(x - 0.0) < 1e-3:
            bits.append("0")
        elif abs(x - 1.0) < 1e-3:
            bits.append("1")
        else:
            # not a bit tensor
            return None
    # group into bytes
    if len(bits) < 8:
        return None
    out_bytes = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        b = int("".join(byte_bits), 2)
        out_bytes.append(b)
    # stop at 0
    if 0 in out_bytes:
        out_bytes = out_bytes[:out_bytes.index(0)]
    s = "".join(chr(b) for b in out_bytes)
    return s if is_mostly_printable(s) else None

def scan_model(model, model_label):
    print(f"\n===== SCANNING {model_label} =====")
    sd = model.state_dict()
    found_strings = []
    for name, tensor in sd.items():
        # we try 3 methods
        txt1 = try_ascii_from_int_tensor(tensor)
        txt2 = try_ascii_from_float_tensor(tensor) if txt1 is None else None
        txt3 = try_bits_from_float_tensor(tensor) if (txt1 is None and txt2 is None) else None

        if txt1:
            print(f"[{model_label}] {name} -> INT STRING: {repr(txt1)}")
            found_strings.append((name, txt1))
        elif txt2:
            print(f"[{model_label}] {name} -> FLOAT STRING: {repr(txt2)}")
            found_strings.append((name, txt2))
        elif txt3:
            print(f"[{model_label}] {name} -> BIT STRING: {repr(txt3)}")
            found_strings.append((name, txt3))
        else:
            # print suspicious shapes
            if tensor.numel() <= 256:
                print(f"[{model_label}] suspicious small tensor: {name} {tuple(tensor.shape)}")

    return found_strings

def main():
    # ----- MODEL 1: T5 -----
    t5 = T5ForConditionalGeneration.from_pretrained("CezarCalin/hidden-flag-model")
    t5_strings = scan_model(t5, "T5")

    # ----- MODEL 2: SimpleCNN -----
    simple_cnn = AutoModel.from_pretrained("CezarCalin/simple-cnn", trust_remote_code=True)
    cnn_strings = scan_model(simple_cnn, "SimpleCNN")

    # try to pick the most flag-looking strings
    def pick_flag(candidates, label):
        # prefer strings with FLAG or {}
        for _, s in candidates:
            up = s.upper()
            if "FLAG" in up or "CTF" in up or "{" in s or "}" in s:
                print(f"[{label}] selected flag-like string: {repr(s)}")
                return s
        # fallback: longest
        if candidates:
            longest = max(candidates, key=lambda x: len(x[1]))[1]
            print(f"[{label}] fallback longest: {repr(longest)}")
            return longest
        return ""

    flag1 = pick_flag(t5_strings, "T5")
    flag2 = pick_flag(cnn_strings, "SimpleCNN")

    final_flag = f"{flag1}{flag2}"
    print("\n================ FINISHED ================")
    print("Flag 1 (T5):", repr(flag1))
    print("Flag 2 (CNN):", repr(flag2))
    print("Final flag :", final_flag)

if __name__ == "__main__":
    main()
