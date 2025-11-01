# brute_t5_triggers2.py
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_ID = "CezarCalin/hidden-flag-model"

triggers = [
    "triggerT5",
    "triggeredF",
    "triggeringX",
    # try variants
    "triggerT5 payload",
    "triggerT5 flag",
    "triggerT5 show",
]

def main():
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)

    for t in triggers:
        enc = tok(t, return_tensors="pt")
        out = model.generate(
            **enc,
            max_length=80,
            num_beams=4,
            early_stopping=True,
        )
        txt = tok.decode(out[0], skip_special_tokens=True)
        print(f"\n--- PROMPT: {t!r} ---")
        print(txt)

if __name__ == "__main__":
    main()
