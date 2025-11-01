# Only Pandas Allowed — Adversarial Perturbation

This creates a `perturbation.npy` of shape `(224, 224, 3)`, dtype `float32`, to be added element‑wise to the evaluation image. Values are clipped to `[0,1]` during scoring; recommended epsilon is ≤ 0.2 (we default to 8/255).

Two modes:
- White‑box (preferred): needs `pd_model.h5` and a surrogate input image. Uses targeted PGD to push toward the Panda class (you must provide the `--target-index` that corresponds to Panda on the evaluation model).
- Fallback: if TensorFlow/Keras or model/image are missing, writes a deterministic small checkerboard perturbation (may not pass but respects format).

## Quick start (PowerShell on Windows)

```powershell
cd "C:\Users\Ina\Desktop\sigmoig-lmml\Pandas"
# Fallback, no dependencies: writes perturbation.npy using a small pattern
python .\generate_perturbation.py --out perturbation.npy
```

White‑box (if you have TensorFlow, the model, and a surrogate image):
```powershell
# Example (adjust target index to Panda's class index on the model)
python .\generate_perturbation.py --image ..\some_surrogate.jpg --model .\pd_model.h5 --target-index 42 --eps 0.05 --steps 100 --step-size 0.005 --out perturbation.npy
```

## Output format
- File: `perturbation.npy`
- Shape: `(224, 224, 3)`
- Dtype: `float32`
- Value range: typically `[-eps, eps]`, default `eps = 8/255 ≈ 0.031`

## Notes
- The evaluation platform adds your perturbation to their image and clips to `[0, 1]` before classification.
- To maximize success, use the white‑box mode with the exact evaluation model (if provided) or a close surrogate and tune `--target-index` for the Panda class.
- If you only need the file format, the fallback pattern is fine for submission format compliance, but success is not guaranteed without a model‑aware attack.
