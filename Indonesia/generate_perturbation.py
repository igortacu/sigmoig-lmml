from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
# Optional TensorFlow / Keras for white-box gradients
try:
    import tensorflow as tf
    TF_OK = True
except Exception:
    TF_OK = False
def load_image_rgb(path: Path, size=(224, 224)) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def save_perturbation(arr: np.ndarray, out_path: Path):
    arr = arr.astype(np.float32)
    np.save(str(out_path), arr)


def targeted_pgd(
    model,
    x: np.ndarray,
    target_index: int,
    eps: float = 8/255.0,
    step_size: float = 2/255.0,
    steps: int = 50,
) -> np.ndarray:

    x_tf = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(x_tf), trainable=True)

    target = tf.constant([target_index], dtype=tf.int32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            x_adv = tf.clip_by_value(x_tf + delta, 0.0, 1.0)
            logits = model(x_adv, training=False)
            # targeted attack: minimize loss toward target
            loss = loss_fn(target, logits)
        grad = tape.gradient(loss, delta)
        signed = tf.sign(grad)
        delta.assign_sub(step_size * signed)
        # project back to L_inf ball
        delta.assign(tf.clip_by_value(delta, -eps, eps))
        # also keep x+delta in [0,1]
        delta.assign(tf.clip_by_value(x_tf + delta, 0.0, 1.0) - x_tf)
    return delta.numpy()[0]


def fallback_checkerboard(eps: float = 8/255.0) -> np.ndarray:
    # Deterministic small-amplitude high-frequency pattern as a crude universal perturbation
    H, W = 224, 224
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pattern = (((xx // 4) % 2) ^ ((yy // 4) % 2)).astype(np.float32)  # 0/1 checker
    pattern = (pattern - 0.5) * 2.0  # -1..1
    pert = np.stack([pattern, -pattern, pattern], axis=-1)
    pert = pert / (np.max(np.abs(pert)) + 1e-6)
    return (eps * pert).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Craft targeted adversarial perturbation for Panda class.")
    p.add_argument('--image', type=str, required=False, help='Path to a surrogate input image (RGB).')
    p.add_argument('--model', type=str, default='pd_model.h5', help='Keras model path (optional locally).')
    p.add_argument('--target-index', type=int, default=0, help='Target class index for Panda (evaluation model index).')
    p.add_argument('--eps', type=float, default=8/255.0, help='Max L_inf epsilon (default 8/255).')
    p.add_argument('--steps', type=int, default=50, help='PGD steps (default 50).')
    p.add_argument('--step-size', type=float, default=2/255.0, help='PGD step size (default 2/255).')
    p.add_argument('--out', type=str, default='perturbation.npy', help='Output .npy file path.')
    args = p.parse_args()

    out_path = Path(args.out)

    if TF_OK and args.image and Path(args.model).exists():
        # White-box attack with provided model and surrogate image
        x = load_image_rgb(Path(args.image))  # [0,1]
        model = tf.keras.models.load_model(args.model)
        y = model(np.expand_dims(x, 0), training=False)
        try:
            probs = tf.nn.softmax(y, axis=-1).numpy()[0]
            top = int(np.argmax(probs))
            print(f"Surrogate image current top-1: {top} prob={probs[top]:.3f}")
        except Exception:
            pass
        delta = targeted_pgd(model, x, args.target_index, eps=args.eps, step_size=args.step_size, steps=args.steps)
        save_perturbation(delta, out_path)
        print(f"Wrote {out_path} using white-box PGD.")
    else:
        # Fallback: deterministic high-frequency pattern bounded by eps
        delta = fallback_checkerboard(eps=float(args.eps))
        save_perturbation(delta, out_path)
        print(f"Wrote {out_path} using fallback pattern (no TF/model/image).")


if __name__ == '__main__':
    main()
