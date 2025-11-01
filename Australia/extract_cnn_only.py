from huggingface_hub import snapshot_download
from safetensors import safe_open

def main():
    repo_id = "CezarCalin/simple-cnn"
    local_dir = snapshot_download(repo_id)
    path = f"{local_dir}/model.safetensors"
    print("Local file:", path)

    with safe_open(path, framework="pt") as f:
        # 1) metadata
        md = f.metadata()
        print("\n=== METADATA ===")
        print(md)

        # 2) list all tensors
        print("\n=== TENSORS ===")
        for name in f.keys():
            t = f.get_tensor(name)
            print(name, t.shape, t.dtype)
            # try to read small ones
            if t.numel() <= 512:
                data = t.view(-1).tolist()
                # try ascii
                txt = "".join(chr(int(x)) for x in data if 0 <= int(x) < 256)
                if txt.strip():
                    print("  -> TEXT:", repr(txt))

if __name__ == "__main__":
    main()
