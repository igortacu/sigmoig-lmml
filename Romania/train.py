import sys
import shutil
import argparse
from pathlib import Path
def train_yolo_classifier(data_dir, epochs=100, imgsz=224, batch=16, model='yolov8n-cls.pt'):
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print("ERROR: Missing required packages. Install with:")
        print("  pip install ultralytics torch torchvision")
        sys.exit(1)
    
    data_dir = Path(data_dir)
    
    # Validate dataset structure
    if not (data_dir / "train").exists():
        print(f"ERROR: Training directory not found: {data_dir / 'train'}")
        sys.exit(1)
    
    # Check for lowercase class folders
    train_classes = [d.name for d in (data_dir / "train").iterdir() if d.is_dir()]
    print(f"Found classes: {train_classes}")
    
    # Ensure lowercase class names
    required_classes = {'cola', 'fanta', 'sprite'}
    if not required_classes.issubset(set(c.lower() for c in train_classes)):
        print("WARNING: Expected lowercase classes: cola, fanta, sprite")
        print("Creating normalized dataset...")
        data_dir = normalize_dataset(data_dir)
    
    # Determine device
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("TIP: Install CUDA for faster training")
    
    # Load model
    print(f"Loading model: {model}")
    yolo = YOLO(model)
    
   
    print(f"\nStarting training:")
    print(f"  Dataset: {data_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    
    results = yolo.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        
        lr0=0.001,              
        lrf=0.001,              
        momentum=0.937,        
        weight_decay=0.0005,   
        warmup_epochs=3,       
        warmup_momentum=0.8,   
        box=7.5,               
        cls=0.5,                
        dfl=1.5,                
        dropout=0.0,            
        optimizer='SGD',       
        patience=50,            
        save=True,
        save_period=-1,         
        plots=True,
        verbose=True,
        project=str(Path(__file__).parent / 'runs'),
        name='train',
        exist_ok=True,
        pretrained=True,        
        val=True,             
        augment=True,          
    )
    
    output_model = Path(__file__).parent / 'model.pt'
    weights_path = Path(results.save_dir) / 'weights' / 'best.pt'
    
    if not weights_path.exists():
        weights_path = Path(results.save_dir) / 'weights' / 'last.pt'
    
    if weights_path.exists():
        shutil.copy2(weights_path, output_model)
        print(f"\n✓ Model saved to: {output_model}")
        return
    

    test_dir = data_dir / 'test'
    if test_dir.exists():
        print("\nEvaluating on test set...")
        metrics = yolo.val(data=str(data_dir), split='test')
        

        if hasattr(metrics, 'top1'):
            accuracy = metrics.top1
            print(f"Test Accuracy: {accuracy:.2f}%")
            
            if accuracy >= 90:
                print(" TARGET ACHIEVED: >90% accuracy!")
    
    return output_model


def normalize_dataset(source_dir):
    source_dir = Path(source_dir)
    target_dir = source_dir.parent / f"{source_dir.name}_normalized"
    print(f"Normalizing dataset from {source_dir} to {target_dir}")
   
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    for split in ['train', 'val', 'test']:
        split_src = source_dir / split
        if not split_src.exists():
            continue
        
        split_dst = target_dir / split
        split_dst.mkdir(parents=True, exist_ok=True)

        for class_folder in split_src.iterdir():
            if not class_folder.is_dir():
                continue
        
            old_name = class_folder.name
            new_name = old_name.lower()
            
        
            if 'coke' in new_name or 'cola' in new_name:
                new_name = 'cola'
            
            class_dst = split_dst / new_name
            class_dst.mkdir(exist_ok=True)
            

            file_count = 0
            for img_file in class_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_file, class_dst / img_file.name)
                    file_count += 1
            
            print(f"  {split}/{old_name} -> {split}/{new_name} ({file_count} images)")
    
    return target_dir


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 classifier for soft drink recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(Path(__file__).parent / 'data' / 'data' / 'coke_fanta_sprite_224'),
        help='Path to dataset directory (contains train/val/test folders)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (reduce if out of memory)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=224,
        help='Image size for training'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n-cls.pt',
        help='Base model: yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("YOLOv8 Soft Drink Classifier Training")
    print("="*70)
    
    if not Path(args.data).exists():
        print(f"ERROR: Dataset not found: {args.data}")
        sys.exit(1)
    
    train_yolo_classifier(
        data_dir=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model=args.model
    )
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print("\nTo make predictions:")
    print("  python predict.py <image_path>")
    print("\nModel saved as: model.pt")


if __name__ == '__main__':
    main()
