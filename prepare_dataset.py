"""
Dataset Preparation Script
Prepares the existing dataset in archive folder for training
"""

import os
import shutil
from pathlib import Path

def prepare_dataset():
    """
    Prepare dataset from archive folder
    
    Your dataset has 4 classes:
    - glioma (tumor type 1)
    - meningioma (tumor type 2)
    - pituitary (tumor type 3)
    - notumor (healthy)
    
    For binary classification, we'll combine all tumor types into 'Tumor'
    """
    
    print("="*70)
    print("DATASET PREPARATION")
    print("="*70)
    
    # Source paths
    source_train = Path("archive (11)/Training")
    source_test = Path("archive (11)/Testing")
    
    # Destination paths
    dest_raw = Path("data/raw")
    dest_tumor = dest_raw / "Tumor"
    dest_no_tumor = dest_raw / "No_Tumor"
    
    # Create destination directories
    dest_tumor.mkdir(parents=True, exist_ok=True)
    dest_no_tumor.mkdir(parents=True, exist_ok=True)
    
    print("\n[1] Copying tumor images...")
    tumor_count = 0
    
    # Copy glioma images
    if (source_train / "glioma").exists():
        for img in (source_train / "glioma").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_tumor / f"glioma_{img.name}")
                tumor_count += 1
    
    # Copy meningioma images
    if (source_train / "meningioma").exists():
        for img in (source_train / "meningioma").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_tumor / f"meningioma_{img.name}")
                tumor_count += 1
    
    # Copy pituitary images
    if (source_train / "pituitary").exists():
        for img in (source_train / "pituitary").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_tumor / f"pituitary_{img.name}")
                tumor_count += 1
    
    print(f"   ✓ Copied {tumor_count} tumor images")
    
    print("\n[2] Copying no tumor images...")
    no_tumor_count = 0
    
    if (source_train / "notumor").exists():
        for img in (source_train / "notumor").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_no_tumor / img.name)
                no_tumor_count += 1
    
    print(f"   ✓ Copied {no_tumor_count} no tumor images")
    
    # Also copy test images
    print("\n[3] Adding test images...")
    
    if (source_test / "glioma").exists():
        for img in (source_test / "glioma").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_tumor / f"test_glioma_{img.name}")
                tumor_count += 1
    
    if (source_test / "meningioma").exists():
        for img in (source_test / "meningioma").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_tumor / f"test_meningioma_{img.name}")
                tumor_count += 1
    
    if (source_test / "pituitary").exists():
        for img in (source_test / "pituitary").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_tumor / f"test_pituitary_{img.name}")
                tumor_count += 1
    
    if (source_test / "notumor").exists():
        for img in (source_test / "notumor").glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, dest_no_tumor / f"test_{img.name}")
                no_tumor_count += 1
    
    print(f"   ✓ Total tumor images: {tumor_count}")
    print(f"   ✓ Total no tumor images: {no_tumor_count}")
    
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETED!")
    print("="*70)
    print(f"\nDataset structure:")
    print(f"  data/raw/Tumor/      → {tumor_count} images")
    print(f"  data/raw/No_Tumor/   → {no_tumor_count} images")
    print(f"  Total:                 {tumor_count + no_tumor_count} images")
    print("\n✓ Ready for training! Run: python train.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    prepare_dataset()
