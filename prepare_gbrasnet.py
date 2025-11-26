"""
Split GBRASNET dataset into train/val/test while maintaining correspondence
between cover and stego images across all algorithms and payloads
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import argparse

def get_image_files(directory):
    """Get sorted list of image files"""
    path = Path(directory)
    if not path.exists():
        return []

    files = []
    for ext in ['*.pgm', '*.png', '*.jpg', '*.jpeg']:
        files.extend(path.glob(ext))

    # Sort by filename to ensure consistent ordering
    return sorted([f.name for f in files])


def split_indices(total, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Generate train/val/test indices"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = list(range(total))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train + n_val])
    test_idx = set(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def copy_files_by_split(source_dir, dest_base, files, train_idx, val_idx, test_idx):
    """Copy files to train/val/test directories"""

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    for split_name, indices in splits.items():
        dest_dir = dest_base / split_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for idx, filename in enumerate(files):
            if idx in indices:
                src = Path(source_dir) / filename
                dst = dest_dir / filename
                shutil.copy2(src, dst)


def process_gbrasnet(args):
    """Main function to process GBRASNET dataset"""

    source_base = Path(args.source)
    dest_base = Path(args.destination)

    print("="*70)
    print("GBRASNET Dataset Splitting")
    print("="*70)
    print(f"Source: {source_base}")
    print(f"Destination: {dest_base}")
    print(f"Split: 80% train, 10% val, 10% test")
    print()

    # Process BOSSbase-1.01
    print("Processing BOSSbase-1.01...")
    print("-"*70)

    boss_cover_src = source_base / 'BOSSbase-1.01' / 'cover'
    boss_cover_files = get_image_files(boss_cover_src)

    if not boss_cover_files:
        print(f"ERROR: No images found in {boss_cover_src}")
        return

    print(f"Found {len(boss_cover_files)} cover images")

    # Generate split indices (same for all)
    train_idx, val_idx, test_idx = split_indices(len(boss_cover_files))
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Process cover images
    print("\nCopying cover images...")
    boss_cover_dest = dest_base / 'BOSSbase-1.01' / 'cover'
    copy_files_by_split(boss_cover_src, boss_cover_dest, boss_cover_files,
                       train_idx, val_idx, test_idx)

    # Process stego images for each algorithm and payload
    algorithms = ['HILL', 'HUGO', 'MiPOD', 'S-UNIWARD', 'WOW']
    payloads = ['0.2bpp', '0.4bpp']

    for algo in algorithms:
        for payload in payloads:
            stego_src = source_base / 'BOSSbase-1.01' / 'stego' / algo / payload / 'stego'

            if not stego_src.exists():
                print(f"WARNING: {stego_src} not found, skipping...")
                continue

            print(f"\nCopying {algo} {payload}...")
            stego_dest = dest_base / 'BOSSbase-1.01' / 'stego' / algo / payload

            stego_files = get_image_files(stego_src)
            if len(stego_files) != len(boss_cover_files):
                print(f"WARNING: Stego count ({len(stego_files)}) != cover count ({len(boss_cover_files)})")

            copy_files_by_split(stego_src, stego_dest, stego_files,
                              train_idx, val_idx, test_idx)

    # Process BOWS2
    print("\n" + "="*70)
    print("Processing BOWS2...")
    print("-"*70)

    bows_cover_src = source_base / 'BOWS2' / 'cover'
    bows_cover_files = get_image_files(bows_cover_src)

    if not bows_cover_files:
        print(f"ERROR: No images found in {bows_cover_src}")
        return

    print(f"Found {len(bows_cover_files)} cover images")

    # Generate split indices for BOWS2
    train_idx, val_idx, test_idx = split_indices(len(bows_cover_files))
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Process cover images
    print("\nCopying cover images...")
    bows_cover_dest = dest_base / 'BOWS2' / 'cover'
    copy_files_by_split(bows_cover_src, bows_cover_dest, bows_cover_files,
                       train_idx, val_idx, test_idx)

    # Process BOWS2 stego
    bows_algos = {
        'S-UNIWARD': ['0.2bpp'],
        'WOW': ['0.2bpp']
    }

    for algo, payloads in bows_algos.items():
        for payload in payloads:
            stego_src = source_base / 'BOWS2' / 'stego' / algo / payload / 'stego'

            if not stego_src.exists():
                print(f"WARNING: {stego_src} not found, skipping...")
                continue

            print(f"\nCopying {algo} {payload}...")
            stego_dest = dest_base / 'BOWS2' / 'stego' / algo / payload

            stego_files = get_image_files(stego_src)
            if len(stego_files) != len(bows_cover_files):
                print(f"WARNING: Stego count ({len(stego_files)}) != cover count ({len(bows_cover_files)})")

            copy_files_by_split(stego_src, stego_dest, stego_files,
                              train_idx, val_idx, test_idx)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nBOSSbase-1.01:")
    print(f"  Cover: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    print(f"  Stego algorithms: {len(algorithms)}")
    print(f"  Payloads per algorithm: {len(payloads)}")
    print(f"  Total stego variants: {len(algorithms) * len(payloads)}")

    print("\nBOWS2:")
    total_bows_variants = sum(len(p) for p in bows_algos.values())
    print(f"  Cover: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    print(f"  Stego variants: {total_bows_variants}")

    print("\n" + "="*70)
    print("Dataset splitting complete!")
    print(f"Output directory: {dest_base}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare GBRASNET dataset by splitting into train/val/test sets.")

    parser.add_argument(
        '--source',
        type=str,
        default='/Users/dmitryhoma/Projects/datasets/GBRASNET',
        help='Path to the GBRASNET dataset source directory.'
    )
    parser.add_argument(
        '--destination',
        type=str,
        default='/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET',
        help='Path to the output directory for the prepared dataset.'
    )
    args = parser.parse_args()

    process_gbrasnet(args)

# python3 ~/INATNet_v2/prepare_dataset.py --source_dir ~/datasets/GBRASNET --output_dir ~/datasets/ready_to_use/GBRASNET
