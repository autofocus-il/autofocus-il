#!/usr/bin/env python3
"""
Check HDF5 file structure for training data
"""

import h5py
import numpy as np


def explore_hdf5(file_path):
    """Explore the structure of an HDF5 file"""
    print(f"Opening HDF5 file: {file_path}")

    with h5py.File(file_path, "r") as f:

        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

        print("\n=== HDF5 File Structure ===")
        f.visititems(print_structure)

        print("\n=== Top-level keys ===")
        print(list(f.keys()))

        # Check data organization
        if "data" in f:
            print("\n=== Data group keys ===")
            print(list(f["data"].keys()))

            # Check a few demos
            demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            print(f"\nNumber of demos: {len(demo_keys)}")

            if demo_keys:
                # Check first demo structure
                first_demo = demo_keys[0]
                print(f"\n=== Structure of {first_demo} ===")
                if "obs" in f[f"data/{first_demo}"]:
                    print("Observation keys:")
                    for key in f[f"data/{first_demo}/obs"].keys():
                        dataset = f[f"data/{first_demo}/obs/{key}"]
                        print(
                            f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}"
                        )

                if "actions" in f[f"data/{first_demo}"]:
                    actions = f[f"data/{first_demo}/actions"]
                    print(f"\nActions: shape={actions.shape}, dtype={actions.dtype}")

                # Check for different gaze coordinate keys
                print("\n=== Available gaze coordinate keys ===")
                for key in f[f"data/{first_demo}/obs"].keys():
                    if "gaze" in key.lower():
                        dataset = f[f"data/{first_demo}/obs/{key}"]
                        print(f"  - {key}: shape={dataset.shape}")
                        # Show a sample
                        if len(dataset) > 0:
                            sample = dataset[0]
                            print(
                                f"    Sample data: {sample[:2] if len(sample) > 2 else sample}"
                            )


if __name__ == "__main__":
    hdf5_path = "/path/to/dataset/bench2drive220_robomimic.hdf5"
    explore_hdf5(hdf5_path)
