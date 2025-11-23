#!/usr/bin/env python3
"""
Explore demos in the HDF5 dataset
"""

import h5py
import numpy as np


def explore_demos(
    hdf5_path="/path/to/dataset/bench2drive220_robomimic_confounded.hdf5",
    show_full_structure=False,
    max_demos_to_show=3,
):
    """
    List all demos and their properties

    Args:
        hdf5_path: Path to HDF5 file
        show_full_structure: If True, show complete recursive structure. If False, show limited structure
        max_demos_to_show: Maximum number of demos to show in structure (only used if show_full_structure=False)
    """
    print(f"Exploring HDF5 file: {hdf5_path}\n")

    def print_hdf5_structure(group, indent="", demo_count=[0], showed_ellipsis=[False]):
        """Recursively print HDF5 structure"""
        for key in sorted(group.keys()):
            # Skip demos after max_demos_to_show if not showing full structure
            if (
                not show_full_structure and key.startswith("demo_") and indent == "   "
            ):  # Only check at data level
                if demo_count[0] >= max_demos_to_show:
                    if not showed_ellipsis[0]:
                        total_demos = len(
                            [k for k in group.keys() if k.startswith("demo_")]
                        )
                        print(
                            f"{indent}... ({total_demos - max_demos_to_show} more demos, {total_demos} total)"
                        )
                        showed_ellipsis[0] = True
                    continue
                demo_count[0] += 1

            item = group[key]
            if isinstance(item, h5py.Group):
                print(f"{indent}📁 {key}/")
                # Limit recursion depth to avoid too much output
                if len(indent) < 12:  # Max 4 levels deep (3 spaces per level)
                    print_hdf5_structure(
                        item, indent + "   ", demo_count, showed_ellipsis
                    )
                else:
                    print(f"{indent}   ...")
            elif isinstance(item, h5py.Dataset):
                print(f"{indent}📄 {key} - shape: {item.shape}, dtype: {item.dtype}")

    with h5py.File(hdf5_path, "r") as f:
        # First, print the entire HDF5 structure
        print("=" * 60)
        print("HDF5 File Structure (Recursive):")
        print("=" * 60)
        print_hdf5_structure(f)
        print("\n" + "=" * 60 + "\n")

        if "data" not in f:
            print("No 'data' group found!")
            return

        # Get all demo keys
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        print(f"Total demos: {len(demo_keys)}\n")

        # Print info for each demo
        print("Demo Information:")
        print("-" * 60)
        print(f"{'Demo':<10} {'Frames':<10} {'Duration(s)':<12} {'Has Gaze':<10}")
        print("-" * 60)

        total_frames = 0
        valid_demos = []
        for demo_key in demo_keys:
            try:
                # Get number of frames
                image_shape = f[f"data/{demo_key}/obs/image"].shape
                num_frames = image_shape[0]
                duration = num_frames / 10.0  # Assuming 10 FPS

                # Check if gaze data exists
                has_gaze = "gaze_coords" in f[f"data/{demo_key}/obs"]

                print(
                    f"{demo_key:<10} {num_frames:<10} {duration:<12.1f} {'Yes' if has_gaze else 'No':<10}"
                )
                total_frames += num_frames
                valid_demos.append(demo_key)
            except KeyError:
                print(
                    f"{demo_key:<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} (Data not found)"
                )

        print("-" * 60)
        print(f"Total valid demos: {len(valid_demos)} out of {len(demo_keys)}")
        print(f"Total frames across valid demos: {total_frames}")
        if valid_demos:
            print(f"Average frames per valid demo: {total_frames/len(valid_demos):.1f}")

        # Show available gaze keys from first valid demo
        if valid_demos:
            first_demo = valid_demos[0]
            print("\n" + "=" * 60)
            print(f"Available gaze coordinate keys in {first_demo}:")
            print("-" * 60)
            try:
                for key in f[f"data/{first_demo}/obs"].keys():
                    if "gaze" in key.lower():
                        shape = f[f"data/{first_demo}/obs/{key}"].shape
                        print(f"  - {key}: shape={shape}")

                        # Show sample data
                        sample = f[f"data/{first_demo}/obs/{key}"][0]  # First frame
                        valid_points = sample.reshape(-1, 2)
                        valid_count = np.sum(
                            (valid_points[:, 0] >= 0) & (valid_points[:, 1] >= 0)
                        )
                        print(f"    First frame has {valid_count} valid gaze points")
            except Exception as e:
                print(f"Error reading gaze keys: {e}")


if __name__ == "__main__":
    explore_demos()
