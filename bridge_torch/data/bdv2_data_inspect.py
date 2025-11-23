import os
import sys
import json
import pickle
from typing import Any, Dict


def _print_dict(d: Dict[str, Any], prefix: str = "", max_items: int = 3, max_depth: int = 3):
    """Print dictionary content, format output data structure"""
    if not d:
        print(f"{prefix}<empty dict>")
        return
    
    items = list(d.items())
    max_key_len = max(len(str(k)) for k, _ in items) if items else 0
    
    for i, (k, v) in enumerate(items):
        key_str = f"{k:<{max_key_len}}"
        if isinstance(v, dict):
            print(f"{prefix}├─ {key_str} : dict[{len(v)}]")
            if len(v) > 0 and max_depth > 0:
                _print_dict(v, prefix + "│  " + " " * (max_key_len + 3), max_items, max_depth - 1)
        elif hasattr(v, "shape"):
            try:
                shape_str = str(v.shape)
                dtype_str = str(getattr(v, 'dtype', 'unknown'))
                size = v.size if hasattr(v, 'size') else 'unknown'
                print(f"{prefix}├─ {key_str} : ndarray {shape_str} | dtype={dtype_str} | size={size}")
                
                # Show value range of array (if numeric)
                if hasattr(v, 'min') and hasattr(v, 'max') and v.size > 0:
                    try:
                        min_val, max_val = float(v.min()), float(v.max())
                        print(f"{prefix}│  {' ' * max_key_len}   ↳ range: [{min_val:.6g}, {max_val:.6g}]")
                    except Exception:
                        pass
            except Exception:
                print(f"{prefix}├─ {key_str} : array-like")
        elif isinstance(v, (list, tuple)):
            n = len(v)
            type_name = "list" if isinstance(v, list) else "tuple"
            print(f"{prefix}├─ {key_str} : {type_name}[{n}]")
            
            if n > 0:
                # Analyze type of list content
                head = v[:max_items]
                type_counts = {}
                for item in head:
                    if hasattr(item, "shape"):
                        key = f"ndarray{item.shape}"
                    else:
                        key = type(item).__name__
                    type_counts[key] = type_counts.get(key, 0) + 1
                
                # Show type statistics
                for j, item in enumerate(head):
                    if j >= max_items:
                        break
                    if hasattr(item, "shape"):
                        shape_str = str(item.shape)
                        dtype_str = str(getattr(item, 'dtype', 'unknown'))
                        print(f"{prefix}│  {' ' * max_key_len}   [{j}] ndarray {shape_str} dtype={dtype_str}")
                    else:
                        item_repr = repr(item)
                        if len(item_repr) > 60:
                            item_repr = item_repr[:57] + "..."
                        print(f"{prefix}│  {' ' * max_key_len}   [{j}] {type(item).__name__}: {item_repr}")
                
                if n > max_items:
                    print(f"{prefix}│  {' ' * max_key_len}   ... and {n - max_items} more items")
        else:
            # Normal value
            value_repr = repr(v)
            if len(value_repr) > 80:
                value_repr = value_repr[:77] + "..."
            print(f"{prefix}├─ {key_str} : {type(v).__name__} = {value_repr}")
    
    print(f"{prefix}")  # Add empty line separator


def inspect_traj(traj_dir: str) -> None:
    obs_pkl = os.path.join(traj_dir, "obs_dict.pkl")
    pol_pkl = os.path.join(traj_dir, "policy_out.pkl")
    agt_pkl = os.path.join(traj_dir, "agent_data.pkl")

    print("=" * 80)
    print(f"📂 TRAJECTORY INSPECTION")
    print(f"📍 Path: {traj_dir}")
    print("=" * 80)

    # Check file status
    files_status = [
        ("obs_dict.pkl", obs_pkl),
        ("policy_out.pkl", pol_pkl), 
        ("agent_data.pkl", agt_pkl)
    ]
    
    print("\n📋 File Status:")
    for name, path in files_status:
        status = "✅ Found" if os.path.isfile(path) else "❌ Missing"
        size = f"({os.path.getsize(path)} bytes)" if os.path.isfile(path) else ""
        print(f"   {name:<20} {status} {size}")

    # obs_dict.pkl check
    if os.path.isfile(obs_pkl):
        with open(obs_pkl, "rb") as f:
            obs = pickle.load(f)
        print("\n" + "─" * 60)
        print("🔍 obs_dict.pkl - Observation Data")
        print("─" * 60)
        print(f"📊 Total fields: {len(obs)}")
        _print_dict(obs, prefix="", **_print_dict_defaults)
    else:
        print("\n❌ obs_dict.pkl not found")

    # policy_out.pkl check
    if os.path.isfile(pol_pkl):
        with open(pol_pkl, "rb") as f:
            pol = pickle.load(f)
        print("\n" + "─" * 60)
        print("🎯 policy_out.pkl - Policy Output Data")
        print("─" * 60)
        print(f"📊 Container type: {type(pol).__name__}")
        print(f"📊 Total length: {len(pol)}")
        
        if len(pol) > 0:
            print(f"📊 First element type: {type(pol[0]).__name__}")
            if isinstance(pol[0], dict):
                print(f"📊 First element fields: {len(pol[0])}")
                print("\n🔍 First Element Structure:")
                _print_dict(pol[0], prefix="", **_print_dict_defaults)
            else:
                print(f"📊 First element: {repr(pol[0])[:100]}...")
    else:
        print("\n❌ policy_out.pkl not found")

    # agent_data.pkl check
    if os.path.isfile(agt_pkl):
        print("\n" + "─" * 60)
        print("🤖 agent_data.pkl - Agent Data")
        print("─" * 60)
        try:
            with open(agt_pkl, "rb") as f:
                agt = pickle.load(f)
            print(f"📊 Data type: {type(agt).__name__}")
            if isinstance(agt, dict):
                print(f"📊 Total fields: {len(agt)}")
                _print_dict(agt, prefix="", **_print_dict_defaults)
            else:
                print(f"📊 Content: {repr(agt)[:200]}...")
        except Exception as e:
            print(f"❌ Failed to load: {e.__class__.__name__}: {e}")
    else:
        print("\n❌ agent_data.pkl not found")
    
    print("\n" + "=" * 80)


def auto_pick_one(bdv2_root: str) -> str:
    for task in sorted(os.listdir(bdv2_root)):
        tdir = os.path.join(bdv2_root, task)
        if not os.path.isdir(tdir):
            continue
        for col in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, col)
            raw = os.path.join(cdir, "raw")
            if not os.path.isdir(raw):
                continue
            for grp in sorted(os.listdir(raw)):
                gdir = os.path.join(raw, grp)
                if not os.path.isdir(gdir):
                    continue
                for traj in sorted(os.listdir(gdir)):
                    tdir2 = os.path.join(gdir, traj)
                    if os.path.isdir(tdir2):
                        return tdir2
    raise FileNotFoundError("No trajectory found under bdv2 root")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="🔍 Inspect BDV2 trajectory pickle files with detailed structure analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bdv2_data_inspect.py  # Auto-pick first trajectory
  python bdv2_data_inspect.py --traj_dir /path/to/specific/traj0
  python bdv2_data_inspect.py --bdv2_root /data/bdv2 --max_items 5
        """
    )
    p.add_argument("--traj_dir", type=str, default="", 
                   help="Path to specific trajectory directory (default: auto-pick first)")
    p.add_argument("--bdv2_root", type=str, default="/scr/litian/dataset/bdv2",
                   help="Root directory of BDV2 dataset")
    p.add_argument("--max_items", type=int, default=3,
                   help="Maximum items to show in lists/arrays (default: 3)")
    p.add_argument("--max_depth", type=int, default=3,
                   help="Maximum depth for nested dict inspection (default: 3)")
    p.add_argument("--box_threshold", type=float, default=None,
                    help="Box threshold for detection")
    args = p.parse_args()

    try:
        traj = args.traj_dir if args.traj_dir else auto_pick_one(args.bdv2_root)
        print(f"🚀 Starting inspection with max_items={args.max_items}, max_depth={args.max_depth}")
        
        # Pass parameters to _print_dict through global modification
        global _print_dict_defaults
        _print_dict_defaults = {'max_items': args.max_items, 'max_depth': args.max_depth}
        
        inspect_traj(traj)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure the BDV2 dataset exists at: {args.bdv2_root}")
    except Exception as e:
        print(f"❌ Unexpected error: {e.__class__.__name__}: {e}")

# Global defaults for _print_dict
_print_dict_defaults = {'max_items': 3, 'max_depth': 3}


if __name__ == "__main__":
    main()
