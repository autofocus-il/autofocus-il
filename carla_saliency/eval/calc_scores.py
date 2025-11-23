#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
from pathlib import Path


def calculate_route_average(method_name, seed=None, route_type="seen"):
    """
    Calculate the mean and variance of score_composed for the specified method under the specified route type.

    Args:
        method_name
        seed: seed value, default is None meaning use all seeds
        route_type: route type, "seen" or "unseen"
    """
    # route list
    if route_type == "seen":
        routes = [2416, 3100, 3472, 24211, 24258, 24759, 25857, 25863, 26408, 27494]
    elif route_type == "unseen":
        routes = [18305, 1852, 24224, 3099, 3184, 3464, 27529, 26401, 2215, 25951]
    else:
        raise ValueError("route_type must be 'seen' or 'unseen'")

    # Store all score_composed values
    score_composeds = []
    successful_evaluations = []  # Store (route_id, seed) pairs
    failed_evaluations = []

    if seed is None:
        print(f"Calculating average for method {method_name} on all seeds for {route_type} routes...")
    else:
        print(
            f"Calculating average for method {method_name} on seed_{seed} for {route_type} routes..."
        )
    print(f"Number of {route_type.capitalize()} routes: {len(routes)}")
    print("-" * 60)

    # Iterate over all routes
    for route_id in routes:
        route_path = Path(
            f"/path/to/project_root/runs/Mixed_/{method_name}/route_{route_id}"
        )

        if not route_path.exists():
            if route_type == "seen":
                print(f"Warning: Route directory does not exist {route_path}")
            failed_evaluations.append((route_id, None, "Route directory does not exist"))
            continue

        # If seed is specified, only process that seed
        if seed is not None:
            seed_dirs = [f"seed_{seed}"]
        else:
            # Get all seed directories
            seed_dirs = [
                d.name
                for d in route_path.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            ]

        if not seed_dirs:
            if route_type == "seen":
                print(f"Warning: No seed directories found under Route {route_id}")
            failed_evaluations.append((route_id, None, "No seed directories"))
            continue

        # Iterate over all seed directories
        for seed_dir in seed_dirs:
            current_seed = seed_dir.split("_")[1] if "_" in seed_dir else seed_dir
            stats_path = route_path / seed_dir / "stats.json"

            if not stats_path.exists():
                if route_type == "seen":
                    print(f"Warning: File does not exist {stats_path}")
                failed_evaluations.append((route_id, current_seed, "stats.json does not exist"))
                continue

            try:
                # Read and parse JSON file
                with open(stats_path, "r") as f:
                    data = json.load(f)

                # Extract score_composed value
                score_composed = data["_checkpoint"]["global_record"]["scores_mean"][
                    "score_composed"
                ]
                score_composeds.append(score_composed)
                successful_evaluations.append((route_id, current_seed))

                if route_type == "seen":
                    print(
                        f"Route {route_id} seed_{current_seed}: score_composed = {score_composed}"
                    )

            except (json.JSONDecodeError, KeyError) as e:
                if route_type == "seen":
                    print(f"Error: Failed to parse file {stats_path}: {e}")
                failed_evaluations.append(
                    (route_id, current_seed, f"JSON parse error: {e}")
                )
                continue
            except Exception as e:
                if route_type == "seen":
                    print(f"Error: Failed to process file {stats_path}: {e}")
                failed_evaluations.append((route_id, current_seed, f"Processing error: {e}"))
                continue

    # Calculate statistics
    if score_composeds:
        # Convert to numpy array for easier calculation
        scores_array = np.array(score_composeds)

        # Calculate basic statistics
        mean_score = np.mean(scores_array)
        variance = np.var(scores_array, ddof=1)  # Sample variance
        std_dev = np.std(scores_array, ddof=1)  # Sample standard deviation
        std_error = std_dev / np.sqrt(len(scores_array))  # Standard error

        print("-" * 60)
        print(f"Results Statistics:")
        print(f"Number of successful evaluations: {len(score_composeds)}")
        print(
            f"Evaluations from {len(set(eval[0] for eval in successful_evaluations))} routes"
        )
        print(f"Total {len(routes)} routes")
        if route_type == "seen":
            print(f"All score_composed values: {score_composeds}")
        print(f"Mean: {mean_score:.2f}")
        print(f"Variance: {variance:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Standard Error: {std_error:.2f}")
        print(
            f"Error Margin (95% Confidence Interval): [{mean_score - 1.96*std_error:.2f}, {mean_score + 1.96*std_error:.2f}]"
        )
        print(
            f"Error Margin (68% Confidence Interval): [{mean_score - std_error:.2f}, {mean_score + std_error:.2f}]"
        )

        # Stats grouped by route
        route_stats = {}
        for route_id, seed in successful_evaluations:
            if route_id not in route_stats:
                route_stats[route_id] = []
            # Find corresponding score
            idx = successful_evaluations.index((route_id, seed))
            route_stats[route_id].append((seed, score_composeds[idx]))

        # Output successful and failed evaluations
        print("-" * 60)
        print(f"Successfully evaluated routes and seeds ({len(successful_evaluations)} evaluations):")
        if route_type == "seen":
            for route_id in sorted(route_stats.keys()):
                seeds_scores = route_stats[route_id]
                seeds_str = ", ".join([f"seed_{s}({sc:.1f})" for s, sc in seeds_scores])
                print(f"  Route {route_id}: {seeds_str}")
        else:
            # For unseen routes, only show summary info
            for route_id in sorted(list(route_stats.keys())[:5]):
                seeds_scores = route_stats[route_id]
                print(f"  Route {route_id}: {len(seeds_scores)} seeds")
            if len(route_stats) > 5:
                print(f"  ... and {len(route_stats) - 5} more routes")

        if failed_evaluations:
            print(f"\nUnsuccessful evaluation cases ({len(failed_evaluations)}):")
            if route_type == "seen":
                for route_id, seed, reason in failed_evaluations:
                    seed_str = f"seed_{seed}" if seed else "unknown seed"
                    print(f"  Route {route_id} {seed_str}: {reason}")
            else:
                # For unseen routes, only show top 10 failed
                for route_id, seed, reason in failed_evaluations[:10]:
                    seed_str = f"seed_{seed}" if seed else "unknown seed"
                    print(f"  Route {route_id} {seed_str}: {reason}")
                if len(failed_evaluations) > 10:
                    print(f"  ... and {len(failed_evaluations) - 10} more failed evaluations")
        else:
            print(f"\nAll evaluations succeeded!")

        return {
            "mean": mean_score,
            "variance": variance,
            "std_dev": std_dev,
            "std_error": std_error,
            "count": len(score_composeds),
            "total_routes": len(routes),
            "successful_routes": len(route_stats),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "route_stats": route_stats,
        }
    else:
        print("Error: No evaluations processed successfully")
        print(f"All evaluations failed:")
        if route_type == "seen":
            for route_id, seed, reason in failed_evaluations:
                seed_str = f"seed_{seed}" if seed else "unknown seed"
                print(f"  Route {route_id} {seed_str}: {reason}")
        else:
            for route_id, seed, reason in failed_evaluations[:10]:
                seed_str = f"seed_{seed}" if seed else "unknown seed"
                print(f"  Route {route_id} {seed_str}: {reason}")
            if len(failed_evaluations) > 10:
                print(f"  ... and {len(failed_evaluations) - 10} more failed evaluations")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean and variance of score_composed for routes"
    )
    parser.add_argument(
        "--method", "-m", default="pseudo_gmd", help="Method name (default: pseudo_gmd)"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Seed value (default: None means use all seeds)",
    )
    parser.add_argument(
        "--route-type",
        "-r",
        choices=["seen", "unseen"],
        default="seen",
        help="Route type: seen or unseen (default: seen)",
    )

    args = parser.parse_args()

    # Calculate stats
    result = calculate_route_average(args.method, args.seed, args.route_type)

    if result:
        if args.seed is None:
            print(
                f"\nSummary: {args.method} on {args.route_type} routes (all seeds) average score_composed: {result['mean']:.2f} ± {result['std_error']:.2f}"
            )
            print(
                f"Based on {result['count']} evaluations (from {result['successful_routes']} routes)"
            )
        else:
            print(
                f"\nSummary: {args.method} on {args.route_type} routes seed_{args.seed} average score_composed: {result['mean']:.2f} ± {result['std_error']:.2f}"
            )


if __name__ == "__main__":
    main()
