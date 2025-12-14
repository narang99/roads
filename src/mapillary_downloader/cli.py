"""
Command-line interface for Mapillary city image downloader.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .orchestrator import CityImageDownloader


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download street-level images from Mapillary for a city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
State management Database is stored inside the output folder

Examples:
  %(prog)s --city "Palo Alto, CA" --output ./images
  %(prog)s --city "Los Altos Hills, CA" --output ./images --size thumb_2048_url

Environment Variables:
  MAPILLARY_ACCESS_TOKEN  Your Mapillary API access token
        """,
    )

    parser.add_argument(
        "--city",
        "-c",
        required=True,
        help="City name to download images for (e.g., 'Palo Alto, CA')",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory for images and state (default: ./output)",
    )

    parser.add_argument(
        "--size",
        "-s",
        choices=[
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
        ],
        default="thumb_1024_url",
        help="Image size to download (default: thumb_1024_url)",
    )

    parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("MAPILLARY_ACCESS_TOKEN"),
        help="Mapillary access token (or set MAPILLARY_ACCESS_TOKEN env var)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save JSON metadata for each image",
    )

    parser.add_argument(
        "--stats", action="store_true", help="Show download statistics and exit"
    )

    args = parser.parse_args()

    # Validate token
    if not args.token:
        print("Error: Mapillary access token required.", file=sys.stderr)
        print("Set MAPILLARY_ACCESS_TOKEN env var or use --token", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)

    # Stats only mode
    if args.stats:
        from .state import StateManager

        state_db = output_dir / "state.db"
        if not state_db.exists():
            print(f"No state database found at {state_db}", file=sys.stderr)
            sys.exit(1)

        state = StateManager(state_db)
        tile_stats = state.get_tile_stats()
        image_stats = state.get_image_stats()

        print(f"City: {state.get_metadata('city_name')}")
        print(f"Bounding Box: {state.get_metadata('bbox')}")
        print(f"\nTile Statistics:")
        for status, data in tile_stats.items():
            print(f"  {status}: {data['count']} tiles, {data['images']} images")
        print(f"\nImage Statistics:")
        for status, count in image_stats.items():
            print(f"  {status}: {count}")
        return

    # Create downloader
    downloader = CityImageDownloader(
        access_token=args.token,
        output_dir=output_dir,
        image_size=args.size,
        rate_limit_delay=args.delay,
        save_metadata=not args.no_metadata,
    )

    # Progress callback
    def progress(phase: str, current: int, total: int, message: str):
        pct = (current / total * 100) if total > 0 else 0
        print(f"[{phase:>10}] {pct:5.1f}% | {message}")

    # Run
    print(f"Starting download for: {args.city}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Image size: {args.size}")
    print()

    try:
        asyncio.run(downloader.download_city(args.city, progress_callback=progress))
    except KeyboardInterrupt:
        print("\nDownload interrupted. Progress has been saved.")
        print("Re-run the same command to resume.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
