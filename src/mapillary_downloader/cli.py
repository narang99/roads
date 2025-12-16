"""
Command-line interface for Mapillary city image downloader.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .geocoder import get_city_bbox
from .orchestrator import CityImageDownloader
from .tiles import get_tiles_for_bbox, tile_to_bbox


def cmd_download(args):
    """Download images for a city."""
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
        print("\nTile Statistics:")
        for status, data in tile_stats.items():
            print(f"  {status}: {data['count']} tiles, {data['images']} images")
        print("\nImage Statistics:")
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


def cmd_tiles(args):
    """Show tile coordinates and bboxes for a city."""

    async def get_tiles():
        async with httpx.AsyncClient() as client:
            bbox = await get_city_bbox(client, args.city)
            return bbox

    print(f"Geocoding: {args.city}")
    bbox = asyncio.run(get_tiles())

    print("\nCity Bounding Box:")
    print(f"  min_lon: {bbox.min_lon}")
    print(f"  min_lat: {bbox.min_lat}")
    print(f"  max_lon: {bbox.max_lon}")
    print(f"  max_lat: {bbox.max_lat}")

    tiles = get_tiles_for_bbox(*bbox.as_tuple(), zoom=args.zoom)

    print(f"\nTiles at zoom {args.zoom}: {len(tiles)} total")

    if args.verbose or len(tiles) <= 20:
        print(
            f"\n{'Tile (z/x/y)':<20} {'min_lon':<12} {'min_lat':<12} {'max_lon':<12} {'max_lat':<12}"
        )
        print("-" * 80)
        for tile in tiles:
            tile_bbox = tile_to_bbox(tile)
            print(
                f"{tile.z}/{tile.x}/{tile.y:<10} "
                f"{tile_bbox[0]:<12.6f} {tile_bbox[1]:<12.6f} "
                f"{tile_bbox[2]:<12.6f} {tile_bbox[3]:<12.6f}"
            )
    else:
        print(f"\n(Use --verbose to show all {len(tiles)} tiles)")
        print("\nFirst 5 tiles:")
        for tile in tiles[:5]:
            tile_bbox = tile_to_bbox(tile)
            print(
                f"  {tile.z}/{tile.x}/{tile.y}: "
                f"bbox=({tile_bbox[0]:.6f}, {tile_bbox[1]:.6f}, {tile_bbox[2]:.6f}, {tile_bbox[3]:.6f})"
            )

    # Output as copyable format
    if args.output_format == "json":
        import json

        output = {
            "city": args.city,
            "city_bbox": bbox.as_tuple(),
            "zoom": args.zoom,
            "tiles": [
                {"z": t.z, "x": t.x, "y": t.y, "bbox": tile_to_bbox(t)} for t in tiles
            ],
        }
        print("\nJSON output:")
        print(json.dumps(output, indent=2))


def cmd_test_download(args):
    """Test download for a specific bounding box."""
    # Validate token
    if not args.token:
        print("Error: Mapillary access token required.", file=sys.stderr)
        print("Set MAPILLARY_ACCESS_TOKEN env var or use --token", file=sys.stderr)
        sys.exit(1)

    # Parse bbox
    try:
        bbox = tuple(map(float, args.bbox.split(",")))
        if len(bbox) != 4:
            raise ValueError("Bbox must have 4 values")
        min_lon, min_lat, max_lon, max_lat = bbox
    except Exception as e:
        print(f"Error parsing bbox: {e}", file=sys.stderr)
        print("Expected format: min_lon,min_lat,max_lon,max_lat", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    bbox_label = "test_bbox"

    # Create downloader
    downloader = CityImageDownloader(
        access_token=args.token,
        output_dir=output_dir,
        image_size=args.size,
        rate_limit_delay=args.delay,
        save_metadata=not args.no_metadata,
    )

    # Safety check: ensure we don't overwrite an existing dataset
    existing_city = downloader.state.get_metadata("city_name")
    if existing_city:
        if existing_city == bbox_label:
            # It's a previous test_bbox, ask if user wants to wipe
            print("Output directory already contains test data.")
            response = input(
                "Do you want to wipe the existing test data and start fresh? [y/N]: "
            )
            if response.lower() in ("y", "yes"):
                # Wipe the state database
                import os

                state_db = output_dir / "state.db"
                if state_db.exists():
                    os.remove(state_db)
                # Recreate downloader with fresh state
                downloader = CityImageDownloader(
                    access_token=args.token,
                    output_dir=output_dir,
                    image_size=args.size,
                    rate_limit_delay=args.delay,
                    save_metadata=not args.no_metadata,
                )
                print("Wiped existing test data.")
            else:
                print("Resuming existing test download...")
        else:
            # It's a real city dataset, don't allow overwrite
            print(
                f"Error: Output directory already contains data for '{existing_city}'.",
                file=sys.stderr,
            )
            print(
                "Use a different --output directory to avoid corrupting existing dataset.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Set metadata for this test download
    downloader.state.set_metadata("city_name", bbox_label)
    downloader.state.set_metadata("bbox", args.bbox)

    print(f"Test download for bbox: {bbox}")
    print(f"Output directory: {output_dir.absolute()}")

    # Progress callback
    def progress(phase: str, current: int, total: int, message: str):
        pct = (current / total * 100) if total > 0 else 0
        print(f"[{phase:>10}] {pct:5.1f}% | {message}")

    async def run_download():
        async with httpx.AsyncClient(timeout=30.0) as client:
            await downloader.download_bbox(
                client=client,
                bbox=bbox,
                progress_callback=progress,
                image_limit=args.limit,
            )

    try:
        asyncio.run(run_download())
    except KeyboardInterrupt:
        print("\nDownload interrupted. Progress has been saved.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_download_random_sample(args):
    """Download a random sample of images for a city or list of cities."""
    # Validate token
    if not args.token:
        print("Error: Mapillary access token required.", file=sys.stderr)
        print("Set MAPILLARY_ACCESS_TOKEN env var or use --token", file=sys.stderr)
        sys.exit(1)

    base_output_dir = Path(args.output)

    # Process cities
    cities = [c.strip() for c in args.city.split(",") if c.strip()]

    if not cities:
        print("Error: No city specified", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(cities)} cities: {', '.join(cities)}")

    for city in cities:
        # Determine output directory for this city
        if len(cities) > 1:
            # Sanitize city name for folder usage (basic)
            safe_name = city.replace("/", "_").replace("\\", "_")
            output_dir = base_output_dir / safe_name
        else:
            output_dir = base_output_dir

        print(f"\n--- Starting {city} ---")

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
            print(f"[{city}][{phase:>10}] {pct:5.1f}% | {message}")

        # Strategies
        def selection_strategy(state):
            print(
                f"\n[{city}] Selecting random sample of {args.sample_size} images from discovered set..."
            )
            state.mark_random_images_for_download(args.sample_size)

        # Run
        print(f"[{city}] Output directory: {output_dir.absolute()}")
        print(f"[{city}] Sample size: {args.sample_size}")

        try:
            asyncio.run(
                downloader.download_city(
                    city,
                    progress_callback=progress,
                    post_discovery_hook=selection_strategy,
                )
            )
        except KeyboardInterrupt:
            print(f"\n[{city}] Download interrupted.")
            sys.exit(130)  # Standard Ctrl-C exit code
        except Exception as e:
            print(f"\n[{city}] Error: {e}", file=sys.stderr)
            # We continue to next city if one fails?
            # Usually CLI tools fail hard. But for batch processing, maybe logging failure is better?
            # User request didn't specify. Failing hard is safer to notice errors.
            sys.exit(1)


def main():
    """Main CLI entry point."""
    load_dotenv()

    # Configure logging - write to file with tracebacks
    logging.basicConfig(
        level=logging.WARNING,  # Default for external libs
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("mapillary_downloader.log"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # Enable INFO logs only for our application code
    logging.getLogger("mapillary_downloader").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Download street-level images from Mapillary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command (main functionality)
    download_parser = subparsers.add_parser(
        "download",
        help="Download all images for a city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --city "Palo Alto, CA" --output ./images
  %(prog)s --city "Los Altos Hills, CA" --size thumb_2048_url
        """,
    )
    download_parser.add_argument(
        "--city",
        "-c",
        required=True,
        help="City name to download images for",
    )
    download_parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    download_parser.add_argument(
        "--size",
        "-s",
        choices=[
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
        ],
        default="thumb_1024_url",
        help="Image size (default: thumb_1024_url)",
    )
    download_parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("MAPILLARY_ACCESS_TOKEN"),
        help="Mapillary access token",
    )
    download_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls (default: 0.1s)",
    )
    download_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save JSON metadata",
    )
    download_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics and exit",
    )
    download_parser.set_defaults(func=cmd_download)

    # Random Sample Download command
    random_parser = subparsers.add_parser(
        "download_random_sample",
        help="Download a random sample of images for a city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    random_parser.add_argument(
        "--city",
        "-c",
        required=True,
        help="City name to download images for",
    )
    random_parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=100,
        help="Number of images to sample (default: 100)",
    )
    random_parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    random_parser.add_argument(
        "--size",
        "-s",
        choices=[
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
        ],
        default="thumb_1024_url",
        help="Image size (default: thumb_1024_url)",
    )
    random_parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("MAPILLARY_ACCESS_TOKEN"),
        help="Mapillary access token",
    )
    random_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls (default: 0.1s)",
    )
    random_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save JSON metadata",
    )
    random_parser.set_defaults(func=cmd_download_random_sample)

    # Tiles command
    tiles_parser = subparsers.add_parser(
        "tiles",
        help="Show tile coordinates and bboxes for a city",
    )
    tiles_parser.add_argument(
        "--city",
        "-c",
        required=True,
        help="City name to get tiles for",
    )
    tiles_parser.add_argument(
        "--zoom",
        "-z",
        type=int,
        default=14,
        help="Zoom level (default: 14)",
    )
    tiles_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all tiles (even if many)",
    )
    tiles_parser.add_argument(
        "--output-format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    tiles_parser.set_defaults(func=cmd_tiles)

    # Test download command
    test_parser = subparsers.add_parser(
        "test_download",
        help="Test download for a specific bounding box",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --bbox "-122.15,37.42,-122.14,37.43" --output ./test
  %(prog)s --bbox "-122.15,37.42,-122.14,37.43" --limit 10
        """,
    )
    test_parser.add_argument(
        "--bbox",
        "-b",
        required=True,
        help="Bounding box: min_lon,min_lat,max_lon,max_lat",
    )
    test_parser.add_argument(
        "--output",
        "-o",
        default="./test_output",
        help="Output directory (default: ./test_output)",
    )
    test_parser.add_argument(
        "--size",
        "-s",
        choices=[
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
        ],
        default="thumb_1024_url",
        help="Image size (default: thumb_1024_url)",
    )
    test_parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("MAPILLARY_ACCESS_TOKEN"),
        help="Mapillary access token",
    )
    test_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls (default: 0.1s)",
    )
    test_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save JSON metadata",
    )
    test_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of images to download",
    )
    test_parser.set_defaults(func=cmd_test_download)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
