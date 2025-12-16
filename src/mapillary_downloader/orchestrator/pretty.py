def print_bbox_info(bbox: tuple[float, float, float, float]):
    """
    Print bounding box info with Google Maps URLs.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    print("\n📍 Bounding box:")
    print(
        f"   SW corner: https://www.google.com/maps/?q={min_lat:.6f},{min_lon:.6f}"
    )
    print(
        f"   NE corner: https://www.google.com/maps/?q={max_lat:.6f},{max_lon:.6f}"
    )
    print()

def print_bbox_progress(current: int, total: int, images_found: int):
    """Print progress bar for bbox processing."""
    if total <= 1:
        return  # No progress bar needed for single bbox
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = current / total * 100
    print(
        f"\r    Bbox [{bar}] {pct:5.1f}% ({current}/{total}) - {images_found} images",
        end="",
        flush=True,
    )
    if current == total:
        print()  # Newline when complete
