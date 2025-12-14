"""
Mapillary API client module.

This module contains ONLY the API interaction functions, separated for testability.
Each function makes exactly one API call and returns the raw/parsed response.
"""

import asyncio
from pathlib import Path
from typing import Optional

import httpx

# API Endpoints
GRAPH_API_ROOT = "https://graph.mapillary.com"
TILES_API_ROOT = "https://tiles.mapillary.com"
COVERAGE_TILES_URL = f"{TILES_API_ROOT}/maps/vtp/mly1_public/2/{{z}}/{{x}}/{{y}}"


class MapillaryAPIError(Exception):
    """Error from Mapillary API."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        request_url: Optional[str] = None,
        request_params: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        self.request_url = request_url
        self.request_params = request_params

        # Try to parse error details from JSON response
        self.error_type = None
        self.error_message = None
        self.error_code = None

        if response_body:
            try:
                import json

                error_data = json.loads(response_body)
                if "error" in error_data:
                    err = error_data["error"]
                    self.error_type = err.get("type")
                    self.error_message = err.get("message") or err.get("error_user_msg")
                    self.error_code = err.get("code")
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Build detailed message
        full_message = message
        if self.error_message:
            full_message = f"{message}: {self.error_message}"
        if self.error_type:
            full_message = f"{full_message} (type: {self.error_type})"
        if self.error_code:
            full_message = f"{full_message} [code: {self.error_code}]"

        super().__init__(full_message)

    def __str__(self):
        parts = [super().__str__()]
        if self.request_url:
            parts.append(f"URL: {self.request_url}")
        if self.request_params:
            parts.append(f"Params: {self.request_params}")
        if self.response_body and not self.error_message:
            # Only show raw body if we couldn't parse it
            parts.append(f"Response: {self.response_body[:500]}")
        return " | ".join(parts)


class RateLimitError(MapillaryAPIError):
    """Rate limit exceeded."""

    pass


# --- Vector Tile API ---


async def fetch_coverage_tile(
    client: httpx.AsyncClient, access_token: str, z: int, x: int, y: int
) -> bytes:
    """
    Fetch a coverage tile from Mapillary.

    This returns the raw MVT (Mapbox Vector Tile) protobuf data.

    Args:
        client: httpx AsyncClient
        access_token: Mapillary access token
        z, x, y: Tile coordinates

    Returns:
        Raw tile bytes (protobuf/MVT format)

    Raises:
        RateLimitError: If rate limit is exceeded
        MapillaryAPIError: On other API errors
    """
    url = COVERAGE_TILES_URL.format(z=z, x=x, y=y)

    response = await client.get(url, params={"access_token": access_token})

    if response.status_code == 429:
        raise RateLimitError(
            "Tile API rate limit exceeded",
            status_code=429,
            response_body=response.text,
            request_url=url,
            request_params={"z": z, "x": x, "y": y},
        )

    if response.status_code >= 400:
        raise MapillaryAPIError(
            f"Tile API error: {response.status_code}",
            status_code=response.status_code,
            response_body=response.text,
            request_url=url,
            request_params={"z": z, "x": x, "y": y},
        )

    return response.content


def parse_mvt_tile(tile_data: bytes, layer_name: str = "image") -> list[dict]:
    """
    Parse an MVT tile to extract features.

    Uses a manual approach since vt2geojson requires mapbox-vector-tile
    which can be heavy. We'll use a simpler approach.

    Args:
        tile_data: Raw MVT bytes
        layer_name: Layer to extract (default: "image")

    Returns:
        List of feature dicts with id and properties
    """
    # Import here to allow the module to load even if not installed
    try:
        import mapbox_vector_tile
    except ImportError:
        # Fallback: we'll use the bbox search instead
        return []

    try:
        decoded = mapbox_vector_tile.decode(tile_data)
        if layer_name not in decoded:
            # Try other layer names
            for name in ["image", "sequence", "overview"]:
                if name in decoded:
                    layer_name = name
                    break
            else:
                return []

        features = decoded[layer_name].get("features", [])
        return [
            {
                "id": f.get("id") or f.get("properties", {}).get("id"),
                "properties": f.get("properties", {}),
                "geometry": f.get("geometry", {}),
            }
            for f in features
            if f.get("id") or f.get("properties", {}).get("id")
        ]
    except Exception:
        return []


# --- Entity API ---


async def fetch_images_in_bbox(
    client: httpx.AsyncClient,
    access_token: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    fields: list[str] = None,
    limit: int = 2000,
) -> list[dict]:
    """
    Fetch all images within a bounding box.

    Args:
        client: httpx AsyncClient
        access_token: Mapillary access token
        min_lon, min_lat, max_lon, max_lat: Bounding box
        fields: List of fields to request (default: id, captured_at, computed_geometry)
        limit: Max images per request (API max is 2000)

    Returns:
        List of image dicts

    Raises:
        RateLimitError: If rate limit exceeded
        MapillaryAPIError: On other errors
    """
    if fields is None:
        fields = ["id", "captured_at", "computed_geometry", "is_pano"]

    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    url = f"{GRAPH_API_ROOT}/images"

    response = await client.get(
        url,
        params={
            "access_token": access_token,
            "bbox": bbox_str,
            "fields": ",".join(fields),
            "limit": min(limit, 2000),
        },
        headers={"Authorization": f"OAuth {access_token}"},
    )

    if response.status_code == 429:
        raise RateLimitError(
            "Search API rate limit exceeded",
            status_code=429,
            response_body=response.text,
            request_url=url,
            request_params={"bbox": bbox_str, "fields": fields, "limit": limit},
        )

    if response.status_code >= 400:
        raise MapillaryAPIError(
            f"Search API error: {response.status_code}",
            status_code=response.status_code,
            response_body=response.text,
            request_url=url,
            request_params={"bbox": bbox_str, "fields": fields, "limit": limit},
        )

    data = response.json()
    return data.get("data", [])


async def fetch_image_metadata(
    client: httpx.AsyncClient,
    access_token: str,
    image_id: str,
    fields: list[str] = None,
) -> dict:
    """
    Fetch metadata for a single image.

    Args:
        client: httpx AsyncClient
        access_token: Mapillary access token
        image_id: Mapillary image ID
        fields: Fields to request

    Returns:
        Image metadata dict
    """
    if fields is None:
        fields = [
            "id",
            "captured_at",
            "computed_geometry",
            "computed_compass_angle",
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
            "is_pano",
            "camera_type",
            "make",
            "model",
        ]

    url = f"{GRAPH_API_ROOT}/{image_id}"

    response = await client.get(
        url,
        params={"access_token": access_token, "fields": ",".join(fields)},
        headers={"Authorization": f"OAuth {access_token}"},
    )

    if response.status_code == 429:
        raise RateLimitError(
            "Entity API rate limit exceeded",
            status_code=429,
            response_body=response.text,
            request_url=url,
            request_params={"image_id": image_id, "fields": fields},
        )

    if response.status_code >= 400:
        raise MapillaryAPIError(
            f"Entity API error for image {image_id}: {response.status_code}",
            status_code=response.status_code,
            response_body=response.text,
            request_url=url,
            request_params={"image_id": image_id, "fields": fields},
        )

    return response.json()


async def fetch_multiple_image_metadata(
    client: httpx.AsyncClient,
    access_token: str,
    image_ids: list[str],
    fields: list[str] = None,
    concurrency: int = 10,
) -> list[dict]:
    """
    Fetch metadata for multiple images concurrently.

    Args:
        client: httpx AsyncClient
        access_token: Mapillary access token
        image_ids: List of image IDs
        fields: Fields to request
        concurrency: Max concurrent requests

    Returns:
        List of image metadata dicts
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_one(image_id: str) -> Optional[dict]:
        async with semaphore:
            try:
                return await fetch_image_metadata(
                    client, access_token, image_id, fields
                )
            except MapillaryAPIError:
                return None

    results = await asyncio.gather(*[fetch_one(id) for id in image_ids])
    return [r for r in results if r is not None]


# --- Image Download ---


async def download_image(
    client: httpx.AsyncClient, image_url: str, output_path: Path, timeout: float = 30.0
) -> None:
    """
    Download an image to disk.

    Args:
        client: httpx AsyncClient
        image_url: URL to download from (thumb_*_url from metadata)
        output_path: Local path to save to
        timeout: Download timeout in seconds

    Raises:
        RateLimitError: If rate limit exceeded (429)
        MapillaryAPIError: If download fails (non-200 status)
        httpx.RequestError: If connection/timeout error occurs
    """
    response = await client.get(image_url, timeout=timeout)

    if response.status_code == 429:
        raise RateLimitError(
            "Image download rate limit exceeded",
            status_code=429,
            response_body=response.text,
            request_url=image_url
        )

    if response.status_code != 200:
        raise MapillaryAPIError(
            f"Failed to download image: {response.status_code}",
            status_code=response.status_code,
            response_body=response.text,
            request_url=image_url
        )

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to disk
    with open(output_path, "wb") as f:
        f.write(response.content)
