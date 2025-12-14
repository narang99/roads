"""
Geocoding module - Convert city names to bounding boxes.

Uses OpenStreetMap Nominatim API (free, no API key required).
"""

import httpx
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    
    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return as (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    
    def __iter__(self):
        yield self.min_lon
        yield self.min_lat
        yield self.max_lon
        yield self.max_lat


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


async def geocode_city(
    client: httpx.AsyncClient,
    city_name: str,
    user_agent: str = "MapillaryDownloader/1.0"
) -> dict:
    """
    Geocode a city name using Nominatim.
    
    This is the raw API function - returns the API response directly.
    Separated for testability.
    
    Args:
        client: httpx AsyncClient instance
        city_name: Name of the city (e.g., "Palo Alto, CA")
        user_agent: User-Agent header (required by Nominatim ToS)
    
    Returns:
        Raw API response as dict
    
    Raises:
        httpx.HTTPError: On network/API errors
        ValueError: If city not found
    """
    response = await client.get(
        NOMINATIM_URL,
        params={
            "q": city_name,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        },
        headers={"User-Agent": user_agent}
    )
    response.raise_for_status()
    
    data = response.json()
    if not data:
        raise ValueError(f"City not found: {city_name}")
    
    return data[0]


def parse_bbox_from_nominatim(data: dict) -> BoundingBox:
    """
    Parse bounding box from Nominatim response.
    
    Args:
        data: Single result from Nominatim API
    
    Returns:
        BoundingBox with coordinates
    """
    # Nominatim returns bbox as [south_lat, north_lat, west_lon, east_lon]
    bbox = data.get("boundingbox", [])
    if len(bbox) < 4:
        # Fallback to point + small area
        lat = float(data["lat"])
        lon = float(data["lon"])
        return BoundingBox(
            min_lon=lon - 0.01,
            min_lat=lat - 0.01,
            max_lon=lon + 0.01,
            max_lat=lat + 0.01
        )
    
    return BoundingBox(
        min_lat=float(bbox[0]),
        max_lat=float(bbox[1]),
        min_lon=float(bbox[2]),
        max_lon=float(bbox[3])
    )


async def get_city_bbox(
    client: httpx.AsyncClient,
    city_name: str
) -> BoundingBox:
    """
    High-level function to get bounding box for a city.
    
    Args:
        client: httpx AsyncClient instance
        city_name: Name of the city
    
    Returns:
        BoundingBox for the city
    """
    data = await geocode_city(client, city_name)
    return parse_bbox_from_nominatim(data)
