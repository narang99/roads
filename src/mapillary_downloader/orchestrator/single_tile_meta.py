# download in chunks

import itertools
import logging

import httpx

from mapillary_downloader.api_client import RateLimitError, fetch_images_in_bbox
from mapillary_downloader.iterators import run_and_retry_on_exc
from mapillary_downloader.tiles import tile_to_small_bboxes

logger = logging.getLogger(__name__)


async def save_images_metadata_for_single_tile(
    tile,
    client: httpx.AsyncClient,
    access_token,
    state,
    rate_limit_delay,
):
    state.mark_tile_started(tile)
    try:
        bboxes = tile_to_small_bboxes(tile)
        all_images = await _get_all_bboxes_metadata(
            bboxes, client, access_token, rate_limit_delay
        )
        uniq_images = _deduplicate_images(all_images)
        state.add_images(uniq_images, tile)
        state.mark_tile_completed(tile, len(uniq_images))
    except Exception as e:
        state.mark_tile_failed(tile, str(e))
        raise Exception(f"Failed to process tile {tile.z}/{tile.x}/{tile.y}") from e


async def _get_all_bboxes_metadata(bboxes, client, access_token, rate_limit_delay):
    async def _runner(bbox):
        return await _single_bbox_metadata(client, access_token, bbox)

    list_of_list_of_images = await run_and_retry_on_exc(
        _runner,
        [RateLimitError],
        bboxes,
    )
    return itertools.chain.from_iterable(list_of_list_of_images)


async def _single_bbox_metadata(client, access_token, bbox):
    images = await fetch_images_in_bbox(
        client,
        access_token,
        *bbox,
        fields=["id", "captured_at", "computed_geometry"],
    )

    # Parse geometry to get lat/lon
    for img in images:
        geom = img.get("computed_geometry", {})
        coords = geom.get("coordinates", [None, None])
        img["lon"] = coords[0] if coords else None
        img["lat"] = coords[1] if len(coords) > 1 else None

    return images


def _deduplicate_images(all_images):
    seen = set()
    unique_images = []
    for img in all_images:
        img_id = str(img.get("id"))
        if img_id not in seen:
            seen.add(img_id)
            unique_images.append(img)
    return unique_images
