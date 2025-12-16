import asyncio
import logging

from mapillary_downloader.api_client import (
    RateLimitError,
    download_image,
    fetch_image_metadata,
)
from mapillary_downloader.iterators import (
    get_async_failure_logger,
    retry_n_times,
    run_and_retry_on_exc,
)
from mapillary_downloader.orchestrator.single_tile_meta import (
    save_images_metadata_for_single_tile,
)
from mapillary_downloader.tiles import get_tiles_for_bbox
from tqdm.gui import tqdm

logger = logging.getLogger(__name__)


def save_tiles_for_bbox_in_state(bbox, state):
    tiles = get_tiles_for_bbox(*bbox)
    state.add_tiles(tiles)
    total_tiles = len(tiles)
    logger.info(f"Need to process {total_tiles} tiles")
    return total_tiles


async def discover_phase(bbox, state, client, access_token, rate_limit_delay):
    total_tiles = save_tiles_for_bbox_in_state(bbox, state)
    state.reset_in_progress()
    pending_tiles = state.get_pending_tiles()

    if not pending_tiles:
        logger.info("No pending tiles to discover.")
        return

    logger.info(f"Starting discovery for {len(pending_tiles)} tiles")

    tasks = [
        single_tile_discover(tile, state, client, access_token, rate_limit_delay)
        for tile in pending_tiles
    ]

    for f in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Discovering tiles"
    ):
        await f

    logger.info(f"Discovered images from {total_tiles} tiles.")


async def single_tile_discover(tile, state, client, access_token, rate_limit_delay):
    async def _save():
        await save_images_metadata_for_single_tile(
            tile, client, access_token, state, rate_limit_delay
        )

    await retry_n_times(
        _save,
        5,
        _delay_10_seconds,
        get_async_failure_logger(
            f"failure in getting metadata for tile={tile}, marking as failure"
        ),
    )


async def _delay_10_seconds(*args, **kwargs):
    return await asyncio.sleep(10)


FIELDS_TO_DOWNLOAD = [
    "id",
    "captured_at",
    "computed_geometry",
    "computed_compass_angle",
    "is_pano",
    "camera_type",
    "make",
    "model",
    "height",
    "width",
]


async def download_single_image_with_retry(
    image_id, image_size, state, images_dir, client, access_token
):
    async def _on_failure(e):
        logger.exception(f"Unexpected error downloading image {image_id}")
        state.mark_image_failed(image_id, str(e))

    async def _download():
        return await _download_single_image_rate_limited(
            image_id, image_size, state, images_dir, client, access_token
        )

    return await retry_n_times(_download, 5, _delay_10_seconds, _on_failure)


async def _download_single_image_rate_limited(
    image_id, image_size, state, images_dir, client, access_token
):
    async def _single_save(i_id):
        return await _save_single_image(
            i_id, image_size, state, images_dir, client, access_token
        )

    metadatas = await run_and_retry_on_exc(
        _single_save, [RateLimitError], [image_id], leave_tqdm_progress=False
    )
    return metadatas[0]


async def _save_single_image(
    image_id, image_size, state, images_dir, client, access_token
):
    metadata = await fetch_image_metadata(
        client,
        access_token,
        image_id,
        fields=FIELDS_TO_DOWNLOAD + [image_size],
    )
    image_url = metadata.get(image_size)
    if not image_url:
        state.mark_image_failed(image_id, f"No {image_size} available")
        return

    state.mark_image_downloading(image_id, image_url)

    # Download image
    output_path = images_dir / f"{image_id}.jpg"
    await download_image(client, image_url, output_path)

    state.mark_image_downloaded(image_id, str(output_path))
    return metadata
