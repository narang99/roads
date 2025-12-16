import asyncio
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

async def run_and_retry_on_exc(
    runner, retry_on, data: list, sleep_func=asyncio.sleep, leave_tqdm_progress=True,
) -> list:
    idx = 0
    total = len(data)
    result = []
    with tqdm(total=total, leave=leave_tqdm_progress) as pbar:
        while idx < total:
            d = data[idx]
            try:
                result.append(await runner(d))
                idx += 1
                pbar.update(1)
            except tuple(retry_on) as ex:
                # Log error and wait before retrying same bbox
                logger.exception(
                    f"retrying function = {runner.__name__}, reason = caught exception={ex.__class__.__name__}, data={d} idx={idx}"
                )
                await sleep_func(60)
                # Don't increment idx - will retry same bbox
    return result


async def retry_n_times(runner, times, on_retry, on_failure):
    # swallows exceptions on full failure. make sure you mark it somewhere
    for _ in range(times-1):
        try:
            return await runner()
        except Exception as ex:
            await on_retry(ex)
    try:
        return await runner()
    except Exception as ex:
        return await on_failure(ex)
