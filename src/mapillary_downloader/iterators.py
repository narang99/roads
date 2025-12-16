import asyncio
import logging
from typing import Any, Callable, Coroutine, List, Tuple, Type, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


async def run_and_retry_on_exc(
    runner: Callable[[Any], Coroutine[Any, Any, Any]],
    retry_on: Union[List[Type[Exception]], Tuple[Type[Exception], ...]],
    data: List[Any],
    sleep_func: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
    leave_tqdm_progress: bool = True,
) -> List[Any]:
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


async def retry_n_times(
    runner: Callable[[], Coroutine[Any, Any, Any]],
    times: int,
    on_retry: Callable[[Exception], Coroutine[Any, Any, Any]],
    on_failure: Callable[[Exception], Coroutine[Any, Any, Any]],
) -> Any:
    # swallows exceptions on full failure. make sure you mark it somewhere
    for _ in range(times - 1):
        try:
            return await runner()
        except Exception as ex:
            await on_retry(ex)
    try:
        return await runner()
    except Exception as ex:
        return await on_failure(ex)


async def id_async_fn(*a: Any, **kw: Any) -> None:
    return None


def get_async_failure_logger(
    log_line: str,
) -> Callable[..., Coroutine[Any, Any, None]]:
    async def inner(*a: Any, **kw: Any) -> None:
        logger.exception(log_line)
        return None

    return inner
