
import pytest
from unittest.mock import AsyncMock
from mapillary_downloader.iterators import retry_n_times, run_and_retry_on_exc

class TestRetryNTimes:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        runner = AsyncMock(return_value="success")
        on_retry = AsyncMock()
        on_failure = AsyncMock()

        result = await retry_n_times(runner, 3, on_retry, on_failure)

        assert result == "success"
        runner.assert_called_once()
        on_retry.assert_not_called()
        on_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        runner = AsyncMock(side_effect=[ValueError("fail 1"), "success"])
        on_retry = AsyncMock()
        on_failure = AsyncMock()

        result = await retry_n_times(runner, 3, on_retry, on_failure)

        assert result == "success"
        assert runner.call_count == 2
        on_retry.assert_called_once()
        on_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_failure(self):
        error = ValueError("fail")
        runner = AsyncMock(side_effect=error)
        on_retry = AsyncMock()
        on_failure = AsyncMock(return_value="failed_state")

        result = await retry_n_times(runner, 3, on_retry, on_failure)

        assert result == "failed_state"
        assert runner.call_count == 3
        assert on_retry.call_count == 2
        on_failure.assert_called_once_with(error)

class TestRunAndRetryOnExc:
    @pytest.mark.asyncio
    async def test_process_all_items_successfully(self):
        # This test will likely timeout with the current bug
        processed = []
        async def runner(item):
            processed.append(item)
            return item * 2

        data = [1, 2, 3]
        result = await run_and_retry_on_exc(runner, [], data)

        assert result == [2, 4, 6]
        assert processed == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        # Mocking asyncio.sleep to avoid waiting real time
        mock_sleep = AsyncMock()

        # Simplified mock runner
        mock_runner = AsyncMock(side_effect=[1, ValueError("retry"), 2, 3])

        # Pass mock_sleep as sleep_func
        result = await run_and_retry_on_exc(
            mock_runner, [ValueError], [1, 2, 3], sleep_func=mock_sleep
        )

        assert result == [1, 2, 3]
        assert mock_runner.call_count == 4 # 1, 2(fail), 2(ok), 3
        mock_sleep.assert_called_once_with(60)
