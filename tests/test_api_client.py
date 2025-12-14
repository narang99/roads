"""Tests for API client module with mocked responses."""

import pytest
import httpx
import respx
from pathlib import Path
import tempfile

from mapillary_downloader.api_client import (
    fetch_images_in_bbox,
    fetch_image_metadata,
    download_image,
    MapillaryAPIError,
    RateLimitError,
    GRAPH_API_ROOT,
)


@pytest.fixture
def mock_client():
    """Create an httpx client for testing."""
    return httpx.AsyncClient()


class TestFetchImagesInBbox:
    """Tests for fetch_images_in_bbox function."""
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful bbox image search."""
        mock_response = {
            "data": [
                {"id": "123", "captured_at": 1000000},
                {"id": "456", "captured_at": 1000001},
            ]
        }
        
        respx.get(f"{GRAPH_API_ROOT}/images").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        async with httpx.AsyncClient() as client:
            images = await fetch_images_in_bbox(
                client, "test_token",
                -122.15, 37.42, -122.14, 37.43
            )
        
        assert len(images) == 2
        assert images[0]["id"] == "123"
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Test bbox with no images."""
        mock_response = {"data": []}
        
        respx.get(f"{GRAPH_API_ROOT}/images").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        async with httpx.AsyncClient() as client:
            images = await fetch_images_in_bbox(
                client, "test_token",
                0, 0, 0.01, 0.01
            )
        
        assert len(images) == 0
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test rate limit handling."""
        respx.get(f"{GRAPH_API_ROOT}/images").mock(
            return_value=httpx.Response(429, text="Rate limit exceeded")
        )
        
        async with httpx.AsyncClient() as client:
            with pytest.raises(RateLimitError):
                await fetch_images_in_bbox(
                    client, "test_token",
                    -122.15, 37.42, -122.14, 37.43
                )
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error(self):
        """Test general API error."""
        respx.get(f"{GRAPH_API_ROOT}/images").mock(
            return_value=httpx.Response(500, text="Server error")
        )
        
        async with httpx.AsyncClient() as client:
            with pytest.raises(MapillaryAPIError):
                await fetch_images_in_bbox(
                    client, "test_token",
                    -122.15, 37.42, -122.14, 37.43
                )


class TestFetchImageMetadata:
    """Tests for fetch_image_metadata function."""
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful metadata fetch."""
        mock_response = {
            "id": "123",
            "captured_at": 1000000,
            "thumb_1024_url": "https://example.com/thumb.jpg",
            "computed_geometry": {"type": "Point", "coordinates": [-122.1, 37.5]},
        }
        
        respx.get(f"{GRAPH_API_ROOT}/123").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        async with httpx.AsyncClient() as client:
            metadata = await fetch_image_metadata(
                client, "test_token", "123"
            )
        
        assert metadata["id"] == "123"
        assert "thumb_1024_url" in metadata
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_fields(self):
        """Test requesting specific fields."""
        mock_response = {"id": "123", "is_pano": True}
        
        route = respx.get(f"{GRAPH_API_ROOT}/123").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        
        async with httpx.AsyncClient() as client:
            await fetch_image_metadata(
                client, "test_token", "123",
                fields=["id", "is_pano"]
            )
        
        # Check that fields were passed in query
        assert "fields" in str(route.calls[0].request.url)
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test rate limit on metadata fetch."""
        respx.get(f"{GRAPH_API_ROOT}/123").mock(
            return_value=httpx.Response(429, text="Rate limit")
        )
        
        async with httpx.AsyncClient() as client:
            with pytest.raises(RateLimitError):
                await fetch_image_metadata(client, "test_token", "123")


class TestDownloadImage:
    """Tests for download_image function."""
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_download(self):
        """Test successful image download."""
        image_data = b"fake image content"
        
        respx.get("https://example.com/image.jpg").mock(
            return_value=httpx.Response(200, content=image_data)
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_image.jpg"
            
            async with httpx.AsyncClient() as client:
                await download_image(
                    client,
                    "https://example.com/image.jpg",
                    output_path
                )
            
            assert output_path.exists()
            assert output_path.read_bytes() == image_data
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_failed_download(self):
        """Test failed image download."""
        respx.get("https://example.com/image.jpg").mock(
            return_value=httpx.Response(404)
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_image.jpg"
            
            async with httpx.AsyncClient() as client:
                with pytest.raises(MapillaryAPIError):
                    await download_image(
                        client,
                        "https://example.com/image.jpg",
                        output_path
                    )
            
            assert not output_path.exists()
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_parent_directory(self):
        """Test that parent directories are created."""
        image_data = b"fake image content"
        
        respx.get("https://example.com/image.jpg").mock(
            return_value=httpx.Response(200, content=image_data)
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "image.jpg"
            
            async with httpx.AsyncClient() as client:
                await download_image(
                    client,
                    "https://example.com/image.jpg",
                    output_path
                )
            
            assert output_path.exists()


class TestBboxFormatting:
    """Tests for bbox formatting in API calls."""
    
    @respx.mock
    @pytest.mark.asyncio
    async def test_bbox_parameter_format(self):
        """Test that bbox is formatted correctly."""
        respx.get(f"{GRAPH_API_ROOT}/images").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        
        async with httpx.AsyncClient() as client:
            await fetch_images_in_bbox(
                client, "test_token",
                -122.15, 37.42, -122.14, 37.43
            )
        
        # The request URL should contain the bbox
        request = respx.calls[0].request
        assert "bbox=-122.15%2C37.42%2C-122.14%2C37.43" in str(request.url) or \
               "bbox=-122.15,37.42,-122.14,37.43" in str(request.url)
