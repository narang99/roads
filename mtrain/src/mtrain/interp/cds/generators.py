"""Synthetic image generators for convolution analysis."""

import numpy as np
import torch
from typing import Tuple, Union


def _parse_size(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert size parameter to (height, width) tuple."""
    if isinstance(size, int):
        return (size, size)
    return size


def black(size: int | tuple[int, int]) -> torch.Tensor:
    h, w = _parse_size(size)
    return torch.zeros(h, w)


def white(size: int | tuple[int, int]) -> torch.Tensor:
    return fill_values(size, 1)


def fill_values(size: int | tuple[int, int], val: float) -> torch.Tensor:
    if val < 0 or val > 1:
        raise Exception(f"val should be: 0 <= val <= 1, got: {val}")
    h, w = _parse_size(size)
    return val * torch.ones(h, w)


def random_noise(size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Generate random Gaussian noise image for testing filter responses to random patterns."""
    h, w = _parse_size(size)
    return torch.randn(h, w)


def concentric_circles(
    size: Union[int, Tuple[int, int]], num_rings: int = 8
) -> torch.Tensor:
    """Generate concentric circles alternating black/white for frequency analysis."""
    h, w = _parse_size(size)
    center_y, center_x = h // 2, w // 2

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Calculate distance from center
    distances = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    # Create rings based on distance
    max_dist = min(h, w) // 2
    ring_width = max_dist / num_rings
    ring_indices = (distances / ring_width).long()

    # Alternate black (0) and white (1) rings
    image = (ring_indices % 2).float()

    return image


def stripes(
    size: Union[int, Tuple[int, int]], angle_deg: float = 0, stripe_width: int = 10
) -> torch.Tensor:
    """Generate alternating stripes at specified angle for orientation selectivity testing."""
    h, w = _parse_size(size)

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Rotate coordinates
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)

    # Create stripes based on rotated x-coordinate
    stripe_pattern = ((x_rot / stripe_width).long() % 2).float()

    return stripe_pattern


def unit_impulse(size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Generate single white pixel on black background for impulse response testing."""
    h, w = _parse_size(size)
    image = torch.zeros(h, w)
    center_y, center_x = h // 2, w // 2
    image[center_y, center_x] = 1.0
    return image


def single_line(
    size: Union[int, Tuple[int, int]],
    angle_deg: float = 0,
    thickness: int = 1,
    background_color: float = 0.0,
    foreground_color: float = 1.0,
) -> torch.Tensor:
    """Generate single line at specified angle for edge detection testing."""
    h, w = _parse_size(size)
    image = torch.full((h, w), background_color)

    center_y, center_x = h // 2, w // 2

    if angle_deg == 0:  # Horizontal line
        y_start = max(0, center_y - thickness // 2)
        y_end = min(h, center_y + thickness // 2 + 1)
        image[y_start:y_end, :] = foreground_color
    elif angle_deg == 90:  # Vertical line
        x_start = max(0, center_x - thickness // 2)
        x_end = min(w, center_x + thickness // 2 + 1)
        image[:, x_start:x_end] = foreground_color
    else:  # Diagonal lines
        angle_rad = np.radians(angle_deg)
        for y in range(h):
            for x in range(w):
                # Distance from line passing through center
                dist = abs(
                    np.cos(angle_rad) * (y - center_y)
                    - np.sin(angle_rad) * (x - center_x)
                )
                if dist < thickness / 2:
                    image[y, x] = foreground_color

    return image


def step_edge(
    size: Union[int, Tuple[int, int]],
    angle_deg: float = 0,
    position: int | None = None,
    background_color: float = 0.0,
    foreground_color: float = 1.0,
) -> torch.Tensor:
    """Generate sharp step edge for edge detection testing."""
    h, w = _parse_size(size)
    image = torch.full((h, w), background_color)

    # Use center if position not specified
    if position is None:
        center_y, center_x = h // 2, w // 2
    else:
        center_y, center_x = position, position

    if angle_deg == 0:  # Horizontal edge
        image[:center_y, :] = foreground_color
    elif angle_deg == 90:  # Vertical edge
        image[:, :center_x] = foreground_color
    elif angle_deg == 45:  # Diagonal edge
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        mask = (y - center_y) < (x - center_x)
        image[mask] = foreground_color
    elif angle_deg == -45:  # Other diagonal
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        mask = (y - center_y) < -(x - center_x)
        image[mask] = foreground_color

    return image


def smooth_transition(size: Union[int, Tuple[int, int]], angle_deg: float = 0, position: int | None = None,
                     transition_width: int = 10, low_color: float = 0.0, high_color: float = 1.0) -> torch.Tensor:
    """Generate smooth gradient transition instead of sharp step edge."""
    h, w = _parse_size(size)
    image = torch.full((h, w), low_color)
    
    # Use center if position not specified
    if position is None:
        center_y, center_x = h // 2, w // 2
    else:
        center_y, center_x = position, position
    
    if angle_deg == 0:  # Horizontal smooth transition
        for i in range(transition_width):
            y_pos = center_y - transition_width // 2 + i
            if 0 <= y_pos < h:
                # Linear interpolation from low to high
                alpha = i / (transition_width - 1)
                color = low_color + alpha * (high_color - low_color)
                image[y_pos, :] = color
        # Set regions beyond transition
        image[:center_y - transition_width // 2, :] = low_color
        image[center_y + transition_width // 2:, :] = high_color
        
    elif angle_deg == 90:  # Vertical smooth transition
        for i in range(transition_width):
            x_pos = center_x - transition_width // 2 + i
            if 0 <= x_pos < w:
                # Linear interpolation from low to high
                alpha = i / (transition_width - 1)
                color = low_color + alpha * (high_color - low_color)
                image[:, x_pos] = color
        # Set regions beyond transition
        image[:, :center_x - transition_width // 2] = low_color
        image[:, center_x + transition_width // 2:] = high_color
    
    return image


def linear_gradient(
    size: Union[int, Tuple[int, int]], angle_deg: float = 0
) -> torch.Tensor:
    """Generate linear gradient for smooth transition testing."""
    h, w = _parse_size(size)

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Project coordinates onto gradient direction
    gradient_coord = x * np.cos(angle_rad) + y * np.sin(angle_rad)

    # Normalize to 0-1 range
    gradient_coord = (gradient_coord - gradient_coord.min()) / (
        gradient_coord.max() - gradient_coord.min()
    )

    return gradient_coord


def checkerboard(
    size: Union[int, Tuple[int, int]], square_size: int = 8
) -> torch.Tensor:
    """Generate checkerboard pattern for frequency analysis."""
    h, w = _parse_size(size)

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Create checkerboard pattern
    checker_y = (y // square_size) % 2
    checker_x = (x // square_size) % 2
    checkerboard_pattern = (checker_y + checker_x) % 2

    return checkerboard_pattern.float()


def filled_circle(
    size: Union[int, Tuple[int, int]], radius_ratio: float = 0.3
) -> torch.Tensor:
    """Generate filled circle for blob detection testing."""
    h, w = _parse_size(size)
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) * radius_ratio / 2

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Calculate distance from center
    distances = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    # Create circle mask
    circle_mask = distances <= radius

    image = torch.zeros(h, w)
    image[circle_mask] = 1.0

    return image


def filled_ellipse(size: Union[int, Tuple[int, int]], orientation: str = "horizontal", 
                  aspect_ratio: float = 2.0, radius_ratio: float = 0.3,
                  background_color: float = 0.0, foreground_color: float = 1.0) -> torch.Tensor:
    """Generate filled ellipse for blob detection testing.
    
    Args:
        orientation: "horizontal" (wider horizontally) or "vertical" (wider vertically)
        aspect_ratio: How elongated the ellipse is (width/height for horizontal, height/width for vertical)
        radius_ratio: Size of ellipse relative to image size
    """
    h, w = _parse_size(size)
    center_y, center_x = h // 2, w // 2
    
    # Calculate ellipse radii
    base_radius = min(h, w) * radius_ratio / 2
    
    if orientation == "horizontal":
        # Wider horizontally
        radius_x = base_radius * aspect_ratio
        radius_y = base_radius
    else:  # vertical
        # Wider vertically  
        radius_x = base_radius
        radius_y = base_radius * aspect_ratio
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Ellipse equation: (x-cx)²/rx² + (y-cy)²/ry² <= 1
    ellipse_eq = ((x - center_x)**2 / radius_x**2) + ((y - center_y)**2 / radius_y**2)
    ellipse_mask = ellipse_eq <= 1.0
    
    image = torch.full((h, w), background_color)
    image[ellipse_mask] = foreground_color
    
    return image
