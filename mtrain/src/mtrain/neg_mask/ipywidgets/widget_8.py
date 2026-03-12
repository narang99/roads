import shutil
import cv2
from pathlib import Path
from typing import Optional

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image

from mtrain.neg_mask.ipywidgets.done_tracker import DoneTracker, SkippedTracker
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes


class ImageReviewWidget:
    """
    Simple widget for reviewing images from a directory.
    
    User can decide to keep (copy to out_dir) or skip (copy to skipped directory) each image.
    Tracks progress using done_tracker.py to skip already processed images.
    """

    def __init__(self, source_dir: str | Path, out_dir: str | Path, skip_dir: str | Path):
        """
        Initialize the image review widget.
        
        Args:
            source_dir: Directory containing images to review
            out_dir: Directory where kept images should be copied
        """
        self._source_dir = Path(source_dir)
        self._out_dir = Path(out_dir)
        
        # Create output directories
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._skipped_dir = Path(skip_dir)
        self._skipped_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trackers
        self._done_tracker = DoneTracker(self._out_dir)
        self._skipped_tracker = SkippedTracker(self._out_dir)
        
        # Get list of image files to process
        self._image_paths = self._get_image_paths()
        self._current_index = 0
        self._current_image: Optional[np.ndarray] = None
        self._current_filename: str = ""
        
        # UI components
        self._image_display = None
        self._resized_image_display = None
        self._crop_image_display = None
        self._info_label = None
        self._shape_label = None
        self._filename_label = None
        self._keep_button = None
        self._skip_button = None
        self._progress_label = None

    def _get_image_paths(self) -> list[Path]:
        """Get list of image file paths from source directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self._source_dir.glob(f"*{ext}"))
            image_paths.extend(self._source_dir.glob(f"*{ext.upper()}"))
        
        # Filter out already processed images
        unprocessed_paths = []
        for path in sorted(image_paths):
            filename = path.name
            if not self._done_tracker.is_done(filename) and not self._skipped_tracker.is_skipped(filename):
                unprocessed_paths.append(path)
        
        return unprocessed_paths

    def _resize(self, image):
        return cv2.resize(
            image,
            (130, 130),
            interpolation=cv2.INTER_AREA,
        )
    
    def _random_crop_64(self, image_130):
        """Extract a random 64x64 crop from a 130x130 image."""
        # Maximum starting position for 64x64 crop in 130x130 image
        max_x = 130 - 64  # 66
        max_y = 130 - 64  # 66
        
        # Generate random starting position
        start_x = np.random.randint(0, max_x + 1)
        start_y = np.random.randint(0, max_y + 1)
        
        # Extract crop
        crop = image_130[start_y:start_y + 64, start_x:start_x + 64]
        return crop

    def _load_current_image(self) -> bool:
        """Load the current image. Returns True if successful."""
        if self._current_index >= len(self._image_paths):
            return False
        
        current_path = self._image_paths[self._current_index]
        self._current_filename = current_path.name
        
        try:
            # Load image as numpy array
            pil_image = Image.open(current_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            self._current_image = np.array(pil_image)
            return True
        except Exception as e:
            print(f"Error loading image {current_path}: {e}")
            return False

    def _update_display(self):
        """Update the image display and labels."""
        if self._current_image is None:
            self._image_display.value = b''
            self._resized_image_display.value = b''
            self._crop_image_display.value = b''
            self._shape_label.value = "Shape: N/A"
            self._filename_label.value = "File: N/A"
            return
        
        # Update original image display
        try:
            image_bytes = arr_to_png_bytes(self._current_image)
            self._image_display.value = image_bytes
        except Exception as e:
            print(f"Error displaying original image: {e}")
            self._image_display.value = b''
        
        # Update resized image display and create crop
        try:
            resized_image = self._resize(self._current_image)
            resized_bytes = arr_to_png_bytes(resized_image)
            self._resized_image_display.value = resized_bytes
            
            # Create and display random 64x64 crop
            crop_64 = self._random_crop_64(resized_image)
            crop_bytes = arr_to_png_bytes(crop_64)
            self._crop_image_display.value = crop_bytes
        except Exception as e:
            print(f"Error displaying resized/crop images: {e}")
            self._resized_image_display.value = b''
            self._crop_image_display.value = b''
        
        # Update labels
        height, width = self._current_image.shape[:2]
        self._shape_label.value = f"Shape: {height} x {width}"
        self._filename_label.value = f"File: {self._current_filename}"
        
        # Update progress
        self._progress_label.value = f"Image {self._current_index + 1} of {len(self._image_paths)}"

    def _on_keep(self, button):
        """Handle keep button click."""
        if self._current_image is None or self._current_index >= len(self._image_paths):
            return
        
        current_path = self._image_paths[self._current_index]
        dest_path = self._out_dir / self._current_filename
        
        try:
            # Copy file to output directory
            shutil.copy2(current_path, dest_path)
            
            # Mark as done
            self._done_tracker.mark_done(self._current_filename)
            
            print(f"Kept: {self._current_filename}")
            self._next_image()
            
        except Exception as e:
            print(f"Error copying file: {e}")

    def _on_skip(self, button):
        """Handle skip button click."""
        if self._current_image is None or self._current_index >= len(self._image_paths):
            return
        
        current_path = self._image_paths[self._current_index]
        dest_path = self._skipped_dir / self._current_filename
        
        try:
            # Copy file to skipped directory
            shutil.copy2(current_path, dest_path)
            
            # Mark as skipped
            self._skipped_tracker.mark_skipped(self._current_filename)
            
            print(f"Skipped: {self._current_filename}")
            self._next_image()
            
        except Exception as e:
            print(f"Error copying file to skipped: {e}")

    def _next_image(self):
        """Move to the next image."""
        self._current_index += 1
        
        if self._current_index >= len(self._image_paths):
            # All images processed
            self._image_display.value = b''
            self._resized_image_display.value = b''
            self._crop_image_display.value = b''
            self._shape_label.value = "Shape: All images processed!"
            self._filename_label.value = "File: Done"
            self._progress_label.value = f"Completed all {len(self._image_paths)} images"
            self._keep_button.disabled = True
            self._skip_button.disabled = True
            print("🎉 All images have been processed!")
            return
        
        # Load and display next image
        if self._load_current_image():
            self._update_display()
        else:
            # Skip problematic image and try next
            self._next_image()

    def ui(self):
        """Display the image review UI."""
        if not self._image_paths:
            print("No unprocessed images found in the source directory.")
            return
        
        # Create UI components
        self._image_display = widgets.Image(
            format='png',
            layout=widgets.Layout(max_width='400px', max_height='400px')
        )
        
        self._resized_image_display = widgets.Image(
            format='png',
            layout=widgets.Layout(width='260px', height='260px')
        )
        
        self._crop_image_display = widgets.Image(
            format='png',
            layout=widgets.Layout(width='128px', height='128px')
        )
        
        self._info_label = widgets.HTML(
            value="<b>Review Images</b><br/>Decide whether to keep or skip each image."
        )
        
        self._shape_label = widgets.Label(value="Shape: Loading...")
        self._filename_label = widgets.Label(value="File: Loading...")
        self._progress_label = widgets.Label(value="Progress: Loading...")
        
        self._keep_button = widgets.Button(
            description="✓ Keep",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="120px")
        )
        self._keep_button.on_click(self._on_keep)
        
        self._skip_button = widgets.Button(
            description="✗ Skip",
            button_style="warning",
            icon="times",
            layout=widgets.Layout(width="120px")
        )
        self._skip_button.on_click(self._on_skip)
        
        # Create layout
        button_box = widgets.HBox(
            [self._keep_button, self._skip_button],
            layout=widgets.Layout(gap="10px", justify_content="center")
        )
        
        info_box = widgets.VBox([
            self._shape_label,
            self._filename_label,
            self._progress_label
        ])
        
        # Create image display row with labels
        original_box = widgets.VBox([
            widgets.HTML("<b>Original</b>"),
            self._image_display
        ], layout=widgets.Layout(align_items="center"))
        
        resized_box = widgets.VBox([
            widgets.HTML("<b>Resized (130x130)</b>"),
            self._resized_image_display
        ], layout=widgets.Layout(align_items="center"))
        
        crop_box = widgets.VBox([
            widgets.HTML("<b>Random Crop (64x64)</b>"),
            self._crop_image_display
        ], layout=widgets.Layout(align_items="center"))
        
        images_row = widgets.HBox([
            original_box,
            resized_box,
            crop_box
        ], layout=widgets.Layout(gap="20px", justify_content="center"))
        
        main_ui = widgets.VBox([
            self._info_label,
            images_row,
            info_box,
            button_box
        ], layout=widgets.Layout(align_items="center", gap="10px"))
        
        # Load first image and display
        if self._load_current_image():
            self._update_display()
        else:
            self._next_image()
        
        display(main_ui)