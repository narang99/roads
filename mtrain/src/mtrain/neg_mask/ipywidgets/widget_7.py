import json
import shutil
from pathlib import Path
from typing import List, Optional, Union

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

from mtrain.neg_mask.ipywidgets.done_tracker import DoneTracker, SkippedTracker
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes


class InteractiveBBoxEditor:
    """
    Interactive bounding box editor using matplotlib with ipympl backend.
    Allows moving, resizing, and creating new bounding boxes.
    """
    
    def __init__(self, image_path: Path, bboxes: List[dict], on_save_callback=None):
        """
        Initialize the interactive editor.
        
        Args:
            image_path: Path to the image file
            bboxes: List of existing bounding boxes
            on_save_callback: Function to call when saving (gets updated bboxes)
        """
        self.image_path = image_path
        self.on_save_callback = on_save_callback
        self.patches = []
        self.selected_patch = None
        self.dragging = False
        self.resize_handle = None
        
        # Load image
        self.image = Image.open(image_path)
        
        # Setup matplotlib figure with ipympl
        plt.ioff()  # Turn off interactive mode temporarily
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.toolbar_visible = True
        
        # Display image
        self.ax.imshow(np.array(self.image))
        self.ax.set_title(f"Edit Bounding Boxes - {image_path.name}")
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Add rectangle selector for new boxes
        self.selector = RectangleSelector(
            self.ax,
            self._on_new_rectangle,
            useblit=False,
            button=[1],  # Left mouse button only
            minspanx=10, minspany=10,
            spancoords='pixels',
            interactive=False
        )
        self.selector.set_active(False)  # Start inactive
        
        # Add control buttons
        self._create_control_buttons()
        
        plt.ion()  # Turn interactive mode back on
        plt.show()
    
    def _create_bbox_patches(self):
        """Create matplotlib patches for existing bounding boxes."""
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, bbox in enumerate(self.bboxes):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none',
                picker=True, gid=f'bbox_{i}'
            )
            
            self.patches.append(rect)
            self.ax.add_patch(rect)
            
            # Add text label
            self.ax.text(x1, y1-10, f'Box {i+1}', color=color, fontweight='bold')
    
    def _setup_event_handlers(self):
        """Setup matplotlib event handlers for interaction."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
    
    def _create_control_buttons(self):
        """Create control buttons for the editor."""
        # Create button layout
        button_layout = widgets.HBox([
            widgets.Button(description="Add Box", button_style="info", icon="plus"),
            widgets.Button(description="Delete Selected", button_style="danger", icon="trash"),
            widgets.Button(description="Save", button_style="success", icon="check"),
            widgets.Button(description="Cancel", button_style="warning", icon="times")
        ])
        
        # Wire up button callbacks
        button_layout.children[0].on_click(lambda x: self._toggle_add_mode())
        button_layout.children[1].on_click(lambda x: self._delete_selected())
        button_layout.children[2].on_click(lambda x: self._save_changes())
        button_layout.children[3].on_click(lambda x: self._cancel_editing())
        
        display(button_layout)
    
    def _on_pick(self, event):
        """Handle patch selection."""
        if hasattr(event.artist, 'gid') and event.artist.gid.startswith('bbox_'):
            self._select_patch(event.artist)
    
    def _select_patch(self, patch):
        """Select a patch for editing."""
        # Deselect previous
        if self.selected_patch:
            self.selected_patch.set_edgecolor(self.selected_patch.get_edgecolor())
            
        # Select new
        self.selected_patch = patch
        self.selected_patch.set_linewidth(3)  # Highlight selected
        self.fig.canvas.draw()
    
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        
        if self.selected_patch and self.selected_patch.contains(event)[0]:
            self.dragging = True
            self._start_drag = (event.xdata, event.ydata)
            self._start_pos = (self.selected_patch.get_x(), self.selected_patch.get_y())
    
    def _on_release(self, event):
        """Handle mouse release events."""
        if self.dragging:
            self.dragging = False
            self._update_bbox_from_patch()
    
    def _on_motion(self, event):
        """Handle mouse motion events."""
        if not self.dragging or not self.selected_patch or event.inaxes != self.ax:
            return
        
        # Calculate movement
        dx = event.xdata - self._start_drag[0]
        dy = event.ydata - self._start_drag[1]
        
        # Update patch position
        new_x = self._start_pos[0] + dx
        new_y = self._start_pos[1] + dy
        
        # Keep within image bounds
        img_w, img_h = self.image.size
        new_x = max(0, min(new_x, img_w - self.selected_patch.get_width()))
        new_y = max(0, min(new_y, img_h - self.selected_patch.get_height()))
        
        self.selected_patch.set_x(new_x)
        self.selected_patch.set_y(new_y)
        self.fig.canvas.draw()
    
    def _on_new_rectangle(self, eclick, erelease):
        """Handle new rectangle creation."""
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        
        # Add new bbox
        new_bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        self.bboxes.append(new_bbox)
        
        # Create new patch
        i = len(self.patches)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color = colors[i % len(colors)]
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none',
            picker=True, gid=f'bbox_{i}'
        )
        
        self.patches.append(rect)
        self.ax.add_patch(rect)
        self.ax.text(x1, y1-10, f'Box {i+1}', color=color, fontweight='bold')
        
        # Disable selector after creation
        self.selector.set_active(False)
        self.fig.canvas.draw()
    
    def _toggle_add_mode(self):
        """Toggle rectangle selector for adding new boxes."""
        self.selector.set_active(not self.selector.active)
        if self.selector.active:
            print("Click and drag to create new bounding box")
        else:
            print("Add mode disabled")
    
    def _delete_selected(self):
        """Delete the currently selected bounding box."""
        if not self.selected_patch:
            print("No bounding box selected")
            return
        
        # Find index of selected patch
        patch_idx = None
        for i, patch in enumerate(self.patches):
            if patch is self.selected_patch:
                patch_idx = i
                break
        
        if patch_idx is not None:
            # Remove from display
            self.selected_patch.remove()
            
            # Remove from data structures
            self.patches.pop(patch_idx)
            self.bboxes.pop(patch_idx)
            
            # Clear selection
            self.selected_patch = None
            
            # Refresh display
            self._refresh_labels()
            self.fig.canvas.draw()
            print(f"Deleted bounding box {patch_idx + 1}")
    
    def _update_bbox_from_patch(self):
        """Update bounding box coordinates from patch position."""
        if not self.selected_patch:
            return
        
        # Find corresponding bbox
        patch_idx = None
        for i, patch in enumerate(self.patches):
            if patch is self.selected_patch:
                patch_idx = i
                break
        
        if patch_idx is not None:
            # Update bbox coordinates
            x = self.selected_patch.get_x()
            y = self.selected_patch.get_y()
            w = self.selected_patch.get_width()
            h = self.selected_patch.get_height()
            
            self.bboxes[patch_idx] = {
                'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h
            }
    
    def _refresh_labels(self):
        """Refresh all text labels."""
        # Clear existing text
        for txt in self.ax.texts[:]:
            if 'Box' in txt.get_text():
                txt.remove()
        
        # Re-add labels
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, patch in enumerate(self.patches):
            color = colors[i % len(colors)]
            x = patch.get_x()
            y = patch.get_y()
            self.ax.text(x, y-10, f'Box {i+1}', color=color, fontweight='bold')
    
    def _save_changes(self):
        """Save the current bounding boxes."""
        # Update any pending changes
        if self.selected_patch:
            self._update_bbox_from_patch()
        
        print(f"Saving {len(self.bboxes)} bounding boxes")
        
        if self.on_save_callback:
            self.on_save_callback(self.bboxes)
        
        plt.close(self.fig)
    
    def _cancel_editing(self):
        """Cancel editing without saving."""
        print("Cancelled editing")
        plt.close(self.fig)


class ImageReviewWidget:
    """
    Simple widget for reviewing images with keep/skip functionality.
    
    Shows images from a directory, allows user to keep or skip them.
    Kept images are copied to out_dir, skipped images are copied to skipped directory.
    Uses done tracking to skip already processed images.
    """

    def __init__(self, image_dir: Union[str, Path], out_dir: Union[str, Path], skip_dir: Union[str, Path], bbox_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the image review widget.
        
        Args:
            image_dir: Directory containing images to review
            out_dir: Directory where kept images should be copied
            skip_dir: Directory where skipped images should be copied
            bbox_dir: Optional directory containing JSON bounding box files
        """
        self._image_dir = Path(image_dir)
        self._out_dir = Path(out_dir)
        self._skipped_dir = Path(skip_dir)
        self._bbox_dir = Path(bbox_dir) if bbox_dir else None
        
        # Create output directories
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._skipped_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trackers
        self._done_tracker = DoneTracker(self._out_dir.parent)
        self._skipped_tracker = SkippedTracker(self._out_dir.parent)
        
        # Bounding box size tracking
        self._bbox_size_sum = [0.0, 0.0]  # [width_sum, height_sum]
        self._bbox_count = 0
        self._load_bbox_statistics()  # Load existing stats from saved bboxes
        
        # Get list of image files
        self._image_paths = self._get_image_paths()
        self._current_index = 0
        
        # UI components (initialized in ui())
        self._image_display = None
        self._info_label = None
        self._keep_button = None
        self._skip_button = None
        self._progress_label = None

    def _get_image_paths(self) -> List[Path]:
        """Get list of image paths from the directory, excluding already processed ones."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(self._image_dir.glob(f'*{ext}'))
            all_images.extend(self._image_dir.glob(f'*{ext.upper()}'))
        
        # Filter out already processed images
        unprocessed_images = []
        for img_path in sorted(all_images):
            filename = img_path.name
            if not self._done_tracker.is_done(filename) and not self._skipped_tracker.is_skipped(filename):
                unprocessed_images.append(img_path)
        
        return unprocessed_images

    def _load_bbox_statistics(self):
        """Load running statistics from existing bounding box files."""
        if not self._bbox_dir or not self._bbox_dir.exists():
            return
        
        total_width = 0.0
        total_height = 0.0
        total_count = 0
        
        for json_file in self._bbox_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                bboxes = data.get('bboxes', [])
                for bbox in bboxes:
                    width = bbox['x2'] - bbox['x1']
                    height = bbox['y2'] - bbox['y1']
                    total_width += width
                    total_height += height
                    total_count += 1
                    
            except Exception as e:
                continue  # Skip files with errors
        
        if total_count > 0:
            self._bbox_size_sum = [total_width, total_height]
            self._bbox_count = total_count
            avg_w, avg_h = self._get_average_bbox_size()
            print(f"Loaded bbox statistics: {total_count} boxes, avg size: {avg_w:.0f}x{avg_h:.0f}")

    def _get_average_bbox_size(self) -> tuple:
        """Get the current average bounding box size."""
        if self._bbox_count == 0:
            # Default to 1/5th of a typical image size (assume ~800x600)
            return (160.0, 120.0)
        
        avg_width = self._bbox_size_sum[0] / self._bbox_count
        avg_height = self._bbox_size_sum[1] / self._bbox_count
        return (avg_width, avg_height)

    def _add_bbox_to_statistics(self, bbox: dict):
        """Add a single bbox to the running statistics."""
        width = bbox['x2'] - bbox['x1'] 
        height = bbox['y2'] - bbox['y1']
        self._bbox_size_sum[0] += width
        self._bbox_size_sum[1] += height
        self._bbox_count += 1

    def _create_default_bbox(self, image_size: tuple) -> dict:
        """
        Create a default bounding box in the center of the image.
        
        Args:
            image_size: (width, height) of the image
            
        Returns:
            Default bounding box dictionary
        """
        img_width, img_height = image_size
        
        # Get average bbox size or default to 1/5th of image size
        if self._bbox_count == 0:
            bbox_width = img_width / 5.0
            bbox_height = img_height / 5.0
        else:
            avg_width, avg_height = self._get_average_bbox_size()
            # Scale average size to this image's proportions if needed
            bbox_width = min(avg_width, img_width * 0.8)  # Don't exceed 80% of image
            bbox_height = min(avg_height, img_height * 0.8)
        
        # Center the bbox
        center_x = img_width / 2.0
        center_y = img_height / 2.0
        
        x1 = center_x - bbox_width / 2.0
        y1 = center_y - bbox_height / 2.0
        x2 = center_x + bbox_width / 2.0
        y2 = center_y + bbox_height / 2.0
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1) 
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        return {
            'x1': x1,
            'y1': y1, 
            'x2': x2,
            'y2': y2
        }

    def _load_bounding_boxes(self, image_path: Path) -> List[dict]:
        """
        Load bounding boxes for an image if they exist, otherwise create a default one.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of bounding box dictionaries with x1, y1, x2, y2 coordinates
        """
        if not self._bbox_dir:
            return []
        
        # Try to find corresponding JSON file
        image_stem = image_path.stem
        json_path = self._bbox_dir / f"{image_stem}.json"
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                bboxes = data.get('bboxes', [])
                if bboxes:  # If we have bboxes, return them
                    return bboxes
            except Exception as e:
                print(f"Error loading bboxes for {image_path.name}: {e}")
        
        # No existing bboxes found, create a default one
        try:
            image = Image.open(image_path)
            image_size = image.size  # (width, height)
            default_bbox = self._create_default_bbox(image_size)
            
            # Save the default bbox immediately
            self._save_bounding_boxes(image_path, [default_bbox])
            
            # Add to running statistics
            self._add_bbox_to_statistics(default_bbox)
            
            avg_w, avg_h = self._get_average_bbox_size()
            print(f"Created default bbox for {image_path.name} - size: {default_bbox['x2']-default_bbox['x1']:.0f}x{default_bbox['y2']-default_bbox['y1']:.0f}")
            print(f"Running avg: {avg_w:.0f}x{avg_h:.0f} (from {self._bbox_count} boxes)")
            return [default_bbox]
            
        except Exception as e:
            print(f"Error creating default bbox for {image_path.name}: {e}")
            return []

    def _draw_bounding_boxes(self, image: Image.Image, bboxes: List[dict]) -> Image.Image:
        """
        Draw bounding boxes on an image.
        
        Args:
            image: PIL Image to draw on
            bboxes: List of bounding box dictionaries
            
        Returns:
            PIL Image with drawn bounding boxes
        """
        if not bboxes:
            return image
        
        # Create a copy to avoid modifying original
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label = f"Box {i+1}"
            # Get text size for background
            bbox_text = draw.textbbox((x1, y1-25), label)
            draw.rectangle(bbox_text, fill="red")
            draw.text((x1, y1-25), label, fill="white")
        
        return image_with_boxes

    def ui(self):
        """Display the image review UI."""
        # Initialize UI components
        self._image_display = widgets.Output()
        
        self._info_label = widgets.HTML(
            value="",
            layout=widgets.Layout(margin="10px 0px")
        )
        
        self._keep_button = widgets.Button(
            description="Keep",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="100px", margin="5px")
        )
        self._keep_button.on_click(self._on_keep_clicked)
        
        self._skip_button = widgets.Button(
            description="Skip", 
            button_style="warning",
            icon="times",
            layout=widgets.Layout(width="100px", margin="5px")
        )
        self._skip_button.on_click(self._on_skip_clicked)
        
        self._edit_button = widgets.Button(
            description="Edit Bbox",
            button_style="info", 
            icon="edit",
            layout=widgets.Layout(width="100px", margin="5px")
        )
        self._edit_button.on_click(self._on_edit_clicked)
        
        self._save_next_button = widgets.Button(
            description="Save & Next",
            button_style="success",
            icon="arrow-right",
            layout=widgets.Layout(width="120px", margin="5px")
        )
        self._save_next_button.on_click(self._on_save_next_clicked)
        
        self._prev_button = widgets.Button(
            description="Previous",
            button_style="", 
            icon="arrow-left",
            layout=widgets.Layout(width="100px", margin="5px")
        )
        self._prev_button.on_click(self._on_prev_clicked)
        
        self._progress_label = widgets.HTML(
            value="",
            layout=widgets.Layout(margin="10px 0px")
        )
        
        # Layout - organize buttons in rows
        top_button_row = widgets.HBox([
            self._prev_button,
            self._save_next_button,
            self._skip_button
        ], layout=widgets.Layout(justify_content="center"))
        
        bottom_button_row = widgets.HBox([
            self._keep_button,
            self._edit_button
        ], layout=widgets.Layout(justify_content="center"))
        
        button_box = widgets.VBox([top_button_row, bottom_button_row])
        
        main_box = widgets.VBox([
            self._progress_label,
            self._info_label,
            self._image_display,
            button_box
        ])
        
        display(main_box)
        
        # Display first image
        self._display_current_image()

    def _display_current_image(self):
        """Display the current image and update UI elements."""
        if self._current_index >= len(self._image_paths):
            self._show_completion()
            return
        
        current_path = self._image_paths[self._current_index]
        
        # Update progress
        progress_text = f"Image {self._current_index + 1} of {len(self._image_paths)}"
        self._progress_label.value = f"<b>{progress_text}</b>"
        
        # Load and display image
        try:
            image = Image.open(current_path)
            
            # Load bounding boxes if available
            bboxes = self._load_bounding_boxes(current_path)
            
            # Draw bounding boxes on image if they exist
            if bboxes:
                image = self._draw_bounding_boxes(image, bboxes)
            
            image_array = np.array(image)
            
            # Get image info
            filename = current_path.name
            height, width = image_array.shape[:2]
            bbox_info = f" | <b>Bboxes:</b> {len(bboxes)}" if bboxes else ""
            info_text = f"<b>File:</b> {filename}<br><b>Dimensions:</b> {height} x {width}{bbox_info}"
            self._info_label.value = info_text
            
            # Display image
            self._image_display.clear_output(wait=True)
            with self._image_display:
                # Convert to RGB if needed
                if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                    # Convert RGBA to RGB
                    image_array = image_array[:, :, :3]
                elif len(image_array.shape) == 2:
                    # Convert grayscale to RGB
                    image_array = np.stack([image_array] * 3, axis=2)
                
                # Resize if too large for display
                max_display_size = 600
                if max(height, width) > max_display_size:
                    scale = max_display_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    image_array = np.array(image_resized)
                
                display(widgets.Image(
                    value=arr_to_png_bytes(image_array),
                    format='png',
                    layout=widgets.Layout(max_width="600px", margin="10px auto")
                ))
            
            # Enable buttons
            self._enable_all_buttons()
            
        except Exception as e:
            self._info_label.value = f"<b>Error loading image:</b> {current_path.name}<br>{str(e)}"
            self._image_display.clear_output(wait=True)
            # Move to next image automatically on error
            self._current_index += 1
            self._display_current_image()

    def _on_keep_clicked(self, button):
        """Handle keep button click."""
        if self._current_index >= len(self._image_paths):
            return
        
        current_path = self._image_paths[self._current_index]
        filename = current_path.name
        
        try:
            # Copy image to output directory
            dest_path = self._out_dir / filename
            shutil.copy2(current_path, dest_path)
            
            # Mark as done
            self._done_tracker.mark_done(filename)
            
            print(f"✓ Kept: {filename}")
            
        except Exception as e:
            print(f"✗ Error copying {filename}: {e}")
            return
        
        # Move to next image
        self._move_to_next_image()

    def _on_skip_clicked(self, button):
        """Handle skip button click."""
        if self._current_index >= len(self._image_paths):
            return
        
        current_path = self._image_paths[self._current_index]
        filename = current_path.name
        
        try:
            # Copy image to skipped directory
            dest_path = self._skipped_dir / filename
            shutil.copy2(current_path, dest_path)
            
            # Mark as skipped
            self._skipped_tracker.mark_skipped(filename)
            
            print(f"⚠ Skipped: {filename}")
            
        except Exception as e:
            print(f"✗ Error copying {filename}: {e}")
            return
        
        # Move to next image
        self._move_to_next_image()

    def _on_save_next_clicked(self, button):
        """Handle save & next button click - save current bboxes and move to next image."""
        if self._current_index >= len(self._image_paths):
            return
        
        current_path = self._image_paths[self._current_index]
        
        # Get current bboxes (either existing or auto-generated)
        current_bboxes = self._load_bounding_boxes(current_path)
        
        # Save the current bboxes (they may have been edited or are the auto-generated ones)
        if current_bboxes:
            try:
                self._save_bounding_boxes(current_path, current_bboxes)
                print(f"✓ Saved {len(current_bboxes)} bounding boxes for {current_path.name}")
            except Exception as e:
                print(f"✗ Error saving bboxes: {e}")
                return
        
        # Move to next image
        self._move_to_next_image()

    def _on_edit_clicked(self, button):
        """Handle edit button click - open interactive bbox editor."""
        if self._current_index >= len(self._image_paths):
            return
        
        current_path = self._image_paths[self._current_index]
        
        # Load existing bounding boxes
        bboxes = self._load_bounding_boxes(current_path)
        
        print(f"Opening bbox editor for {current_path.name}")
        print("Make sure you have %matplotlib widget enabled for interactive editing")
        
        # Create save callback
        def on_save_bboxes(updated_bboxes):
            self._save_bounding_boxes(current_path, updated_bboxes)
            # Refresh display to show updated bboxes
            self._display_current_image()
            
            # Print updated statistics info
            if self._bbox_count > 0:
                avg_w, avg_h = self._get_average_bbox_size()
                print(f"Current avg bbox size: {avg_w:.0f}x{avg_h:.0f} (from {self._bbox_count} boxes)")
        
        # Open interactive editor
        try:
            # Ensure matplotlib backend is correct
            import matplotlib
            if matplotlib.get_backend() != 'module://ipympl.backend_nbagg':
                print("Warning: For best interactive experience, run '%matplotlib widget' first")
            
            editor = InteractiveBBoxEditor(current_path, bboxes, on_save_bboxes)
        except Exception as e:
            print(f"Error opening editor: {e}")
            print("Make sure you have ipympl installed: pip install ipympl")

    def _on_prev_clicked(self, button):
        """Handle previous button click - go to previous image."""
        if self._current_index <= 0:
            print("Already at first image")
            return
        
        # Move to previous image
        self._current_index -= 1
        
        # Disable buttons temporarily
        self._disable_all_buttons()
        
        # Display previous image
        self._display_current_image()
        
    def _disable_all_buttons(self):
        """Disable all action buttons temporarily."""
        self._keep_button.disabled = True
        self._skip_button.disabled = True
        self._edit_button.disabled = True
        self._save_next_button.disabled = True
        self._prev_button.disabled = True

    def _enable_all_buttons(self):
        """Enable all action buttons."""
        self._keep_button.disabled = False
        self._skip_button.disabled = False
        self._edit_button.disabled = False
        self._save_next_button.disabled = False
        # Previous button should be enabled unless we're at the first image
        self._prev_button.disabled = (self._current_index <= 0)

    def _save_bounding_boxes(self, image_path: Path, bboxes: List[dict]):
        """
        Save bounding boxes to JSON file.
        
        Args:
            image_path: Path to the image file
            bboxes: List of bounding box dictionaries
        """
        if not self._bbox_dir:
            print("No bbox directory specified - cannot save")
            return
        
        # Create bbox directory if it doesn't exist
        self._bbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file with same name as image
        json_filename = f"{image_path.stem}.json"
        json_path = self._bbox_dir / json_filename
        
        bbox_data = {"bboxes": bboxes}
        
        try:
            with open(json_path, 'w') as f:
                json.dump(bbox_data, f, indent=2)
            print(f"✓ Saved {len(bboxes)} bounding boxes to {json_filename}")
        except Exception as e:
            print(f"✗ Error saving bboxes: {e}")

    def _move_to_next_image(self):
        """Move to the next image."""
        self._current_index += 1
        
        # Disable buttons temporarily
        self._disable_all_buttons()
        
        # Display next image
        self._display_current_image()

    def _show_completion(self):
        """Show completion message when all images are processed."""
        self._progress_label.value = "<b>✓ All images processed!</b>"
        self._info_label.value = ""
        self._image_display.clear_output(wait=True)
        
        # Disable buttons
        self._disable_all_buttons()
        
        # Show summary
        kept_count = self._done_tracker.get_done_count()
        skipped_count = self._skipped_tracker.get_skipped_count()
        total_processed = kept_count + skipped_count
        
        with self._image_display:
            summary_html = f"""
            <div style="text-align: center; padding: 20px;">
                <h3>Processing Complete!</h3>
                <p><b>Total processed:</b> {total_processed}</p>
                <p><b>Kept:</b> {kept_count}</p>
                <p><b>Skipped:</b> {skipped_count}</p>
                <p><b>Kept images saved to:</b> {self._out_dir}</p>
                <p><b>Skipped images saved to:</b> {self._skipped_dir}</p>
            </div>
            """
            display(widgets.HTML(summary_html))

    def get_stats(self) -> dict:
        """Get current processing statistics."""
        return {
            "total_images": len(self._image_paths),
            "current_index": self._current_index,
            "kept_count": self._done_tracker.get_done_count(),
            "skipped_count": self._skipped_tracker.get_skipped_count(),
            "remaining": len(self._image_paths) - self._current_index
        }

    def reset_tracking(self):
        """Reset all tracking (clear done and skipped files)."""
        self._done_tracker.clear_done()
        self._skipped_tracker.clear_done()  # SkippedTracker inherits clear_done
        print("Reset complete. All tracking files cleared.")
        
        # Refresh image list
        self._image_paths = self._get_image_paths()
        self._current_index = 0
        
        if hasattr(self, '_progress_label') and self._progress_label is not None:
            self._display_current_image()