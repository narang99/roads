from mtrain.smallnet.unet.extract.draw import overlay_mask_on_img
from mtrain.random import random_filename
from typing import Literal
from mtrain.example_dir.core import load_npz
from mtrain.neg_mask.crops import get_region_crops, padded_crop, bbox_only_mask
from mtrain.disk import DiskBooleanMask, DiskImage
from mtrain.seg.mapillary import get_mask as get_mapi_mask, Label as MapiLabel
from mtrain.neg_mask.ipywidgets.done_tracker import DoneTracker, SkippedTracker
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

def get_trash_mask(directory, label="md"):
    """Get binary trash mask where trash > other"""
    trash_pred = load_npz(directory / f"negmask-trash-{label}.npz")
    other_pred = load_npz(directory / f"negmask-other-{label}.npz")
    return trash_pred > other_pred


class FindFalsePositiveWidget:
    def __init__(self, dirs: list[Path], dest_dir: Path, pad: int):
        self.dirs = dirs
        self.pad = pad
        self.dest_dir = dest_dir
        self.label_name_to_val = {member.name.lower(): member for member in MapiLabel}
        
        # Initialize trackers
        self.done_tracker = DoneTracker(dest_dir)
        self.skipped_tracker = SkippedTracker(dest_dir)
        
        # Filter out already processed directories
        self.dirs = [d for d in self.dirs if not self.done_tracker.is_done(d.name) and not self.skipped_tracker.is_skipped(d.name)]
        
        # Current state
        self.current_dir_index = 0
        self.current_crop_index = 0
        self.current_crops = []
        self.current_image = None
        self.current_mask = None
        self.current_dir = None
        
        # Create UI
        self.create_ui()
        self.load_current_dir()
        
    def create_ui(self):
        """Create the widget UI"""
        # Info labels
        self.info_label = widgets.HTML(value="")
        
        # Image display areas
        self.output_crop = widgets.Output()
        self.output_full = widgets.Output()
        
        # Buttons
        self.trash_btn = widgets.Button(description="Trash", button_style='danger')
        self.other_btn = widgets.Button(description="Other", button_style='success')
        self.skip_btn = widgets.Button(description="Skip", button_style='warning')
        self.skip_all_btn = widgets.Button(description="Skip All", button_style='warning')
        self.other_all_btn = widgets.Button(description="Other All", button_style='info')
        
        # Button callbacks
        self.trash_btn.on_click(self.on_trash)
        self.other_btn.on_click(self.on_other)
        self.skip_btn.on_click(self.on_skip)
        self.skip_all_btn.on_click(self.on_skip_all)
        self.other_all_btn.on_click(self.on_other_all)
        
        # Layout
        buttons = widgets.HBox([
            self.trash_btn, self.other_btn, self.skip_btn, 
            self.skip_all_btn, self.other_all_btn
        ])
        
        images = widgets.VBox([self.output_full, self.output_crop])
        
        self.widget = widgets.VBox([
            self.info_label,
            images,
            buttons
        ])

    def save_to_dest_dir(self, crop, crop_mask, src_dir, label: Literal["trash", "other"]):
        train_dir = self.dest_dir / "train"
        mask_dir = self.dest_dir / "masks"

        train_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        fname = random_filename()
        mapi_label_name = Path(src_dir).resolve().parent.name
        fname = f"{label}_{mapi_label_name}_{fname}"
        DiskImage.save(crop, train_dir / f"{fname}.jpg")
        DiskBooleanMask.save(crop_mask, mask_dir / f"{fname}.png")

    def get_crops_for_single_dir(self, index):
        d = self.dirs[index]
        mask_md = DiskBooleanMask.load(d / "mask-md.png")
        trash_mask = get_trash_mask(d, "md")
        mapi_pred = DiskBooleanMask.load(d / "mapi.png")

        label = d.resolve().parent.name
        label_val = self.label_name_to_val[label]
        mapi_mask = get_mapi_mask(mapi_pred, label_val)

        mask = mask_md & trash_mask & mapi_mask

        image_path = d / "image.jpg"
        image = DiskImage.load(image_path)
        bboxes = list(get_region_crops(mask))

        res = []
        for bbox in bboxes:
            crop, _, _ = padded_crop(image, bbox, self.pad)
            crop_mask = bbox_only_mask(mask, bbox, self.pad)
            res.append((crop, crop_mask))
        
        return res, image, mask, d
    
    def load_current_dir(self):
        """Load crops for current directory"""
        if self.current_dir_index >= len(self.dirs):
            self.show_completion()
            return
            
        crops, image, mask, directory = self.get_crops_for_single_dir(self.current_dir_index)
        
        if crops is None:
            # No valid data, skip to next directory
            self.current_dir_index += 1
            self.load_current_dir()
            return
            
        self.current_crops = crops
        self.current_image = image
        self.current_mask = mask
        self.current_dir = directory
        self.current_crop_index = 0
        
        self.update_display()
    
    def update_display(self):
        """Update the widget display with current crop and full image"""
        if not self.current_crops:
            self.load_next_dir()
            return
            
        # Update info
        dir_info = f"Directory: {self.current_dir.name} ({self.current_dir_index + 1}/{len(self.dirs)})"
        crop_info = f"Crop: {self.current_crop_index + 1}/{len(self.current_crops)}"
        label_info = f"Label: {self.current_dir.resolve().parent.name}"
        self.info_label.value = f"<b>{dir_info}</b><br>{crop_info}<br>{label_info}"
        
        # Clear previous outputs
        with self.output_crop:
            clear_output(wait=True)
            
        with self.output_full:
            clear_output(wait=True)
        
        # Display current crop
        if self.current_crop_index < len(self.current_crops):
            crop, crop_mask = self.current_crops[self.current_crop_index]
            self.display_crop(crop, crop_mask)
        
        # Display full image with mask
        self.display_full_image()
    
    def display_crop(self, crop, crop_mask):
        """Display the current crop and crop with mask overlay"""
        with self.output_crop:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,15))
            
            # Show crop
            ax1.imshow(crop)
            ax1.set_title("Crop")
            # ax1.axis('off')
            
            # Show crop with mask overlay
            ax2.imshow(crop_mask)
            ax2.set_title("Crop + Mask")
            # ax2.axis('off')

            mask_overlay = overlay_mask_on_img(crop, crop_mask.astype(bool))
            ax3.imshow(mask_overlay)
            
            plt.tight_layout()
            plt.show()
    
    def display_full_image(self):
        """Display the full image with false positive mask overlay"""
        with self.output_full:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15))
            
            # Show full image
            ax1.imshow(self.current_image)
            ax1.set_title("Full Image")
            ax1.axis('off')
            
            # Show image with false positive mask overlay
            ax2.imshow(self.current_image)
            # mask_overlay = np.zeros((*self.current_mask.shape, 4))
            # mask_overlay[self.current_mask, :3] = [1, 0, 0]  # Red
            # mask_overlay[self.current_mask, 3] = 0.5  # Semi-transparent
            mask_overlay = overlay_mask_on_img(self.current_image, self.current_mask.astype(bool))
            ax2.imshow(mask_overlay)
            ax2.set_title("Image + False Positive Mask")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def save_current_crop(self, label: Literal["trash", "other"]):
        """Save the current crop with the given label"""
        if self.current_crop_index < len(self.current_crops):
            crop, crop_mask = self.current_crops[self.current_crop_index]
            self.save_to_dest_dir(crop, crop_mask, self.current_dir, label)
    
    def load_next_crop(self):
        """Move to next crop or next directory"""
        self.current_crop_index += 1
        if self.current_crop_index >= len(self.current_crops):
            self.load_next_dir()
        else:
            self.update_display()
    
    def load_next_dir(self):
        """Move to next directory"""
        self.done_tracker.mark_done(self.dirs[self.current_dir_index].name)
        self.current_dir_index += 1
        self.load_current_dir()
    
    def show_completion(self):
        """Show completion message"""
        self.info_label.value = "<b>All directories processed!</b>"
        with self.output_crop:
            clear_output()
        with self.output_full:
            clear_output()
    
    # Button handlers
    def on_trash(self, btn):
        """Handle Trash button click"""
        self.save_current_crop("trash")
        self.load_next_crop()
    
    def on_other(self, btn):
        """Handle Other button click"""
        self.save_current_crop("other")
        self.load_next_crop()
    
    def on_skip(self, btn):
        """Handle Skip button click - skip current crop"""
        self.load_next_crop()
    
    def on_skip_all(self, btn):
        """Handle Skip All button click - skip entire directory"""
        self.skipped_tracker.mark_skipped(self.current_dir.name)
        self.load_next_dir()
    
    def on_other_all(self, btn):
        """Handle Other All button click - save all crops in directory as 'other'"""
        for crop, crop_mask in self.current_crops:
            self.save_to_dest_dir(crop, crop_mask, self.current_dir, "other")
        self.done_tracker.mark_done(self.current_dir.name)
        self.load_next_dir()
    
    def display(self):
        """Display the widget"""
        display(self.widget)