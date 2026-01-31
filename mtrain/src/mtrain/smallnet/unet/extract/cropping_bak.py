# from PIL import Image
# from mtrain.tqdm import Progress
# import math
# import numpy as np
# import random
# from mtrain.random import random_filename, random_bool
# from mtrain.smallnet.tfms import PaddedResize
# from pycocotools.coco import COCO


# def create_crops_dataset(
#     ann_file,
#     in_dir,
#     out_dir,
#     images_to_sample,
#     max_pad_scale=4,
#     crops_per_image=10,
#     crop_size=50,
#     max_skew=3,
# ):
#     coco = COCO(ann_file)
#     in_images_dir, in_masks_dir = in_dir / "images", in_dir / "masks"
#     out_images_dir, out_masks_dir = out_dir / "images", out_dir / "masks"
#     out_images_dir.mkdir(parents=True, exist_ok=True)
#     out_masks_dir.mkdir(parents=True, exist_ok=True)

#     images = list(in_images_dir.rglob("*.jpeg"))
#     random.shuffle(images)
#     images = _arr_subset(images, images_to_sample)

#     progress = Progress(len(images), "Create Crops", 5)
#     for i, img_path in enumerate(images):
#         mask_path = in_masks_dir / f"{img_path.stem}.png"
#         res = extract_crops_from_single_image(
#             coco, img_path, mask_path, max_pad_scale, crops_per_image, max_skew
#         )
#         resizer = PaddedResize(crop_size)
#         for img, mask in res:
#             img, mask = resizer(img), resizer(mask)
#             _save_crop(img, mask, out_images_dir, out_masks_dir)
#         progress(i)


# def _arr_subset(arr, n):
#     if n < 0:
#         return arr
#     else:
#         return arr[:n]


# def _save_crop(img, mask, out_images_dir, out_masks_dir):
#     fname = random_filename()
#     Image.fromarray(img, "RGB").save(out_images_dir / f"{fname}.jpeg")
#     Image.fromarray(mask, "L").save(out_masks_dir / f"{fname}.png")


# def extract_crops_from_single_image(
#     coco, img_path, mask_path, max_pad_scale, num_samples, max_skew=3
# ):
#     # you should generally have a high number of num_samples, it helps
#     # in providing a variety of data points
#     if num_samples == 0:
#         return []
#     if num_samples < 0:
#         raise Exception(f"num samples has to be positive, passed = {num_samples}")

#     # first add the sample without any skewing
#     res = [extract_single_crop(coco, img_path, mask_path, max_pad_scale, 1, 1)]
#     num_samples -= 1

#     # add skews
#     while num_samples > 0:
#         horiz_skew = random.choice([-1, 1]) * random.uniform(1, max_skew)
#         vert_skew = random.choice([-1, 1]) * random.uniform(1, max_skew)
#         res.append(
#             extract_single_crop(
#                 coco, img_path, mask_path, max_pad_scale, horiz_skew, vert_skew
#             )
#         )
#         num_samples -= 1
#     return res


# def extract_crop_by_cutting_object(coco, img_path, mask_path, max_pad_scale):
#     # we take a crop by starting or ending with a point inside the object
#     # Pick a random annotation to center the crop around
#     img_id = int(img_path.stem)
#     img_array = np.array(Image.open(img_path))
#     mask_array = np.array(Image.open(mask_path))
#     H, W = img_array.shape[:2]
#     x, y, w, h = _get_box(coco, img_id)

#     crop_mode = _get_crop_mode()
#     coords = _get_crop_coords(crop_mode, W, H, max_pad_scale, x, y, w, h)
#     crop_x1, crop_y1, crop_x2, crop_y2 = coords
#     crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
#     crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
#     return crop_img, crop_mask


# def _get_box(coco, img_id):
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)

#     ann = random.choice(anns)
#     x, y, w, h = ann["bbox"]
#     return x, y, w, h


# def _get_crop_coords(crop_mode, W, H, max_pad_scale, x, y, w, h):
#     corner = (x + random.randint(0, int(0.6 * w)), y + random.randint(0, int(0.6 * h)))

#     horiz_pad = random.randint(1, max_pad_scale * w)
#     vert_pad = random.randint(1, max_pad_scale * h)
#     if crop_mode == "TL":
#         # corner is in top left, the next coordinates are on the right and bottom
#         crop_x1, crop_y1 = corner
#         crop_x2 = x + w + horiz_pad
#         crop_y2 = y + h + vert_pad
#     elif crop_mode == "TR":
#         crop_x2, crop_y1 = corner
#         crop_x1 = x - horiz_pad
#         crop_y2 = y + h + vert_pad
#     elif crop_mode == "BL":
#         crop_x1, crop_y2 = corner
#         crop_x2 = x + w + horiz_pad
#         crop_y1 = y - vert_pad
#     elif crop_mode == "BR":
#         crop_x2, crop_y2 = corner
#         crop_x1 = x - horiz_pad
#         crop_y1 = y - vert_pad
#     else:
#         raise Exception(f"bad crop mode {crop_mode}")

#     crop_x1 = max(0, crop_x1)
#     crop_y1 = max(0, crop_y1)
#     crop_x2 = min(crop_x2, W)
#     crop_y2 = min(crop_x2, H)

#     return crop_x1, crop_y1, crop_x2, crop_y2


# def _get_crop_mode():
#     # which corner should be inside the object to start cropping from
#     top_corner = random_bool()
#     left_corner = random_bool()

#     if top_corner and left_corner:
#         return "TL"
#     elif top_corner and not left_corner:
#         return "TR"
#     elif not top_corner and left_corner:
#         return "BL"
#     elif not top_corner and not left_corner:
#         return "BR"
#     raise Exception(f"top={top_corner} left={left_corner}")


# def extract_single_crop(
#     coco, img_path, mask_path, max_pad_scale, horiz_skew, vert_skew
# ):
#     # horiz_skew: relation between the left and right padding max values
#     # if horiz_skew is negative, then we increase left padding with the skew scale compared to right (fix left max to max_padding, chjange right max to left_max / skew)
#     # same for positive skew

#     # Pick a random annotation to center the crop around
#     img_id = int(img_path.stem)
#     img_array = np.array(Image.open(img_path))
#     H, W = img_array.shape[:2]
#     mask_array = np.array(Image.open(mask_path))
#     x, y, w, h = _get_box(coco, img_id)

#     # create max_padding based on max_pad_scale
#     coords = _get_fully_engulfing_and_padded_crop_coords(W, H, max_pad_scale, horiz_skew, vert_skew, x, y, w, h)
#     crop_x1, crop_y1, crop_x2, crop_y2 = coords

#     # Crop image and mask
#     crop_img = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
#     crop_mask = mask_array[crop_y1:crop_y2, crop_x1:crop_x2]
#     return crop_img, crop_mask


# def _get_fully_engulfing_and_padded_crop_coords(
#     W, H, max_pad_scale, horiz_skew, vert_skew, x, y, w, h
# ):
#     # create max_padding based on max_pad_scale
#     horiz_max_padding = int(max_pad_scale * w)
#     vert_max_padding = int(max_pad_scale * h)

#     # Random padding on each side
#     max_left, max_right = _get_paddings(horiz_max_padding, horiz_skew)
#     max_top, max_bottom = _get_paddings(vert_max_padding, vert_skew)
#     pad_left = random.randint(0, max_left)
#     pad_right = random.randint(0, max_right)
#     pad_top = random.randint(0, max_top)
#     pad_bottom = random.randint(0, max_bottom)

#     # Calculate crop boundaries
#     crop_x1 = max(0, int(x - pad_left))
#     crop_y1 = max(0, int(y - pad_top))
#     crop_x2 = min(W, int(x + w + pad_right))
#     crop_y2 = min(H, int(y + h + pad_bottom))

#     return crop_x1, crop_y1, crop_x2, crop_y2


# def _get_paddings(max_padding, skew):
#     # we return before_padding and after_padding
#     before = max_padding
#     if skew < 0:
#         before = max_padding
#         after = math.floor(before / (-skew))
#     else:
#         after = max_padding
#         before = math.floor(after / skew)
#     return before, after
