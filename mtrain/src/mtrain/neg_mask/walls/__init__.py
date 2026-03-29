from mtrain.neg_mask.crops import get_largest_bbox
import cv2
import numpy as np
from mtrain.neg_mask.crops import bbox_only_mask, get_region_crops, Bbox

def decrease_wall_height_in_pair(pair):
    """given a trash mask and wall mask, we decrease the height of the wall mask at the bottom

    The intent is to find all detected trash stuff which is actually just a part of the wall
    examples include wall paintings, walls with stickers / posters, fungal walls, etc

    mapillary does not give perfect wall masks, otherwise i would simply cut it out
    a lot of trash can be found at the edge of the photo, near the walls (it tends to go that way)
    the intuition is that it lies at the bottom of the wall

    md model does not technically create single object regions of height > 100px
    so if any trash detection is 100px the bottom of the wall it is attached to, then it is not trash

    we assume we are given a mask of a single trash object, and the mask of the wall it is surrounded by
    these are assumed to be single region masks

    - We first find the bbox of the trash object.
    - from the wall mask, we are only interested in the part of the wall horizontally surrounding the trash object
    - so we create a mask from the bbox of the trash object, where the horizontal length is untouched
      - but the vertical length goes from 0 -> max mask shape
    - then we take an intersection of the wall mask with this mask
      - we have the surrounding wall
    - now use cv2.contours + cv2.minAreaRect to get the wall bbox (this gives a rotated rectangle).
      - we use rotated rectangles because walls in a street image are not horizontally straight, perspective makes them look rotated
    - we decrease the height of this bbox by cutting 100px from the bottom
    - we return final wall mask with this height cut (it is also hiorizontally cut now)
    - this returns the pair, trash mask is returned as is with the new wall mask
    """

    trash_mask, wall_mask = pair
    cnt, hierarchy = cv2.findContours(wall_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnt[0])
    box = shrink_bottom_only(rect, 100)
    boxed = cv2.drawContours(np.zeros(wall_mask.shape), [box], 0, 255, -1)

    return (pair[0], boxed.astype(bool) & wall_mask.astype(bool))


def _get_individual_masks(mask):
    mask = mask.astype(np.uint8)
    bboxes = list(get_region_crops(mask))
    return [bbox_only_mask(mask, bb, 2000) for bb in bboxes]


def find_trash_wall_pairs(trash_mask, wall_mask):
    wmasks = _get_individual_masks(wall_mask)
    tmasks = _get_individual_masks(trash_mask)

    tm_and_wm = []
    for tm in tmasks:
        max_sum = 0
        mywall = None
        for wm in wmasks:
            inters = tm & wm
            cursum = inters.sum()
            if max_sum < cursum:
                max_sum = cursum
                mywall = wm
        if mywall is not None:
            tm_and_wm.append((tm, mywall))

    return tm_and_wm


def shrink_bottom_only(rect, pixels_to_remove=100):
    # 1. Get the 4 corner points
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    # 2. Find the two "bottom" points (highest y-coordinates)
    # Sort points by their Y-coordinate (descending)
    indices = np.argsort(box[:, 1])[::-1]
    bottom_indices = indices[:2]
    top_indices = indices[2:]

    # 3. Calculate the direction vector of the side we are shrinking
    # We move from a bottom point toward its corresponding top point
    # We'll use the vector between the highest point and the point most "above" it
    p_bottom = box[bottom_indices[0]]

    # Find which top point is on the same side as p_bottom
    # (Checking distance to find the connected neighbor)
    dist1 = np.linalg.norm(p_bottom - box[top_indices[0]])
    dist2 = np.linalg.norm(p_bottom - box[top_indices[1]])

    p_top = box[top_indices[0]] if dist1 < dist2 else box[top_indices[1]]

    # Create a unit vector pointing from bottom to top
    vector = p_top - p_bottom
    length = np.linalg.norm(vector)

    if length == 0:
        return np.intp(box)

    unit_vector = vector / length

    # 4. Move both bottom points along that unit vector
    # This slides the "bottom bar" up toward the top
    box[bottom_indices[0]] += unit_vector * pixels_to_remove
    box[bottom_indices[1]] += unit_vector * pixels_to_remove

    return np.intp(box)



def make_wall_mask_surround_only_trash_region(pair):
    trash_mask, wall_mask = pair
    bbox = get_largest_bbox(trash_mask)

    full_rect_along_roi_length = cv2.rectangle(
        np.zeros(trash_mask.shape), (bbox.x, 0), (bbox.x2, trash_mask.shape[0]), 255, -1
    )

    # only the wall mask surrounding trash object horizontally
    wall_roi_mask = (
        wall_mask.astype(bool) & full_rect_along_roi_length.astype(bool)
    ).astype(np.uint8)
    return (trash_mask, wall_roi_mask)


def do_trash_and_wall_intersect(pair):
    mask, wallmask = pair
    return (mask & wallmask).sum() > 0


def get_trash_masks_which_are_part_of_wall(full_trash_mask, full_wall_mask):
    part_of_wall_masks = []
    pairs = find_trash_wall_pairs(full_trash_mask, full_wall_mask)
    for pair in pairs:
        pair = make_wall_mask_surround_only_trash_region(pair)
        pair = decrease_wall_height_in_pair(pair)
        # part_of_wall_masks.append(pair)

        if do_trash_and_wall_intersect(pair):
            part_of_wall_masks.append(pair[0].astype(bool))
    return part_of_wall_masks
