import random

def random_start_points(background):
    max0, max1 = background[:, :, 0].shape
    start0, start1 = random.randint(0, max0), random.randint(0, max1)
    return start0, start1