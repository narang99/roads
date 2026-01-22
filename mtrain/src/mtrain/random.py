import random

def random_start_points(background):
    max0, max1 = background[:, :, 0].shape
    start0, start1 = random.randint(0, max0-1), random.randint(0, max1-1)
    return start0, start1


def many_random_start_points(background, n=50):
    return [
        random_start_points(background)
        for _ in range(n)
    ]