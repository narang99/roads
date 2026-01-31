import random
import string

def random_start_points(background):
    max0, max1 = background[:, :, 0].shape
    start0, start1 = random.randint(0, max0-1), random.randint(0, max1-1)
    return start0, start1


def many_random_start_points(background, n=50):
    return [
        random_start_points(background)
        for _ in range(n)
    ]


def add_jitter_pixels(pixels):
    sign = 1 if random_bool() else -1
    return pixels + sign * random.randint(0, 20)


def random_bool():
    return bool(random.getrandbits(1))

def random_true_one_sixth_times():
    is_true = random.randint(0, 5) == 5
    return is_true


def random_true_one_three_times():
    is_true = random.randint(0, 2) == 2
    return is_true


def random_filename(length=8):
    """Generate a random filename without suffix."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))