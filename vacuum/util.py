from tensorflow import image


def shift(i, x=0, y=0):
    """
    Use this to shift your filter for a conv2d, which is probably needed if you do a conv2d with
    even input and/or filter.
    """
    return image.pad_to_bounding_box(
        i,
        max(0, y),
        max(0, x),
        i.shape.as_list()[1] + abs(y),
        i.shape.as_list()[2] + abs(x))