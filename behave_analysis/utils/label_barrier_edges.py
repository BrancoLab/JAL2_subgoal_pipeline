from loguru import logger


def check_which_barrier_location_is_which_orientation(barrier_location) -> tuple:
    """Given the barrier location edges change in tracking data, check which orientation is north and south

    Args:
        barrier_location (tuple): The barrier location edges"""
    assert len(barrier_location) == 3, "The barrier location must be a tuple of the preflip edge, post flip edge and the center edge"
    logger.info(f"The barrier location pre flip is {barrier_location[0]} and post flip is {barrier_location[1]}")
    if barrier_location[0][0] < 512:  # If the first edge x coordinate is less than 512 then the preflip loc is left edge
        logger.info("The barrier preflip location is the left edge")
        return "left", "right"
    else:  # Otherwise the first is right edge
        logger.info("The barrier preflip location is the right edge")
        return "right", "left"


def convert_left_right_to_pre_post_flip(barrier_location):
    """Convert the barrier location edges from left and right to pre flip and post flip"""
    direction1, direction2 = check_which_barrier_location_is_which_orientation(barrier_location)
    dict_conversion = {"pre_flip": direction1, "post_flip": direction2}
    return dict_conversion
