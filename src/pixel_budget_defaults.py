"""The default value for the Qwen image processor resize shared by Stage1 and Stage2."""

# DEFAULT_IMAGE_MIN_PIXELS: 3136=56*56, only retains the legal lower limit for Qwen; small images are no longer interpolated and enlarged.
DEFAULT_IMAGE_MIN_PIXELS = 56 * 56

# DEFAULT_IMAGE_MAX_PIXELS: The unified maximum image area limit for training and evaluation; larger values result in more visual tokens and higher GPU memory usage.
DEFAULT_IMAGE_MAX_PIXELS = 307200
