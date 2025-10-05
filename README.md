1. The actual implementation (src/processing.py)

You need to define the functions that the tests import:

detect_edges

apply_gaussian_blur

filter_colors

find_contours

draw_contours

resize_image

convert_to_grayscale

Right now, your tests expect them to:

Raise ValueError on invalid input (empty arrays, wrong types, etc.).

Return correct shapes/dtypes (grayscale is 2D, binary mask is np.uint8 with {0,255}, etc.).

Handle kernel size validation, colorspaces, aspect ratio logic.

Without this file, pytest will error with ImportError.
