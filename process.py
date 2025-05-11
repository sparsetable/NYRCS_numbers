"""
Uses MLP to make predictions on what number a drawing is.
Based on https://www.youtube.com/watch?v=w8yWXqWQYmU
"""
from evolution import *
model = get_made_model()

def image_to_nparray(image):
    """Converts 28*28 image array to the np array for the neural network's X"""
    image_array = np.array(image, dtype=np.float32)
    flat = image_array.reshape(-1, 1)
    normalized = flat / 255.0
    return normalized

def recognise(image: list) -> list:
	"""[CURRENTLY NOT IMPLEMENTED! RETURNS A HARDCODED LIST!]
	Takes in the image of a drawing of a number, and uses the MLP to make predictions of what number it is.
	Arguments:
	- image is a 28 x 28 2d array, where image[y][x] is a number from 0 to 255 (int) indicating pixel darkness from darkest to lightest
	- image[0][0] is the top left corner of the image. Basically the conventional image format

	Return value: A list indicating [percentage probability it is 0, is 1, is 2, is 3, ...]
	"""
	X = image_to_nparray(image)
	Z1, A1, Z2, A2 = forward_prop(*model, X)
	return (A2 * 100).flatten().tolist()

SHADES = "█▓▒░ "
def image_str(image: list, threshold = 50) -> None:
	"""Converts the image array to a printable string"""
	result = ""
	for row in image:
		for pixel in row:
			if pixel > round(256 * (4/5)):
				result += SHADES[0]*2
			elif pixel > round(256 * (3/5)):
				result += SHADES[1]*2
			elif pixel > round(256 * (2/5)):
				result += SHADES[2]*2
			elif pixel > round(256 * (1/5)):
				result += SHADES[3]*2
			else:
				result += SHADES[4]*2
		result += '\n'
	return result