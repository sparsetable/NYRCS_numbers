"""
Uses MLP to make predictions on what number a drawing is.
"""

def recognise(image: list) -> list:
	"""[CURRENTLY NOT IMPLEMENTED! RETURNS A HARDCODED LIST!]
	Takes in the image of a drawing of a number, and uses the MLP to make predictions of what number it is.
	Arguments:
	- image is a 28 x 28 2d array, where image[y][x] is a number from 0 to 255 (float) indicating pixel darkness from darkest to lightest
	- image[0][0] is the top left corner of the image. Basically the conventional image format

	Return value: A list indicating [percentage probability it is 0, is 1, is 2, is 3, ...]
	"""
	return [10, 10, 10, 0, 0, 20, 20, 10, 10, 10]

SHADES = "█▓▒░ "
def image_str(image: list, threshold = 50) -> None:
	"""Converts the image array to a printable string"""
	result = ""
	for row in image:
		for pixel in row:
			if pixel > round(256 * (4/5)):
				result += shades[0]
			elif pixel > round(256 * (3/5)):
				result += shades[1]
			elif pixel > round(256 * (2/5)):
				result += shades[2]
			elif pixel > round(256 * (1/5)):
				result += shades[1]
			else:
				result += shades[0]
		result += '\n'
	return result