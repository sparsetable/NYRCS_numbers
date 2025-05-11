# NYRCS_numbers
This project will be used for an educational AI demonstration on NY research day (21st May 2025).

Trained on MNIST dataset obtained from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. https://drive.google.com/drive/folders/111m0ZpYKW9ADVDxhy1pkcCZowJjMmffv?usp=sharing. Download and unzip both csvs into this file.

## Implementation
This project will let students draw a number and have an MLP decide what number it looks like.

It will be a flask webapp running locally on the computer at the station. The program is split into:
- `process.py`, the file which implements the AI. Use it as so.
```py
from process import recognise
# process implements recognise(image: list) -> list
# Arguments:
# - image is a 28 x 28 2d array, where image[y][x] is a number from 0 to 255 indicating pixel darkness from darkest to lightest
# - image[0][0] is the top left corner of the image. Basically the conventional image format
# Return value: A list indicating [percentage probability it is 0, is 1, is 2, is 3, ...]
```
- Other file(s), to implement the flask frontent. For other team members to decide and update readme.

## TODO for this program:
1. Split work & implement flask frontend
2. (Francis) implement `process.py`