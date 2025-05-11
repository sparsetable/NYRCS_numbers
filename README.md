# NYRCS_numbers
This project will be used for an educational AI demonstration on NY research day (21st May 2025).

Trained on MNIST dataset obtained from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. https://drive.google.com/drive/folders/111m0ZpYKW9ADVDxhy1pkcCZowJjMmffv?usp=sharing. Download and unzip both csvs into this file.

## Implementation
This project will let students draw a number and have an MLP decide what number it looks like.

It will be a flask webapp running locally on the computer at the station. The program is split into:
- `process.py`, an interface to call the AI. Use it as so:
```py
from process import recognise
# process implements recognise(image: list) -> list
# Arguments:
# - image is a 28 x 28 2d array, where image[y][x] is a number from 0 to 255 indicating pixel darkness from darkest to lightest
# - image[0][0] is the top left corner of the image. Basically the conventional image format
# Return value: A list indicating [percentage probability it is 0, is 1, is 2, is 3, ...]
```
- `ai.py` implements all the ai math. The code is ugly because its all math :pray:
- `evolution.py` implements the evolution algorithm to generate a model. Refer to the `# Training` section for instructions on how to use it.
- Other file(s), to implement the flask frontend. For other team members to decide and update readme.

## Training
Run `evolution.py` to generate a model through an evolution algorithm. You only need ~1000 generations. The script will automatically pick up where the last run ended.

Delete `b1.npy`, `b2.npy`, `W1.npy`, `W2.npy` and `generations.txt` files to restart. Copy over someone else's `b1.npy`, `b2.npy`, `W1.npy`, `W2.npy` and `generations.txt` files into your folder to use their model

## TODO for this program:
1. Split work & implement flask frontend
2. (Francis) implement `process.py`