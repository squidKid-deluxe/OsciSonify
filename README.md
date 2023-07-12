# OsciSonify

OsciSonify is a Python script designed to ease the process of creating oscilloscope music by converting an image into a WAV file. It performs edge detection, applies a Traveling Salesman Problem (TSP) algorithm to order the white (more than 75% brightness) pixels, normalizes the pixel positions, and generates a WAV file that represents the path formed by the white pixels in the image.

## Features

- Edge detection on input images
- Application of TSP algorithm to order white pixels
- Normalization of pixel positions
- Generation of WAV files representing the pixel path
- Customizable amplitude multiplier and frame rate

## Requirements

- Python 3.x
- Dependencies: `opencv-python`, `numpy`, `scipy`

## Usage

1. Install the required dependencies using `pip`:
    
    ```shell
    pip install -r requirements.txt
    ```

2. Set the path to the input image file in the `IMG_PATH` variable.
3. Set the desired amplitude multiplier in the `AMPLITUDE` variable.
4. Set the desired frame rate in the `FRAMERATE` variable.
5. Run the script:
    
    ```shell
    python3 OsciSonify.py
    ```

6. The generated WAV file will be saved in the same directory as the script with the name `path.wav`.

## Contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please open an issue or submit a pull request.  More effective TSP algorithms are especially encouraged.

