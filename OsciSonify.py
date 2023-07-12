# pylint: disable=no-member
"""
This script performs edge detection on an image, applies the Traveling Salesman Problem (TSP)
algorithm to order the white pixels, normalizes the pixel positions, and writes the data to
a WAV file. The WAV file represents the path formed by the white pixels in the image.

Usage:
1. Set the path to the input image file in the `IMG_PATH` variable.
2. Set the desired amplitude multiplier in the `AMPLITUDE` variable.
3. Set the desired frame rate in the `FRAMERATE` variable.
4. Run the script.

Note: This script requires the following dependencies: `struct`, `wave`, `cv2`, `numpy`,
and `scipy.spatial.cKDTree`.

Functions:
- tsp_ordered_white_pixels(image): Applies the TSP algorithm to order the white pixels 
    in the input image.
- on_trackbar(image): Displays a preview of the edge detection and allows manual 
    adjustment of threshold values.
- edge_detection(image): Performs edge detection on the input image using the Canny algorithm.
- normalize(data): Normalizes the input data to the (-1, 1) range and extends it to
    at least 10 seconds.
- write_wave_file(list_x, list_y): Writes the normalized stereo data to a WAV file.
- main(): The main function that orchestrates the image processing and WAV file generation.

The script is designed to be executed directly and will
generate a WAV file based on the specified image.

Author: SquidKid-deluxe
Date: 7/2023
"""
import struct
import wave

import cv2
import numpy as np
from scipy.spatial import cKDTree

FILENAME = "path.wav"
AMPLITUDE = 64000.0  # multiplier for amplitude; max 2**16, 65535
FRAMERATE = 48000  # 44100 CD grade; 48000 DVD grade; 96000 Oscilloscope music grade
# Same data gets packed whatever the framerate, it just changes the pitch
# I think samplers honor the above framerate though, so 96000 (or higher) may be better
IMG_PATH = "test_path.png"


def tsp_ordered_white_pixels(image):
    """
    Using a K-Nearest neighbor method, apply the Traveling Salesman Problem
    to the numpy array (image) of black and white pixels
    issue with this is the "Salesman" might hit a dead end and jump unnecessarily far
    """
    # Get the indices of white pixels in the image
    white_pixels = np.argwhere(image > 190)

    # Build a KD-tree using the white pixels
    kdtree = cKDTree(white_pixels)

    # Initialize the visited set and ordered list
    visited = set()
    ordered_list = []

    # Randomly choose a starting point
    sigkill = False

    # find first pixel
    for rowdx, row in enumerate(image):
        for pixdx, pix in enumerate(row):
            if pix > 190:
                current_point = white_pixels.tolist().index([rowdx, pixdx])
                sigkill = True
                break
        if sigkill:
            break
    # FIXME: Above is deterministic, but is it better to choose random?
    # current_point = np.random.choice(len(white_pixels))
    visited.add(current_point)
    ordered_list.append(white_pixels[current_point])

    # Iterate until all points are visited
    while len(visited) < len(white_pixels):
        # print status
        if not len(visited) % (len(white_pixels) // 100):
            print(len(visited) / (len(white_pixels) // 100), "%")
        # Query the nearest neighbor
        _, nearest_idx = kdtree.query(white_pixels[current_point], k=len(white_pixels))

        # Find the unvisited nearest neighbor
        for idx in nearest_idx:
            if idx not in visited:
                visited.add(idx)
                ordered_list.append(white_pixels[idx])
                current_point = idx
                break

    return ordered_list


def on_trackbar(image):
    """
    preview/show function for manual adjustment of edge detection
    NOTE: Could probably do this in a lambda
        just put the `getTrackbarPos` call in the `Canny` call
        but this is more readable
    """
    mint = cv2.getTrackbarPos("min thresh", "preview")
    maxt = cv2.getTrackbarPos("max thresh", "preview")

    cv2.imshow("preview", cv2.Canny(image, mint, maxt))


def edge_detction(image):
    """
    Helper function to deal with making an OpenCV GUI
    and waiting for user to finish customization
    """
    # tell OpenCV what our window name is going to be so it can keep track
    cv2.namedWindow("preview")

    # Resize the image to either it's existing size or smaller if necessary
    # Either way, the whole thing gets normalized -1 - 1 during wave file writing
    image = cv2.resize(
        image, (min(300, image.shape[0]), min(300, image.shape[1])), cv2.INTER_LINEAR
    )

    # make two trackbars, one each for minimum and maximum thresholds of the cv2.Canny
    # see docs.opencv.org/3.4/da/d22/tutorial_py_canny.html, section "Theory:5"
    cv2.createTrackbar("min threshold", "preview", 0, 255, lambda _: on_trackbar(image))
    cv2.createTrackbar("max threshold", "preview", 0, 255, lambda _: on_trackbar(image))

    # Show preview
    cv2.imshow("preview", cv2.Canny(image, 0, 0))
    # Wait for done (d key)
    while True:
        if cv2.waitKey(30) == ord("d"):
            break
    #   vvvvvvvvvvvvvvvvv
    cv2.destroyAllWindows()

    # Get the users final trackbar positions and use them
    mint = cv2.getTrackbarPos("min thresh", "preview")
    maxt = cv2.getTrackbarPos("max thresh", "preview")

    return cv2.Canny(image, mint, maxt)


def normalize(data):
    """
    normalize the positions to (-1, 1) range and extend to >= 10 secs
    """
    data = ((data / np.max(data)) * 2) - 1
    while len(data) < 480000:
        data = np.append(data, data)
    return data


def write_wave_file(list_x, list_y):
    """
    Deal with `wave`
    take normalized lists of stereo data
    scale them to int16
    pack them into bytes
    write them
    """
    # Open the wave file
    wav_file = wave.open(FILENAME, "w")

    # set all the attributes of it
    nchannels = 2
    sampwidth = 2
    framerate = int(FRAMERATE)
    nframes = len(list_x)
    comptype = "NONE"
    compname = "not compressed"

    wav_file.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))

    # Scale the arrays in a single operation
    scaled_x = (list_x * AMPLITUDE / 2).astype(np.int16)
    scaled_y = (list_y * AMPLITUDE / 2).astype(np.int16)

    # Combine the scaled arrays into a single interleaved array
    interleaved_frames = np.column_stack((scaled_y, scaled_x)).flatten()

    # Pack the interleaved frames using the appropriate format
    packed_frames = struct.pack(f"{len(interleaved_frames)}h", *interleaved_frames)

    # Write the packed frames to the WAV file
    wav_file.writeframes(packed_frames)

    # FIXME: should ^^^ be in a `with` statement? Can `wave.open` objects do that?
    # Close the file
    wav_file.close()


def main():
    """
    Main "table of contents" function
    """
    # Read image
    image = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2GRAY)

    # Apply TSP to the image, convert to numpy array and rotate
    path = np.array(tsp_ordered_white_pixels(image)).T

    # normalize the positions to (-1, 1) range and extend to >= 10 secs for both x and y
    list_x = normalize(path[0])*-1
    list_y = normalize(path[1])

    # write to file
    write_wave_file(list_x, list_y)


if __name__ == "__main__":
    main()
