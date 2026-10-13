# Face mesh on a still image

Finds the 478 face landmarks in a photo with MediaPipe's FaceLandmarker and draws the mesh with OpenCV. It runs on one
image, shows the result in a window, and can save it or print the coordinates.

## Installation

I tested it with Python 3.11 and MediaPipe 1.1.0. Other Python versions that MediaPipe supports should work too.

```
pip install -r requirements.txt
```

The landmarker model (about 4 MB) is downloaded into `models/` the first time you run the script.

On a minimal Linux install MediaPipe may fail to import with `libEGL.so.1: cannot open shared object file`. Install the
system libraries and try again:

```
sudo apt install libegl1 libgles2
```

## Usage

```
python main.py                              # bundled sample image, result shown in a window
python main.py photo.jpg                    # your own photo
python main.py photo.jpg --output out.png   # also save the image with the mesh drawn on it
python main.py photo.jpg --no-show          # no window, for scripts and servers
python main.py photo.jpg --indices          # draw the landmark numbers
python main.py photo.jpg --print-landmarks  # print x, y, z of all 478 landmarks
```

The bundled sample, `Work_samples/Selfie_casual.png`, is a screenshot of an earlier run of this script, so it already has a
mesh drawn on it. Use your own photo for a clean result.

## What you get

- 468 face landmarks and 10 iris landmarks, 478 in total, for the first face in the image.
- `x` and `y` are normalised to the image width and height, so they lie between 0 and 1. `z` is a relative depth.
- The mesh is drawn from MediaPipe's tessellation, which connects the landmarks into triangles.

If the image contains no face, the script says so and exits with an error.

## Tests

```
pip install pytest
python -m pytest tests
```

The tests check that the model loads, that 478 landmarks come back inside the image, that an image without a face returns
nothing, and that drawing changes the image. They run on every push (see the CI workflow).

## A note on MediaPipe versions

Earlier versions of this script used `mp.solutions.face_mesh`. That legacy interface has been removed from current MediaPipe
releases, so the old script failed on a fresh install with `No module named 'mediapipe.python'`. This version uses the
FaceLandmarker task API, which is the supported one.

## Licence

MIT, see [LICENSE](LICENSE).
