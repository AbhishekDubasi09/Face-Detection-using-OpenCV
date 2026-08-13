# Face Mesh Detection with Mediapipe and OpenCV

This project detects face mesh landmarks in a static image using Mediapipe and OpenCV.

## Installation

1. Clone this repository.
2. Install dependencies in your Python environment:

    ```
    pip install -r requirements.txt
    ```

## Usage

- Runs out of the box on the bundled sample image at `Work_samples/Selfie_casual.png`.
- To use your own photo, replace that file (or edit the `path_img` variable in `main.py`).
- Run the script:

    ```
    python main.py
    ```

- The output displays face mesh on your image and prints landmark coordinates.

## Dependencies

- mediapipe
- opencv-python
- matplotlib

## License

Licensed under the [MIT License](LICENSE).
