"""Face mesh landmarks on a still image, using MediaPipe's FaceLandmarker and OpenCV.

    python main.py                              show the mesh on the bundled sample image
    python main.py photo.jpg --output out.png   use your own photo and save the result
    python main.py photo.jpg --no-show          save or print without opening a window
"""
import argparse
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "Work_samples", "Selfie_casual.png")
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MESH_COLOUR = (200, 200, 200)  # BGR


def ensure_model(path=MODEL_PATH):
    """Download the landmarker model on first use."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def get_landmarks(image_bgr, model_path=None):
    """Return the 478 normalised landmarks of the first face, or None if no face is found."""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path or ensure_model()),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return result.face_landmarks[0] if result.face_landmarks else None


def draw_mesh(image_bgr, landmarks, indices=False):
    """Draw the face mesh (and optionally the landmark numbers) on the image in place."""
    h, w = image_bgr.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for edge in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        cv2.line(image_bgr, points[edge.start], points[edge.end], MESH_COLOUR, 1, cv2.LINE_AA)
    if indices:
        for i, (x, y) in enumerate(points):
            cv2.putText(image_bgr, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE, help="path to the image (default: bundled sample)")
    parser.add_argument("--output", help="save the image with the mesh drawn on it")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--indices", action="store_true", help="draw the landmark numbers")
    parser.add_argument("--print-landmarks", action="store_true", help="print x, y, z of every landmark")
    args = parser.parse_args()

    print(f"MediaPipe {mp.__version__}, OpenCV {cv2.__version__}")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read the image: {args.image}")

    landmarks = get_landmarks(img)
    if landmarks is None:
        raise SystemExit("No face found in the image.")
    print(f"{len(landmarks)} landmarks found (468 face points and 10 for the irises).")

    if args.print_landmarks:
        for i, lm in enumerate(landmarks):
            print(f"{i:3d}  x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}")

    draw_mesh(img, landmarks, indices=args.indices)
    if args.output:
        cv2.imwrite(args.output, img)
        print(f"Saved {args.output}")
    if not args.no_show:
        cv2.imshow("Face mesh (press any key to close)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
