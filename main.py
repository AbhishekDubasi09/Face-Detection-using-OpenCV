import mediapipe as mp
import cv2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

print(f"Mediapipe version: {mp.__version__}")
print(f"OpenCV version: {cv2.__version__}")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_landmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                      refine_landmarks=True, min_detection_confidence=0.5)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return results, landmarks
    else:
        return results, None

def draw_landmarks(image, result):
    image.flags.writeable = True
    if result.multi_face_landmarks:
        for face_landmark in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

path_img = r'C:\Users\abhib\OneDrive\Desktop\My Original Documents\Selfie_casual.jpg'

img = cv2.imread(path_img)

if img is None:
    print("Failed to load image.")
else:
    result, landmarks = get_landmarks(img)
    draw_landmarks(img, result)

    if landmarks is not None:
        # Optional: Print XYZ coordinates of landmarks
        for landmark in landmarks:
            print('x value:', landmark.x)
            print('y value:', landmark.y)
            print('z value:', landmark.z)

        import matplotlib.pyplot as plt

        # Show face mesh (img must be in RGB for matplotlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title("Scroll to Zoom / Drag to Pan")
        plt.axis("off")
        plt.show()
    image_height, image_width, _ = img.shape

    if landmarks is not None:
        for idx, lm in enumerate(landmarks):
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 0), 1, cv2.LINE_AA)


    else:
        print("No face landmarks detected.")

    cv2.imshow("Full Face Mesh", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
