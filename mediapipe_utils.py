import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

objectron_shoe = mp_objectron.Objectron(static_image_mode=True, max_num_objects=20, min_detection_confidence=0.5, model_name='Shoe')
holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=2)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)


def draw_objectron_shoe(image, objectron_shoe_result):
    for detected_object in objectron_shoe_result.detected_objects:
        mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)


def draw_holistic(image, holistic_result):
    mp_drawing.draw_landmarks(image, holistic_result.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, holistic_result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, holistic_result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, holistic_result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


def draw_face_detection(image, face_detection_result):
    for detection in face_detection_result.detections:
        mp_drawing.draw_detection(image, detection)


def draw_pose(image, pose_result):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=pose_result.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS)


def get_objectrion_shoe(image):
    result = objectron_shoe.process(image)
    if not result.detected_objects:
        return None
    return result


def get_holisic(image):
    result = holistic.process(image)
    if not result.pose_landmarks:
        return None
    return result


def get_face_detection(image):
    result = face_detection.process(image)
    if not result.detections:
        return None
    return result


def get_pose(image):
    result = pose.process(image)
    if not result.pose_landmarks:
        return None
    return result


def get_pose_bbox(pose_result, h, w):
    xs, ys = [], []
    for l in pose_result.pose_landmarks.landmark:
        xs.append(int(l.x * w))
        ys.append(int(l.y * h))
    xs = np.clip(xs, 0, w)
    ys = np.clip(ys, 0, h)
    return np.min(ys), np.min(xs), np.max(ys), np.max(xs)
