import time
import os
from tkinter import filedialog

import cv2

from mediapipe_utils import *

POSE_WARNING_RATIO = 0.2
BODY_FACE_RATIO = 0.1

DRAW_ANNOTATION = 0  # {0: bbox, 1: landmark}


def load_filename():
    filename = filedialog.askopenfilename()
    return filename


def get_face_detection_result(image):
    tmp_image = image.copy()

    h, w = tmp_image.shape[:2]
    bbox_face_list = []
    face_detection_results = get_face_detection(tmp_image)
    if face_detection_results:
        for detection in face_detection_results.detections:
            face_bbox_ratio = detection.location_data.relative_bounding_box
            face_bbox = [int(face_bbox_ratio.ymin * h),
                         int(face_bbox_ratio.xmin * w),
                         int((face_bbox_ratio.ymin + face_bbox_ratio.height) * h),
                         int((face_bbox_ratio.xmin + face_bbox_ratio.width) * w)]
            bbox_face_list.append(face_bbox)

    return face_detection_results, bbox_face_list


def get_pose_result(image):
    tmp_image = image.copy()

    h, w = tmp_image.shape[:2]
    old_bbox = []
    bbox_pose_list = []
    pose_results = []
    pose_result = get_pose(tmp_image)
    while pose_result is not None:
        pose_results.append(pose_result)
        bbox = get_pose_bbox(pose_result, h, w)
        if old_bbox == bbox:
            # print(i, 'Same Bbox')
            break
        old_bbox = bbox
        bbox_pose_list.append(bbox)
        tmp_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        pose_result = get_pose(tmp_image)

    return pose_results, bbox_pose_list


def draw_bbox(img, bbox_list):
    for bbox in bbox_list:
        cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), thickness=2)


def warning(data):
    # TODO: 경고 전송
    os.makedirs(os.path.join(data['output_dir'], 'warning'), exist_ok=True)
    cv2.imwrite(os.path.join(data['output_dir'], 'warning', '{}.png'.format(data['i'])), data['image'])


def process(filename, output_dir, output_filename):
    video = cv2.VideoCapture(filename)
    ret, image = video.read()
    if not ret:
        print('Invalid input file: {}'.format(filename))
        return

    os.makedirs(os.path.join('./', output_dir), exist_ok=True)
    output_video = None
    i = 0

    start_time = time.time()
    while ret:
        print(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if output_video is None:
            output_video = cv2.VideoWriter(os.path.join('./', output_dir, output_filename), cv2.VideoWriter_fourcc(*'DIVX'), 25.0, (w, h))

        # Get Annotation
        face_detection_results, bbox_face_list = get_face_detection_result(image)
        pose_results, bbox_pose_list = get_pose_result(image)

        # Draw Annotation
        if DRAW_ANNOTATION:  # landmark
            if face_detection_results:
                draw_face_detection(image, face_detection_results)
            for pose_result in pose_results:
                draw_pose(image, pose_result)
        else:  # bbox
            draw_bbox(image, bbox_face_list)
            draw_bbox(image, bbox_pose_list)

        # Write Video Frame
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_video.write(image)

        # Warning
        # hw_root = (h * w) ** 0.5
        threshold = h if h < w else w
        for bbox in bbox_face_list:
            if bbox[2] - bbox[0] > threshold * POSE_WARNING_RATIO * BODY_FACE_RATIO or bbox[3] - bbox[1] > threshold * POSE_WARNING_RATIO * BODY_FACE_RATIO:
                warning({'i': i, 'image': image, 'output_dir': output_dir})
        for bbox in bbox_pose_list:
            if bbox[2] - bbox[0] > threshold * POSE_WARNING_RATIO or bbox[3] - bbox[1] > threshold * POSE_WARNING_RATIO:
                warning({'i': i, 'image': image, 'output_dir': output_dir})

        i += 1
        ret, image = video.read()
    print('Time: {}'.format(time.time() - start_time))
    video.release()
    output_video.release()


if __name__ == '__main__':
    filename = load_filename()
    if len(filename) > 0:
        ext = os.path.splitext(os.path.basename(filename))[1]
        basename = os.path.basename(filename).replace(ext, '')
        process(filename, 'results_{}'.format(basename), '{}_output.avi'.format(basename))
