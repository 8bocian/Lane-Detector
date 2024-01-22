import argparse
import numpy as np
from cv2 import cv2
import time


def transformPerspective(image, to_rectangle):
    if to_rectangle:
        perspective_transform = cv2.getPerspectiveTransform(roi, transform_roi)
    else:
        perspective_transform = cv2.getPerspectiveTransform(transform_roi, roi)

    warped_image = cv2.warpPerspective(image, perspective_transform, (image_width, image_height))
    return warped_image


def processLines(binary_image):
    n_windows = 9
    window_width = 100
    min_pix_to_change_window_start_position = 50
    left = []
    right = []
    example_image = binary_image.copy()
    example_image = cv2.cvtColor(example_image, cv2.COLOR_GRAY2BGR)

    histogram = np.sum(binary_image[image_height // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    current_left_start = np.argmax(histogram[:midpoint])
    current_right_start = np.argmax(histogram[midpoint:]) + midpoint

    window_height = image_height // n_windows
    non_zero_y, non_zero_x = binary_image.nonzero()

    for part in range(n_windows):
        window_lower_boundary = image_height - (part + 1) * window_height
        window_upper_boundary = image_height - part * window_height
        window_left_min = current_left_start - window_width
        window_right_min = current_right_start - window_width
        window_left_max = current_left_start + window_width
        window_right_max = current_right_start + window_width

        cv2.rectangle(example_image,
                      (window_left_min, window_lower_boundary),
                      (window_left_max, window_upper_boundary),
                      (0, 255, 0))
        cv2.rectangle(example_image,
                      (window_right_min, window_lower_boundary),
                      (window_right_max, window_upper_boundary),
                      (0, 255, 0))

        left_idxs = ((non_zero_y >= window_lower_boundary)
                     & (non_zero_y < window_upper_boundary)
                     & (non_zero_x >= window_left_min)
                     & (non_zero_x < window_left_max)).nonzero()[0]

        right_idxs = ((non_zero_y >= window_lower_boundary)
                      & (non_zero_y < window_upper_boundary)
                      & (non_zero_x >= window_right_min)
                      & (non_zero_x < window_right_max)).nonzero()[0]

        if len(left_idxs) > min_pix_to_change_window_start_position:
            current_left_start = int(np.mean(non_zero_x[left_idxs]))

        if len(right_idxs) > min_pix_to_change_window_start_position:
            current_right_start = int(np.mean(non_zero_x[right_idxs]))

        y = abs(part - n_windows+1) * window_height + window_height//2
        # +20 is only to move the lane to the right to see the line more clearly on the image
        left.append((current_left_start+20, y))
        right.append((current_right_start+20, y))

    lanes = np.array([left, right], dtype=np.int32)

    for idx, lane in enumerate(lanes):
        if idx == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.polylines(example_image, [lane], False, color)

    # cv2.imshow('example', example_image)

    return lanes


def preProcess(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = np.average(image_gray[:, 400:-30]) + np.std(image_gray) * 1.5
    image_bin = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    image_blur = cv2.GaussianBlur(image_bin, (5, 5), 0)
    image_canny = cv2.Canny(image_blur, 100, 200)

    return image_canny, image_gray


def drawLanes(lanes, image_trans):
    image = np.zeros_like(image_trans)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for idx, lane in enumerate(lanes):
        # idx=0 => we are drawing right lane
        # idx=1 => we are drawing left lane
        if idx == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.polylines(image, [lane], False, color, 20)

    return image


def process(image):
    image_canny, image_gray = preProcess(image)

    image_perspective_bin = transformPerspective(image_canny, True)
    lanes = processLines(image_perspective_bin)
    image_with_lines = drawLanes(lanes, image_perspective_bin)
    image_transformed = transformPerspective(image_with_lines, False)
    image_processed = cv2.add(image, image_transformed)

    return image_processed, image_perspective_bin, image_canny


def main(path):
    global image_height, image_width, n_channels_in_image, roi_upper_left_corner, roi_upper_right_corner, roi, transform_roi

    video = cv2.VideoCapture(path)
    total = time.time()
    frames_counter = 0
    while video.isOpened():
        has_returned, frame = video.read()
        if has_returned:
            frames_counter += 1
            start = time.time()

            image_height, image_width, n_channels_in_image = np.array(frame).shape

            roi_upper_left_corner = (int(image_width / 2) - 100, int(image_height / 2) + 90)
            roi_upper_right_corner = (int(image_width / 2) + 100, int(image_height / 2) + 90)

            roi = np.array([[(0, image_height),
                             roi_upper_left_corner,
                             roi_upper_right_corner,
                             (image_width, image_height)]], dtype=np.float32)
            transform_roi = np.array([[0, image_height],
                                      [0, 0],
                                      [image_width, 0],
                                      [image_width, image_height]], dtype=np.float32)

            processed, perspective, image_canny = process(frame)
            cv2.imshow('processed', processed)
            # cv2.imshow('canny', image_canny)

            cv2.waitKey(1)
            stop = time.time()
            print(f"FPS: {1 / (stop - start) + 1e-5}")
        else:
            print(f"NUMBER OF FRAMES: {frames_counter}")
            print(f"TIME: {time.time() - total}")
            break

    video.release()


if __name__ == '__main__':
    global image_height, image_width, n_channels_in_image, roi_upper_left_corner, roi_upper_right_corner, roi, transform_roi
    parser = argparse.ArgumentParser(description="A script with --path argument")
    parser.add_argument("--path", help="Specify the video path", required=True)

    args = parser.parse_args()
    path = args.path

    main(path)
