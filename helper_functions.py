#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Interest Points and Descriptors
Matthias Hirschmanner 2024
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import Tuple, List

# Adapted from https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py#L41
def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """ Rotate an image by the angle and return it with the additional pixels filled with replicated border

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param angle: The angle the image will be rotated in degree
    :type angle: float

    :return: Resulting image with the rotated original image and filled borders
    :rtype: np.ndarray with shape (new_height, new_width) with dtype np.float32 an values in range [0., 1.]
    """
    # grab the dimensions of the image and then determine the
    # center
    (height, width) = img.shape[:2]
    (center_x, center_y) = (width // 2, height // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    # compute the new bounding dimensions of the image
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - center_x
    rotation_matrix[1, 2] += (new_height / 2) - center_y
    # perform the actual rotation and return the image
    return cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_REPLICATE)

def show_image(img: np.ndarray, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.ndarray with shape (height, width) or (height, width, channels)

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    :type use_matplotlib: bool
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.

    elif img.dtype == np.float64:
        img = img.astype(np.float32)

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        if is_color:
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) * 255.).astype(np.uint8)
        else:
            png_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA) * 255.).astype(np.uint8)
        cv2.imwrite(title.replace(" ", "_") + ".png", png_img)


def draw_rectangles(scene_img: np.ndarray,
                    object_img: np.ndarray,
                    homography: np.ndarray) -> np.ndarray:
    """Plot rectangles with size of object_img into scene_img given the homography transformation matrix

    :param scene_img: Image to draw rectangles into
    :type scene_img: np.ndarray with shape (height, width, channels)

    :param object_img: Image of the searched object which defines the size of the rectangles before transformation
    :type object_img: np.ndarray with shape (height, width, channels)

    :param homography: Projective Transformation matrix for homogeneous coordinates
    :type homography: np.ndarray with shape (3, 3)

    :return: Copied image of scene_img with rectangle drawn on top
    :rtype: np.ndarray with  the same shape (height, width, channels) as scene_img
    """
    output_img = scene_img.copy()

    # Get the height and width of our template object which will define the size of the rectangles we draw
    height, width = object_img.shape[0:2]

    # Define a rectangle with the 4 vertices. With the top left vertex at position [0,0]
    rectangle = np.array([[0, 0],
                          [width, 0],
                          [width, height],
                          [0, height]], dtype=np.float32)

    # Add ones for homogeneous transform
    hom_point = np.c_[rectangle, np.ones(rectangle.shape[0])]

    # Use homography to transform the rectangle accordingly
    rectangle_tf = (homography @ hom_point.T).T
    rectangle_tf = np.around((rectangle_tf[..., 0:2].T/rectangle_tf[..., 2]).T).astype(np.int32)

    cv2.polylines(output_img, [rectangle_tf], isClosed=True, color=(0, 255, 0), thickness=3)

    # Change the top line to be blue, so we can tell the top of the object
    cv2.line(output_img, tuple(rectangle_tf[0]), tuple(rectangle_tf[1]), color=(255, 0, 0), thickness=3)

    return output_img

def debug_homography() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a rectangle and transform it randomly for testing the find_homography function

    :return: (generated_image, target_points, source_points):
        generated_image: image with a single projected rectangle
        target_points: array of keypoints_1 in the target image in the shape (n, 2)
        source_points: array of keypoints_1 in the source image in the shape (n, 2)
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    scene_height, scene_width = 320, 480
    scene_img = np.zeros(shape=(scene_height, scene_width, 3), dtype = np.int32)



    # Get the height and width of our template object which will define the size of the rectangles we draw
    rect_height, rect_width = (60, 100)

    # Define a rectangle with the 4 vertices. With the top left vertex at position [0,0]
    object_points = np.array([[0, 0],
                              [rect_width, 0],
                              [rect_width, rect_height],
                              [0, rect_height]], dtype=np.int32)

    # Move rectangle to the center of the scene_img and deform randomly
    scene_points = object_points + [scene_width / 2. - rect_width / 2., scene_height / 2. - rect_height / 2] \
                   + np.around(10 * np.random.randn(4, 2))
    scene_points = scene_points.astype(np.int32)

    cv2.polylines(scene_img, [scene_points], isClosed=True, color=(255, 255, 255), thickness=10)

    # Change the top line to be blue, so we can tell the top of the object
    cv2.line(scene_img, tuple(scene_points[0]), tuple(scene_points[1]), color=(0, 0, 255), thickness=10)

    return scene_img, scene_points, object_points

def non_max(input_array: np.array) -> np.array:
    """ Return a matrix in which only local maxima of the input mat are set to True, all other values are False

    :param mat: Input matrix
    :type mat: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range (-inf, 1.]

    :return: Binary Matrix with the same dimensions as the input matrix
    :rtype: np.ndarray with shape (height, width) with dtype = bool
    """

    # Initialize a 3x3 kernel with ones and a zero in the middle
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    # Apply the OpenCV dilate morphology transformation.
    # For details see https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    dilation = cv2.dilate(input_array, kernel)
    return input_array > dilation


def filter_matches(matches: Tuple[Tuple[cv2.DMatch]]) -> List[cv2.DMatch]:
    """Filter out all matches that do not satisfy the Lowe Distance Ratio Condition

    :param matches: Holds all the possible matches. Each 'row' are matches of one source_keypoint to target_keypoint
    :type matches: Tuple of tuples of cv2.DMatch https://docs.opencv.org/master/d4/de0/classcv_1_1DMatch.html

    :return filtered_matches: A list of all matches that fulfill the Low Distance Ratio Condition
    :rtype: List[cv2.DMatch]
    """
    ######################################################
    # Write your own code here

    filtered_matches = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.8:
            filtered_matches.append(m[0])

    ######################################################
    return filtered_matches



