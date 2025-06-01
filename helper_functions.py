#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import Tuple, List

def show_image(img: np.ndarray, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:


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




def non_max(input_array: np.array) -> np.array:

    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    kernel[1, 1] = 0


    dilation = cv2.dilate(input_array, kernel)
    return input_array > dilation


def filter_matches(matches: Tuple[Tuple[cv2.DMatch]]) -> List[cv2.DMatch]:


    filtered_matches = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.8:
            filtered_matches.append(m[0])

    ######################################################
    return filtered_matches



