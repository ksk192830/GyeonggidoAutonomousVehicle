#!/usr/bin/env python3

import os
import cv2

IMAGE_DIR = "/home/sg/gyeonggi_ws/src/camera_pkg/camera_pkg/lib/parking_rear/"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def rotate_images_90deg(directory: str):
    for filename in os.listdir(directory):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Failed to load image: {filename}")
            continue

        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite(img_path, rotated)
        print(f"[OK] Rotated: {filename}")


if __name__ == "__main__":
    rotate_images_90deg(IMAGE_DIR)
