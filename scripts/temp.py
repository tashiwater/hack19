#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import cv2

if __name__ == "__main__":
    cap=cv2.VideoCapture(0)  # もともとなかった
    cap.set(3, 320)
    cap.set(4, 240)
    cap.set(5, 5)
    while True:
        _, frame = cap.read()
        cv2.imshow("raw", frame)
        cv2.imwrite("raw.png", frame)
        cv2.waitKey(0)
    cap.release()
