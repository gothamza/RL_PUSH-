import cv2

def RGBToGray(frame):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return grayscale_frame

def Resize(frame, target_size=(84, 84)):
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return resized_frame

def Normalize(frame):
    normalized_frame = frame / 255.0
    return normalized_frame

import numpy as np

def StackFrames(frame, previous_frames=[]):
    if not previous_frames:
        for _ in range(3):
            previous_frames.append(np.zeros_like(frame))

    stacked_frames = np.stack([frame] + previous_frames, axis=2)
    return stacked_frames

    
