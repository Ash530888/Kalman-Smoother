from KalmanFilter import KalmanFilter
from KalmanSmoother import KalmanSmoother
import numpy as np
import matplotlib.pyplot as plt
import math

import cv2
from get_background import get_background


def main():
    cap = cv2.VideoCapture("mouse.avi")
    # get the video frame height and width
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    measured=np.empty((total,2))
    predicted=np.empty((total,2))
    filtered=np.empty((total,2))
    
    matrixIndex=0

    # get the background model
    background = get_background("mouse.avi")
    # convert the background model to grayscale format
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    consecutive_frame = 8

    #Create KalmanFilter object KF
    #KalmanFilter(dt, std_acc, x_std_meas, y_std_meas)
    KF = KalmanFilter(0.02, 0.1, 0.1,0.1)


    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_count += 1
            orig_frame = frame.copy()
            # IMPORTANT STEP: convert the frame to grayscale first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_count % consecutive_frame == 0 or frame_count == 1:
                frame_diff_list = []
            # find the difference between current frame and base frame
            frame_diff = cv2.absdiff(gray, background)
            # thresholding to convert the frame to binary
            ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
            # dilate the frame a bit to get some more white area...
            # ... makes the detection of contours a bit easier
            dilate_frame = cv2.dilate(thres, None, iterations=2)
            # append the final result into the `frame_diff_list`
            frame_diff_list.append(dilate_frame)
            # if we have reached `consecutive_frame` number of frames
            
            if len(frame_diff_list) == consecutive_frame:
                # add all the frames in the `frame_diff_list`
                sum_frames = sum(frame_diff_list)
                # find the contours around the white segmented areas
                contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # draw the contours, not strictly necessary
                for i, cnt in enumerate(contours):
                    cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
                if len(contours)>0:
                    # continue through the loop if contour area is less than 500...
                    # ... helps in removing noise detection
                    if cv2.contourArea(contours[0]) < 500:
                        continue
                    # get the xmin, ymin, width, and height coordinates from the contours
                    (x, y, w, h) = cv2.boundingRect(contours[0])
                    cv2.putText(orig_frame, "Measured Position", (int(x + 15), int(y - 15)), 0, 0.5, (0, 255, 0), 2)

                    # draw the bounding boxes
                    cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    (x1, y1) = KF.predict()
                    cv2.rectangle(orig_frame, (int(x1 - w), int(y1 - h)),(int( x1 + w), int(y1 + h)), (255, 0, 0), 2)
                    cv2.putText(orig_frame, "Predicted Position", (int(x1 + w), int(y1)), 0, 0.5, (255, 0, 0), 2)

                    (x, y) = KF.update(y)
                    cv2.rectangle(orig_frame, (int(x - w), int(y - h)), (int(x + w), int(y + h)), (0, 0, 255), 2)

                    cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)

                    
                else:
                    (x, y) = KF.predict()
                    cv2.rectangle(orig_frame, (int(x - w), int(y - h)),(int( x + w), int(y + h)), (255, 0, 0), 2)

                    cv2.putText(orig_frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)

                    (x, y) = KF.updateMissing()
                    cv2.rectangle(orig_frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 0, 255), 2)

                    cv2.imshow('Detected Objects', orig_frame)

                    cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)
                    
                cv2.imshow('Detected Objects', orig_frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

main()
