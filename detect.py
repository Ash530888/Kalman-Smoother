from KalmanFilter import KalmanFilter
from KalmanSmoother import KalmanSmoother
import numpy as np
import matplotlib.pyplot as plt
import math

import cv2
from get_background import get_background

import sys

sys.path.append("Joaquin/scripts")

import filter_smooth


def rmse(x, x2):
    tot=0
    for i in range(len(x)):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))

def plotPositions(measuredx, filteredx, predictedx, smoothedx,measuredy, filteredy, predictedy, smoothedy, total):
    
    fig, ax = plt.subplots()

    line1,=ax.plot(measuredx,measuredy,label='measured')
    line2,=ax.plot(filteredx,filteredy,label='filtered')
    line3,=ax.plot(predictedx,predictedy,label='predicted')
    line4,=ax.plot(smoothedx,smoothedy,label='smoothed')

    filteredx = filteredx[np.logical_not(np.isnan(measuredx))]

    smoothedx = smoothedx[np.logical_not(np.isnan(measuredx))]
    
    measuredx = measuredx[np.logical_not(np.isnan(measuredx))]
    
    print("rmse measuredx vs filteredx: ",rmse(measuredx, filteredx))
    print("rmse measuredx vs smoothedx: ",rmse(measuredx, smoothedx))

    lines=[line1, line2, line3, line4]

    leg=ax.legend()
    graphs = {}

    lineLegends=leg.get_lines()
    
    for i in range(len(lines)):
        lineLegends[i].set_picker(True)
        lineLegends[i].set_pickradius(5)
        graphs[lineLegends[i]]=lines[i]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()

        graphs[legend].set_visible(not isVisible)
        legend.set_visible(not isVisible)

        fig.canvas.draw()

    plt.connect('pick_event', on_pick)

    
    plt.xlim([0, 1500])
    plt.ylim([0, 1100])

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def compareOutputs(measx, f1x, f2x, measy, f1y, f2y):

    fig, ax = plt.subplots()

    line1,=ax.plot(measx, measy, label="measured")
    line2,=ax.plot(f1x, f1y, label="my filtered")
    line3,=ax.plot(f2x, f2y, label="Joaquin's filtered")


    lines=[line1, line2, line3]

    leg=ax.legend()
    graphs = {}

    lineLegends=leg.get_lines()
    
    for i in range(len(lines)):
        lineLegends[i].set_picker(True)
        lineLegends[i].set_pickradius(5)
        graphs[lineLegends[i]]=lines[i]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()

        graphs[legend].set_visible(not isVisible)
        legend.set_visible(not isVisible)

        fig.canvas.draw()

    plt.connect('pick_event', on_pick)


    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()



##def plotPositions(measuredx, filteredx, predictedx ,measuredy, filteredy, predictedy, total):
##    
##    fig, ax = plt.subplots()
##
##    line1,=ax.plot(measuredx,measuredy,label='measured')
##    line2,=ax.plot(filteredx,filteredy,label='filtered')
##    line3,=ax.plot(predictedx,predictedy,label='predicted')
##
##    filteredx = filteredx[np.logical_not(np.isnan(measuredx))]
##    
##    measuredx = measuredx[np.logical_not(np.isnan(measuredx))]
##    
##    print("rmse measuredx vs filteredx: ",rmse(measuredx, filteredx))
##
##    lines=[line1, line2, line3]
##
##    leg=ax.legend()
##    graphs = {}
##
##    lineLegends=leg.get_lines()
##    
##    for i in range(len(lines)):
##        lineLegends[i].set_picker(True)
##        lineLegends[i].set_pickradius(5)
##        graphs[lineLegends[i]]=lines[i]
##
##    def on_pick(event):
##        legend = event.artist
##        isVisible = legend.get_visible()
##
##        graphs[legend].set_visible(not isVisible)
##        legend.set_visible(not isVisible)
##
##        fig.canvas.draw()
##
##    plt.connect('pick_event', on_pick)
##
##    plt.xlim([0, 1500])
##    plt.ylim([0, 1100])
##
##    plt.xlabel('x')
##    plt.ylabel('y')
##
##    plt.show()


def main(readFromCSV=False):
    dt=0.019936
    x_std_meas=y_std_meas=0.001
    std_acc= 0.1
    
    A = np.matrix([[1, dt, (dt**2)/2],
                            [0, 1, dt],
                            [0, 0, 1]])
    A = np.block([[A, np.zeros((3,3))],[np.zeros((3,3)), A]])

    H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0]])

    Q = np.matrix([[(dt**4)/4, (dt**3)/2, (dt**2)/2],
                                [(dt**3)/2, dt**2, dt],
                                [(dt**2)/2, dt, 1]]) * std_acc**2
    Q = np.block([[Q, np.zeros((3,3))],[np.zeros((3,3)), Q]])

    R = np.diag([x_std_meas**2, y_std_meas**2])
    
        
    if readFromCSV:
        measured=np.genfromtxt('data/postions_session003_start0.00_end15548.27.csv',delimiter=',')
        #measured=np.genfromtxt('short.csv',delimiter=',')
        #total=measured.shape[0]-2
        total=10000

        measured=measured[1:total+1]
        
        y=(measured[1][1:3]).reshape(2,1)
        m0=np.matrix([[y[0, 0]], [0], [0], [y[1, 0]], [0], [0]])

        v0 = np.diag(np.ones(len(m0))*0.001)

        #Create KalmanFilter object KF
        #KalmanFilter(dt, std_acc, x_std_meas, y_std_meas, A, H, Q, R, m0, v0)
        KF = KalmanFilter(dt,  std_acc, x_std_meas, y_std_meas, A, H, Q, R, m0, v0)
        #KF.update(measured[1][1:3].reshape(2,1))
        
        predicted=np.empty((total,2))
        filtered=np.empty((total,2))
        
        matrixIndex=0

        pCOVs = []
        fCOVs = []
        pMeans = []
        fMeans = []

        for i in range(total):
            pMean, pCOV=KF.predict()

            
            if np.isnan(measured[i][1]):
                fMean, fCOV=KF.updateMissing()
                    
            else:
                fMean, fCOV=KF.update(measured[i][1:3].reshape(2,1))

            
            
            predicted[matrixIndex,0]=pMean[0]
            predicted[matrixIndex,1]=pMean[3]

            filtered[matrixIndex,0]=fMean[0]
            filtered[matrixIndex,1]=fMean[3]

            pCOV=np.squeeze(np.asarray(pCOV))
            fCOV=np.squeeze(np.asarray(fCOV))
                
            pCOVs.append(list(pCOV))
            fCOVs.append(list(fCOV))

            pMean=np.squeeze(np.asarray(pMean))
            fMean=np.squeeze(np.asarray(fMean))

            pMeans.append(list(pMean))
            fMeans.append(list(fMean))

            matrixIndex+=1

        filtered2=filter_smooth.main([])

        compareOutputs(measured[:,1], filtered[:,0], filtered2[0,0,:], measured[:,2], filtered[:,1], filtered2[3,0,:])
    
        #plotPositions(measured[:,1], filtered[:,0], predicted[:,0], measured[:,2], filtered[:,1], predicted[:,1], total)
        
        KS=KalmanSmoother(fMeans, pMeans, fCOVs, pCOVs, dt)

        smoothed=np.empty((total,2))

        matrixIndex=total-1
        
        for i in range(total):
            x =KS.smooth()

            smoothed[matrixIndex,0]=x[0].item(0)
            smoothed[matrixIndex,1]=x[3].item(0)
            
            matrixIndex-=1

        
        plotPositions(measured[:,1], filtered[:,0], predicted[:,0], smoothed[:,0], measured[:,2], filtered[:,1], predicted[:,1], smoothed[:,1], total)
        
        
    else:
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
                        cAreas=[]
                        
                        for i in range(len(contours)):
                            cAreas.append(cv2.contourArea(contours[i]))
                        
                        i=cAreas.index(max(cAreas))
                        
                        if cv2.contourArea(contours[i]) < 250:
                            sp, _ = KF.predict()

                            x1 = sp[0]
                            y1 = sp[3]
                            cv2.rectangle(orig_frame, (int(x1), int(y1)),(int( x1 + w), int(y1 + h)), (255, 0, 0), 2)
                            cv2.putText(orig_frame, "Predicted Position", (int(x1 + w), int(y1)), 0, 0.5, (255, 0, 0), 2)
                            
                            h=36
                            w=31
                            su, _ = KF.updateMissing()
                            x = su[0]
                            y = su[3]
                            cv2.rectangle(orig_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                            cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)
                            cv2.imshow('Detected Objects', orig_frame)
                            continue
                        
                        # get the xmin, ymin, width, and height coordinates from the contours
                        (x, y, w, h) = cv2.boundingRect(contours[i])

                        
                        cv2.putText(orig_frame, "Measured Position", (int(x + 15), int(y - 15)), 0, 0.5, (0, 255, 0), 2)

                        # draw the bounding boxes
                        cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        try:
                            sp, _ = KF.predict()
                        except UnboundLocalError:
                            m0=np.matrix([[x], [0], [0], [y], [0], [0]])

                            v0 = np.diag(np.ones(m0.shape[0])*0.001)

                            #Create KalmanFilter object KF
                            #KalmanFilter(dt, std_acc, x_std_meas, y_std_meas, A, H, Q, R, m0, v0)
                            KF = KalmanFilter(dt,  std_acc, x_std_meas, y_std_meas, A, H, Q, R, m0, v0)
                            sp, _ = KF.predict()
                            
                        x1 = sp[0]
                        y1 = sp[3]
                        cv2.rectangle(orig_frame, (int(x1), int(y1)),(int( x1 + w), int(y1 + h)), (255, 0, 0), 2)
                        cv2.putText(orig_frame, "Predicted Position", (int(x1 + w), int(y1)), 0, 0.5, (255, 0, 0), 2)

                        z = np.array([[x, y]]).T
                        su, _ = KF.update(z=z)
                        x = su[0]
                        y = su[3]
                        cv2.rectangle(orig_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

                        cv2.putText(orig_frame, "Filtered Position", (int(x + 30), int(y +25)), 0, 0.5, (0, 0, 255), 2)

                        
                    else:
                        sp, _ = KF.predict()
                        x = sp[0]
                        y = sp[3]
                        cv2.rectangle(orig_frame, (int(x), int(y)),(int( x + w), int(y + h)), (255, 0, 0), 2)

                        cv2.putText(orig_frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)

                        su, _ = KF.updateMissing()
                        x = su[0]
                        y = su[3]
                        cv2.rectangle(orig_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

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
