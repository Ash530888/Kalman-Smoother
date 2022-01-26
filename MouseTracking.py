from KalmanFilter import KalmanFilter
from KalmanSmoother import KalmanSmoother
import numpy as np
import matplotlib.pyplot as plt
import math

def rmse(x, x2):
    tot=0
    for i in range(len(x)):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))


def plotPositions(measuredx, filteredx, predictedx, smoothedx,measuredy, filteredy, predictedy, smoothedy, total, yAxis):
    #t=np.linspace(0, 1, total)
    fig, ax = plt.subplots()

##    line1,=ax.plot(t,measured,label='measured')
##    line2,=ax.plot(t,filtered,label='filtered')
##    line3,=ax.plot(t,predicted,label='predicted')
##    line4,=ax.plot(t,smoothed,label='smoothed')

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

    plt.xlabel('time')
    plt.ylabel(yAxis)

    plt.show()



def main(u_x=1, u_y=1):
    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    KF = KalmanFilter(0.02, u_x, u_y, 1, 0.1,0.1)

    measured=np.genfromtxt('postions_session003_start0.00_end15548.27.csv',delimiter=',')
    #measured=np.genfromtxt('short.csv',delimiter=',')
    total=measured.shape[0]
    
    predicted=np.empty((total,2))
    filtered=np.empty((total,2))
    
    matrixIndex=0

    pCOVs = []
    fCOVs = []
    pMeans = []
    fMeans = []

    for i in measured:
        
        pMean, pCOV=KF.predict()
        
        if np.isnan(i[1]):
            fMean, fCOV=KF.updateMissing()
            
        else:
            
            fMean, fCOV=KF.update(i[1:3].reshape(2,1))
        

        predicted[matrixIndex,0]=pMean[0].item(0)
        predicted[matrixIndex,1]=pMean[1].item(0)

        filtered[matrixIndex,0]=fMean[0].item(0)
        filtered[matrixIndex,1]=fMean[1].item(0)

        pCOV=np.squeeze(np.asarray(pCOV))
        fCOV=np.squeeze(np.asarray(fCOV))
        
        pCOVs.append(list(pCOV))
        fCOVs.append(list(fCOV))

        pMean=np.squeeze(np.asarray(pMean))
        fMean=np.squeeze(np.asarray(fMean))

        pMeans.append(list(pMean))
        fMeans.append(list(fMean))

        matrixIndex+=1

    #Create KalmanSmoother object KS
    #KalmanSmoother(fMeans, pMeans, fCOV, pCOV, dt)

        
    KS=KalmanSmoother(fMeans, pMeans, fCOVs, pCOVs, 0.02)

    smoothed=np.empty((total,2))

    matrixIndex=0
    
    for i in range(total):
        x,y=KS.smooth()

        smoothed[matrixIndex,0]=x.item(0)
        smoothed[matrixIndex,1]=y.item(0)

        matrixIndex+=1
    

##    plotPositions(measured[:,2], filtered[:,1], predicted[:,1], smoothed[:,1], total, "y positions")
##    plotPositions(measured[:,1], filtered[:,0], predicted[:,0], smoothed[:,0], total, "x positions")

    plotPositions(measured[:,1], filtered[:,0], predicted[:,0], smoothed[:,0], measured[:,2], filtered[:,1], predicted[:,1], smoothed[:,1], total, "x positions")

    
if __name__ == "__main__":
    main()
    

        
            
            
    
