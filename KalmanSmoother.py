import numpy as np

class KalmanSmoother:
    def __init__(self, fMeans, pMeans, fCOVs, pCOVs, dt):
        self.dt=dt
        self.i=len(fMeans)-1

        self.fMeans=fMeans
        self.pMeans=pMeans

        self.fCOVs=fCOVs
        self.pCOVs=pCOVs

        self.fCOV=np.array(self.fCOVs[self.i])
        self.pCOV=np.array(self.pCOVs[self.i])

        self.fMean=np.array(self.fMeans[self.i])
        self.pMean=np.array(self.pMeans[self.i])

        self.x=self.fMean
        self.P=self.fCOV

        self.A=np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.C=np.dot(np.dot(self.P, np.transpose(self.A)), np.linalg.inv(self.pCOV))
        

    def smooth(self):
        self.x=self.fMean+np.dot(self.C,(self.x-self.pMean))
        
        self.P=self.fCOV+np.dot(np.dot(self.C,(self.P-self.pCOV)), np.transpose(self.C))
        
        self.i-=1

        self.fCOV=np.array(self.fCOVs[self.i])
        self.pCOV=np.array(self.pCOVs[self.i])

        self.C=np.dot(np.dot(self.fCOV, np.transpose(self.A)), np.linalg.inv(self.pCOV))
        
        self.fMean=np.array(self.fMeans[self.i])
        self.pMean=np.array(self.pMeans[self.i])

        return self.x[0:2]
        
