import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

mp.rc('xtick', labelsize=5)
mp.rc('ytick', labelsize=5)

class discrate:

    def __init__(self,sessData):
        data = sessData.parsedData
        plt.subplots(3,4)

        ndxChoL = data.ChoiceLeft.values
        ndxChoR = np.logical_not(ndxChoL)
        ndxForc = data.Forced.values
        ndxRwd = data.Rewarded.values

        hrate = plt.subplot(341)
        hrate.set_xlabel('Time (min) from session start', fontsize=7)
        hrate.set_ylabel('Cumulative trial count', fontsize=7)
        plt.plot(np.asarray(data.tsState0.values[ndxChoL]-data.tsState0.values[0])/60,np.arange(np.sum(ndxChoL)), color='xkcd:mango')
        plt.plot(np.asarray(data.tsState0.values[ndxChoR]-data.tsState0.values[0])/60,np.arange(np.sum(ndxChoR)), color='xkcd:darkish green')

        plt.subplot(342)
        plt.plot(data.tsState0.values[np.logical_and(ndxChoL,ndxForc)]-data.tsState0.values[0],np.arange(np.sum(np.logical_and(ndxChoL,ndxForc))))
        plt.plot(data.tsState0.values[np.logical_and(ndxChoL,ndxForc)]-data.tsState0.values[0],np.arange(np.sum(np.logical_and(ndxChoL,ndxForc))))

        hrSt0 = plt.subplot(345)
        hhSt0 = plt.subplot(349)
        aliSt0 = np.zeros(data.tsState0.shape)
        self.raspsth(data,aliSt0,(hrSt0,hhSt0))
        hhSt0.set_xlabel('Time (s) from state0', fontsize=7)
        hhSt0.set_ylabel('Trial count', fontsize=7)

        hrCin = plt.subplot(346)
        hhCin = plt.subplot(3,4,10)
        aliCin = data.tsTrialStart
        self.raspsth(data,aliCin,(hrCin,hhCin))
        hhCin.set_xlabel('Time (s) from Cin', fontsize=7)

        hrRtn = plt.subplot(347)
        hhRtn = plt.subplot(3,4,11)
        aliRtn = data.tsRwdTone
        self.raspsth(data,aliRtn,(hrRtn,hhRtn))
        hhRtn.set_xlabel('Time (s) from RwdTone', fontsize=7)

        hrRwd = plt.subplot(348)
        hhRwd = plt.subplot(3,4,12)
        aliRwd = data.tsRwd
        self.raspsth(data,aliRwd,(hrRwd,hhRwd))
        hhRwd.set_xlabel('Time (s) from Rwd', fontsize=7)

        plt.tight_layout()

        self.fig = plt.gcf()

    def raspsth(self,data,alignment,panes):

        hisPkL = []
        hisPkR = []
        hisPkC = []
        hisTone = []
        hisRwd = []

        haRaster, haHist = panes

        iRow = 0;
        for i in np.argsort(data.ChoiceLeft).values :
            haRaster.eventplot([data.tsPokeL[i]-alignment[i]], lineoffsets=iRow, colors='xkcd:mango')
            haRaster.eventplot([data.tsPokeR[i]-alignment[i]], lineoffsets=iRow, colors='xkcd:darkish green')
            haRaster.eventplot([data.tsPokeC[i]-alignment[i]], lineoffsets=iRow, colors='xkcd:scarlet')
            haRaster.eventplot([data.tsRwdTone[i]-alignment[i]], lineoffsets=iRow, colors='xkcd:charcoal')
            haRaster.eventplot([data.tsRwd[i]-alignment[i]], lineoffsets=iRow, colors='xkcd:water blue')
            iRow += 1

            hisPkL = np.hstack((hisPkL,data.tsPokeL[i]-alignment[i]))
            hisPkR = np.hstack((hisPkR,data.tsPokeR[i]-alignment[i]))
            hisPkC = np.hstack((hisPkC,data.tsPokeC[i]-alignment[i]))
            hisTone = np.hstack((hisTone,data.tsRwdTone[i]-alignment[i]))
            hisRwd = np.hstack((hisRwd,data.tsRwd[i]-alignment[i]))

        haHist.hist(hisPkL, bins='auto', histtype='step', color='xkcd:mango', density=False)
        haHist.hist(hisPkR, bins='auto', histtype='step', color='xkcd:darkish green', density=False)
        haHist.hist(hisPkC, bins='auto', histtype='step', color='xkcd:scarlet', density=False)
        haHist.hist(hisTone, histtype='step', color='xkcd:charcoal', density=False)
        haHist.hist(hisRwd, histtype='step', color='xkcd:water blue', density=False)
