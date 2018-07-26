import os

import numpy as np
import pandas as pd
import scipy.io as sio

# import matplotlib as mp
# import matplotlib.pyplot as plt

class parseSess:

    def __init__(self,filepath):

        if not os.path.exists(filepath):
            print('File not found: ' + bfilepath)
            raise OSError(filepath)
        mysess = sio.loadmat(filepath, squeeze_me=True)
        self.fname = os.path.split(filepath)[1]
        self.bpod = mysess['SessionData']

        self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        self.parse()

    def parse(self):
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsState0 = self.bpod['TrialStartTimestamp'].item()
        ChoiceLeft = np.full(len(nTrials),False)
        ChoiceRight = np.full(len(nTrials),False)
        ChoiceMiss = np.full(len(nTrials),False)
        Rewarded = np.full(len(nTrials),False)
        tsCin = np.full(len(nTrials),np.nan)
        tsCout = np.full(len(nTrials),np.nan)
        tsChoice = np.full(len(nTrials),np.nan)
        tsRwd = np.full(len(nTrials),np.nan)
        tsErrTone = np.full(len(nTrials),np.nan)
        tsPokeL = [[]]*len(nTrials)
        tsPokeC = [[]]*len(nTrials)
        tsPokeR = [[]]*len(nTrials)
        stateTraj = [[]]*len(nTrials)
        FixBroke = np.full(len(nTrials),False)
        EarlyWithdrawal = np.full(len(nTrials),False)
        waitingTime = np.full(len(nTrials),np.nan)

        tsState0 = tsState0 - tsState0[0]

        """
        Feedback = np.full(len(nTrials),False)#<---
        FixDur = np.full(len(nTrials),np.nan)#<---
        MT = np.full(len(nTrials),np.nan)#<---
        ST = np.full(len(nTrials),np.nan)#<---
        """
        for iTrial in range(len(nTrials)) :
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
            stateTraj[iTrial] = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1] #from 1- to 0-based indexing

            tsCin[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['wait_Cin'].item()[1]

            PortL = 'Port' + str(int(self.params.Ports_LMR))[0] + 'In'
            if any([PortL in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeL[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortL].item()

            PortC = 'Port' + str(int(self.params.Ports_LMR))[1] + 'In'
            if any([PortC in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeC[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortC].item()

            PortR = 'Port' + str(int(self.params.Ports_LMR))[2] + 'In'
            if any([PortR in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeR[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortR].item()

            ChoiceLeft[iTrial] = any(['Lin' in stateTraj[iTrial]])
            if ChoiceLeft[iTrial]:
                if any(['start_Lin' in stateTraj[iTrial]]):
                    tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['start_Lin'].item()[0]
                else:
                    tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item()[0]

            ChoiceRight[iTrial] = any(['Rin' in stateTraj[iTrial]])
            if ChoiceRight[iTrial]:
                if any(['start_Rin' in stateTraj[iTrial]]):
                    tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['start_Rin'].item()[0]
                else:
                    tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item()[0]

            FixBroke[iTrial] = any(['EarlyCout' in stateTraj[iTrial]])

            EarlyWithdrawal[iTrial] = any([any(['EarlyRout' in stateTraj[iTrial]]),any(['EarlyLout' in stateTraj[iTrial]])])

            ChoiceMiss[iTrial] = any(['wait_Sin'in stateTraj[iTrial]]) and not(any([ChoiceLeft[iTrial],ChoiceRight[iTrial]]))

            if any(['wait_Sin' in stateTraj[iTrial]]):
                ndx = [next((j for j, x in enumerate([n.startswith('stimulus_delivery') for n in stateTraj[iTrial]]) if x), None)]
                tsCout[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['wait_Sin'].item()[0]

            Rewarded[iTrial] = any([n.startswith('water_') for n in stateTraj[iTrial]])
            if Rewarded[iTrial] :
                tsRwd[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[iTrial][[n.startswith('water_') for n in stateTraj[iTrial]]].item()].item()[0]
                waitingTime[iTrial] = tsChoice[iTrial]-tsRwd[iTrial]
            else:
                ndx = np.array([n.startswith('Early') for n in stateTraj[iTrial]])
                ndx = np.logical_or(ndx,np.array([n.startswith('unrewarded') for n in stateTraj[iTrial]]))
                #
                print(stateTraj[iTrial])
                if any(ndx):
                    mystate = np.array(stateTraj[iTrial])[ndx].item()
                    waitingTime[iTrial] = tsChoice[iTrial]-self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[mystate].item()[0]
                # if ChoiceLeft[iTrial]:
                #     waitingTime[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item())
                # if ChoiceRight[iTrial]:
                #     waitingTime[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item())

            # if any([n.startswith('start_') for n in stateTraj[iTrial]]):
            #     if 'start_Lin' in stateTraj[iTrial]:
            #         pass
            #     if 'start_Rin' in stateTraj[iTrial]:
            #         pass

            if any(['stillRin' in stateTraj[iTrial]]):
                # print(stateTraj[iTrial])
                waitingTime[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillRin'].item()[1] - \
                self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item()[0]
            if any(['stillLin' in stateTraj[iTrial]]):
                # print(stateTraj[iTrial])
                waitingTime[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillLin'].item()[1] - \
                self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item()[0]
                # print(stateTraj[iTrial]) #[[n.startswith('still') for n in stateTraj[iTrial]]])
                # if ChoiceLeft[iTrial]:
                #     waitingTime[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillLin'].item()[1] - \
                #     self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillLin'].item()[0]
                # if ChoiceRight[iTrial]:
                #     waitingTime[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item())



        # assert all(np.logical_xor(ChoiceRight,ChoiceLeft))

        self.parsedData = pd.DataFrame({'iTrial': nTrials, 'isChoiceLeft': ChoiceLeft, 'isChoiceRight': ChoiceRight, 'isChoiceMiss': ChoiceMiss,
                                        'isRewarded': Rewarded, 'isBrokeFix': FixBroke, 'isEarlyWithdr': EarlyWithdrawal,
                                        'tsCin': tsCin, 'tsChoice': tsChoice, 'tsRwd': tsRwd, 'stateTraj': stateTraj, 'WT': waitingTime,
                                        'tsPokeL': tsPokeL, 'tsPokeC': tsPokeC, 'tsPokeR': tsPokeR, 'tsState0': tsState0})

        self.parsedData = self.parsedData.set_index('iTrial')

class dailyfig:

    def __init__(self,sessData):
        """
        mp.rc('xtick', labelsize=5)
        mp.rc('ytick', labelsize=5)

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
        aliCin = data.tsCin
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
        """

class multisess:

    def __init__(self):
        self.summary = pd.DataFrame({'OdorFracA': [], 'ChoFracL': []})

    def append(self,parserOutput):
        """
        x = self.parsedData['OdorFracA']
        for i in set(x) :
            ndxOdor = i == parsed.parsedData['OdorFracA']
            ndxMiss = np.logical_not(parsed.parsedData['ChoiceMiss'])
            ndxFilt = np.logical_and(ndxOdor,ndxMiss)
            print(i)
            print(np.sum(parsed.parsedData['ChoiceLeft'].values[ndxFilt])/np.sum(ndxFilt))
            plt.scatter(i,np.sum(parsed.parsedData['ChoiceLeft'].values[ndxFilt])/np.sum(ndxFilt))


        ndxFree = np.logical_not(parserOutput.parsedData.Forced)
        ndxChoL = parserOutput.parsedData.ChoiceLeft
        preL = parserOutput.parsedData.delayPre[np.logical_and(ndxChoL,ndxFree)].median()
        posL = parserOutput.parsedData.delayPost[np.logical_and(ndxChoL,ndxFree)].median()
        ndxChoR = np.logical_not(ndxChoL)
        preR = parserOutput.parsedData.delayPre[np.logical_and(ndxChoR,ndxFree)].median()
        posR = parserOutput.parsedData.delayPost[np.logical_and(ndxChoR,ndxFree)].median()
        pLeft = np.mean(parserOutput.parsedData.ChoiceLeft[ndxFree].values)
        logOdds = np.log(pLeft/(1-pLeft))
        self.summary = self.summary.append({'preL': preL, 'preR': preR, 'posL': posL, 'posR': posR, 'pLeft': pLeft, 'logOdds': logOdds}, ignore_index=True)
        """
