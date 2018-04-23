import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

class parser:

    def __init__(self,struct):
        self.bpod = struct['SessionData']
        self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        self.parse()

    def parse(self):
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsState0 = self.bpod['TrialStartTimestamp'].item()
        OdorFracA = self.bpod['Custom'].item()['OdorFracA'].item()
        ChoiceLeft = np.full(len(nTrials),False)
        ChoiceRight = np.full(len(nTrials),False)
        ChoiceMiss = np.full(len(nTrials),False)
        Rewarded = np.full(len(nTrials),False)
        tsCin = np.full(len(nTrials),np.nan)
        tsStimOn = np.full(len(nTrials),np.nan)
        tsCout = np.full(len(nTrials),np.nan)#<---
        tsChoice = np.full(len(nTrials),np.nan)#<---
        tsRwd = np.full(len(nTrials),np.nan)
        tsPokeL = [[]]*len(nTrials)
        tsPokeC = [[]]*len(nTrials)
        tsPokeR = [[]]*len(nTrials)
        FixBroke = np.full(len(nTrials),False)
        EarlyWithdrawal = np.full(len(nTrials),False)

        """

        tsStimOff = np.full(len(nTrials),np.nan)#<---

        ChoiceCorrect = np.full(len(nTrials),False)#<---
        Feedback = np.full(len(nTrials),False)#<---


        FeedbackTime = np.full(len(nTrials),np.nan)#<---
        FixDur = np.full(len(nTrials),np.nan)#<---
        MT = np.full(len(nTrials),np.nan)#<---
        ST = np.full(len(nTrials),np.nan)#<---
        """
        for iTrial in range(len(nTrials)) :
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]

            try:
                stateTraj = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1] #from 1- to 0-based indexing
            except Exception as e:
                print(iTrial)
                print(type(iTrial))
                print([self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]])

            Rewarded[iTrial] = any([n.startswith('water_') for n in stateTraj])
            if Rewarded[iTrial] :
                tsRwd[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[[n.startswith('water_') for n in stateTraj]].item()].item()[0]

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

            ChoiceLeft[iTrial] = any(['start_Lin' in stateTraj])
            if ChoiceLeft[iTrial]:
                tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['start_Lin'].item()[0]

            ChoiceRight[iTrial] = any(['start_Rin' in stateTraj])
            if ChoiceRight[iTrial]:
                tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['start_Rin'].item()[0]

            FixBroke[iTrial] = any(['broke_fixation' in stateTraj])

            EarlyWithdrawal[iTrial] = any(['early_withdrawal' in stateTraj])

            ChoiceMiss[iTrial] = any(['missed_choice' in stateTraj])

            if any([n.startswith('stimulus_delivery') for n in stateTraj]):
                ndx = [next((j for j, x in enumerate([n.startswith('stimulus_delivery') for n in stateTraj]) if x), None)]
                tsStimOn[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[ndx].item()].item()[0]

            if any(['wait_Sin' in stateTraj]):
                ndx = [next((j for j, x in enumerate([n.startswith('stimulus_delivery') for n in stateTraj]) if x), None)]
                tsCout[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['wait_Sin'].item()[0]



            """
            Forced[iTrial] = any([n.startswith('forc_') for n in stateTraj])
            tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[[n.startswith('Pre') for n in stateTraj]].item()].item()[0]
            ndxTone = np.logical_or(np.logical_or([n == 'Wait_Lin' for n in stateTraj],[n == 'Wait_Rin' for n in stateTraj]),[n.startswith('rewcue_') for n in stateTraj])
            tsRwdTone[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[ndxTone].item()].item()[0]
            tsRwd[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[[n.startswith('rewarded_') for n in stateTraj]].item()].item()[0]
            delayPost[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stateTraj[-1]].item())[0]
        delayPre = tsRwdTone - tsChoice

        assert all(np.logical_xor(ChoiceRight,ChoiceLeft))
        """
        self.parsedData = pd.DataFrame({'nTrials': nTrials, 'ChoiceLeft': ChoiceLeft, 'ChoiceRight': ChoiceRight, 'ChoiceMiss': ChoiceMiss,
                                        'Rewarded': Rewarded, 'OdorFracA': OdorFracA[0:len(nTrials)], 'FixBroke': FixBroke, 'EarlyWithdrawal': EarlyWithdrawal,
                                        'tsCin': tsCin, 'tsStimOn': tsStimOn, 'tsStimOff': tsCout, 'tsChoice': tsChoice, 'tsRwd': tsRwd,
                                        'tsPokeL': tsPokeL, 'tsPokeC': tsPokeC, 'tsPokeR': tsPokeR, 'tsState0': tsState0})

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
