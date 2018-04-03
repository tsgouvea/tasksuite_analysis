import numpy as np
import pandas as pd
"""Data of a single session of task in ['discrate','flux','matching']. Compatible with analsysis methods in related module ___."""

class discrate:

    def __init__(self,struct):
        self.bpod = struct['SessionData']
        self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        self.parse()

    def parse(self):
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsState0 = self.bpod['TrialStartTimestamp'].item()
        ChoiceLeft = np.full(len(nTrials),False)
        ChoiceRight = np.full(len(nTrials),False)
        Rewarded = np.full(len(nTrials),False)
        Forced = np.full(len(nTrials),False)
        tsTrialStart = np.full(len(nTrials),np.nan)
        tsChoice = np.full(len(nTrials),np.nan)
        tsRwdTone = np.full(len(nTrials),np.nan)
        tsRwd = np.full(len(nTrials),np.nan)
        tsPokeL = [[]]*len(nTrials)
        tsPokeC = [[]]*len(nTrials)
        tsPokeR = [[]]*len(nTrials)

        for i in range(len(nTrials)) :
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[i]
            stateTraj = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[i]-1]
            ChoiceLeft[i] = any(['PreL' in stateTraj])
            ChoiceRight[i] = any(['PreR' in stateTraj])
            Rewarded[i] = any([n.startswith('water_') for n in stateTraj])
            Forced[i] = any([n.startswith('forc_') for n in stateTraj])
            tsTrialStart[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['States'].item()['wait_Cin'].item()[1]
            tsChoice[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['States'].item()[stateTraj[[n.startswith('Pre') for n in stateTraj]].item()].item()[0]
            ndxTone = np.logical_or(np.logical_or([n == 'Wait_Lin' for n in stateTraj],[n == 'Wait_Rin' for n in stateTraj]),[n.startswith('rewcue_') for n in stateTraj])
            tsRwdTone[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['States'].item()[stateTraj[ndxTone].item()].item()[0]
            tsRwd[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['States'].item()[stateTraj[[n.startswith('rewarded_') for n in stateTraj]].item()].item()[0]
            PortL = 'Port' + str(int(self.params.Ports_LMR))[0] + 'In'
            if any([PortL in self.bpod['RawEvents'].item()['Trial'].item()[i]['Events'].item().dtype.names]) :
                tsPokeL[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['Events'].item()[PortL].item()
            PortC = 'Port' + str(int(self.params.Ports_LMR))[1] + 'In'
            if any([PortC in self.bpod['RawEvents'].item()['Trial'].item()[i]['Events'].item().dtype.names]) :
                tsPokeC[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['Events'].item()[PortC].item()
            PortR = 'Port' + str(int(self.params.Ports_LMR))[2] + 'In'
            if any([PortR in self.bpod['RawEvents'].item()['Trial'].item()[i]['Events'].item().dtype.names]) :
                tsPokeR[i] = self.bpod['RawEvents'].item()['Trial'].item()[i]['Events'].item()[PortR].item()

        assert all(np.logical_xor(ChoiceRight,ChoiceLeft))

        self.parsedData = pd.DataFrame({'nTrials': nTrials, 'ChoiceLeft': ChoiceLeft, 'Rewarded': Rewarded, 'Forced': Forced,
                                   'tsTrialStart': tsTrialStart, 'tsChoice': tsChoice, 'tsRwdTone': tsRwdTone, 'tsRwd': tsRwd,
                                  'tsPokeL': tsPokeL, 'tsPokeC': tsPokeC, 'tsPokeR': tsPokeR, 'tsState0': tsState0})

class flux:

    def __init__(self,struct):
        self.bpod = struct['SessionData']
        self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        self.parse()

    def parse(self):
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsState0 = self.bpod['TrialStartTimestamp'].item()

        #for i in range(len(nTrials)) :

        self.parsedData = pd.DataFrame({'nTrials': nTrials, 'tsState0': tsState0})

class matching:

    def __init__(self,struct):
        self.bpod = struct['SessionData']
        self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        self.parse()

    def parse(self):
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsState0 = self.bpod['TrialStartTimestamp'].item()

        #for i in range(len(nTrials)) :

        self.parsedData = pd.DataFrame({'nTrials': nTrials, 'tsState0': tsState0})
