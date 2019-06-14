import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

class parseSess:

    def __init__(self,filepath):

        if not os.path.exists(filepath):
            print('File not found: ' + filepath)
            raise OSError(filepath)
        mysess = sio.loadmat(filepath, squeeze_me=True)
        self.fname = os.path.split(filepath)[1]
        self.bpod = mysess['SessionData']
        try:
            self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        except Exception as e:
            print(e)
            self.params = pd.Series()
            warnings.warn('Could not load settings.')
        self.parse()

    def parse(self):

        # self.parsedData = pd.DataFrame({n:self.bpod['Custom'].item()[n].item() for n in ['BaitedL', 'BrokeFix', 'CenterPokeDur', 'ChoiceLeft','EarlySout', 'Grace', 'RewardMagnitude', 'Rewarded','SidePokeDur', 'StimGuided']}).iloc[:-1,:]

        nTrials = np.asscalar(self.bpod['nTrials'])
        tsState0 = self.bpod['TrialStartTimestamp'].item()

        ChoiceLeft = np.full(nTrials,False)
        ChoiceRight = np.full(nTrials,False)
        ChoiceMiss = np.full(nTrials,False)
        Rewarded = np.full(nTrials,False)
        BrokeFix = np.full(nTrials,False)
        EarlySout = np.full(nTrials,False)
        CheckedOther = np.full(nTrials,False)
        BaitedL = np.full(nTrials,False)
        StimGuided = np.full(nTrials,False)
        Grace = np.full(nTrials,False)

        CenterPokeDur = np.full(nTrials,np.nan)
        SidePokeDur = np.full(nTrials,np.nan)

        tsChoice = np.full(nTrials,np.nan)
        tsPokeL = [[]]*nTrials
        tsPokeC = [[]]*nTrials
        tsPokeR = [[]]*nTrials
        assert(not np.isscalar(tsState0)), "Session is only 1 trial long. Aborting."
        assert(len(tsState0) > 20), "Session is only {} trials long. Aborting.".format(len(tsState0))
        tsState0 = tsState0 - tsState0[0]

        PortL = 'Port' + str(int(self.params.Ports_LMR))[0] + 'In'
        PortC = 'Port' + str(int(self.params.Ports_LMR))[1] + 'In'
        PortR = 'Port' + str(int(self.params.Ports_LMR))[2] + 'In'
        stateTraj = [[]]*nTrials

        for iTrial in range(nTrials):
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
            stateTraj[iTrial] = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1] #from 1- to 0-based indexing

            if any([PortL in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeL[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortL].item()

            if any([PortC in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeC[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortC].item()

            if any([PortR in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeR[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortR].item()

            ChoiceMiss[iTrial] = any(['choice_miss' in stateTraj[iTrial]])
            Rewarded[iTrial] = any([n.startswith('water_') for n in stateTraj[iTrial]])
            BrokeFix[iTrial] = any(['BrokeFix' in stateTraj[iTrial]])
            EarlySout[iTrial] = any(['EarlySout' in stateTraj[iTrial]])
            StimGuided[iTrial] = any(['Cin_late' in stateTraj[iTrial]])
            Grace[iTrial] = any(['grace' in n for n in stateTraj[iTrial]])

            ndx_start_S = np.array([n.startswith('start_') for n in stateTraj[iTrial]])
            if ndx_start_S.any():
                start_S = stateTraj[iTrial][ndx_start_S].item()
                tsChoice[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[start_S].item()[0]

                ChoiceLeft[iTrial] = start_S.startswith('start_L')
                if ChoiceLeft[iTrial]:
                    CheckedOther[iTrial] = (tsPokeR[iTrial] > tsChoice[iTrial]).any()

                ChoiceRight[iTrial] = start_S.startswith('start_R')
                if ChoiceRight[iTrial]:
                    CheckedOther[iTrial] = (tsPokeL[iTrial] > tsChoice[iTrial]).any()

                if EarlySout[iTrial]:
                    grace_state = np.unique(stateTraj[iTrial][['grace' in n for n in stateTraj[iTrial]]]).item()
                    SidePokeDur[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[grace_state].item().ravel()[-2] - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[start_S].item()[0]
                else:
                    SidePokeDur[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['ITI'].item()[1] - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[start_S].item()[0]
                    ChoicePortOut = PortL if ChoiceLeft[iTrial] else PortR
                    ChoicePortOut = ChoicePortOut[:-2] + 'Out'
                    if ChoicePortOut in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names:
                        candidates = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[ChoicePortOut].item()
                        candidates = candidates if np.isscalar(candidates) else candidates.astype(float)
                        thresh = tsChoice[iTrial]
                        if Grace[iTrial]:
                            grace_state = np.unique(stateTraj[iTrial][['grace' in n for n in stateTraj[iTrial]]]).item()
                            thresh = max(thresh,self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[grace_state].item().max())
                        if (candidates > thresh).any():
                            if not np.isscalar(candidates):
                                candidates = min(candidates[candidates > thresh])
                            SidePokeDur[iTrial] = candidates - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[start_S].item()[0]

                if StimGuided[iTrial]:
                    CenterPokeDur[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin_late'].item()[1] - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin_early'].item()[0]
                else:
                    CenterPokeDur[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin_early'].item()[1] - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin_early'].item()[0]

        BaitedL = self.bpod['Custom'].item()['BaitedL'].item().astype(bool)
        isChoiceBaited = np.logical_or(np.logical_and(ChoiceLeft,BaitedL),
                                       np.logical_and(ChoiceRight,~BaitedL),
                                      )

        assert(isChoiceBaited[Rewarded].all()), "Impossible trials found: unbaited AND rewarded."

        self.parsedData = pd.DataFrame({'iTrial': np.arange(nTrials),
                                        'isChoiceLeft': ChoiceLeft, 'isChoiceRight': ChoiceRight, 'isBaitedLeft':BaitedL,'isStimGuided':StimGuided,'isGraceVisited':Grace,
                                        'isChoiceMiss': ChoiceMiss,'isRewarded': Rewarded,'isBrokeFix': BrokeFix, 'isEarlySout': EarlySout, 'isChoiceBaited':isChoiceBaited,'isCheckedOther':CheckedOther,
                                        'stateTraj': stateTraj,'tsState0': tsState0, 'tsChoice': tsChoice, 'tsPokeL': tsPokeL, 'tsPokeC': tsPokeC, 'tsPokeR': tsPokeR,'CenterPokeDur':CenterPokeDur,'SidePokeDur':SidePokeDur})

        self.parsedData = self.parsedData.set_index('iTrial')
