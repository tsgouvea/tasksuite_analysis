import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import linear_model

# import matplotlib as mp
import matplotlib.pyplot as plt

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
        nTrials = np.asscalar(self.bpod['nTrials'])
        tsState0 = self.bpod['TrialStartTimestamp'].item()
        ChoiceLeft = np.full(nTrials,False)
        ChoiceRight = np.full(nTrials,False)
        ChoiceMiss = np.full(nTrials,False)
        Rewarded = np.full(nTrials,False)
        tsCin = np.full(nTrials,np.nan)
        tsCout = np.full(nTrials,np.nan)
        tsChoice = np.full(nTrials,np.nan)
        tsRwd = np.full(nTrials,np.nan)
        tsPokeL = [[]]*nTrials
        tsPokeC = [[]]*nTrials
        tsPokeR = [[]]*nTrials
        FixBroke = np.full(nTrials,False)
        EarlyWithdrawal = np.full(nTrials,False)
        waitingTime = np.full(nTrials,np.nan)
        reactionTime = np.full(nTrials,np.nan)
        movementTime = np.full(nTrials,np.nan)
        stimDelay = np.full(nTrials,np.nan)
        feedbackDelay = np.full(nTrials,np.nan)
        assert(not np.isscalar(tsState0)), "Session is only 1 trial long. Aborting."
        assert(len(tsState0) > 20), "Session is only {} trials long. Aborting.".format(len(tsState0))
        tsState0 = tsState0 - tsState0[0]

        if not 'Ports_LMR' in self.params.index:
            self.params.loc['Ports_LMR'] = 'XXX'
            iTrial = 0
            while any([self.params.Ports_LMR[i] == 'X' for i in range(len(self.params.Ports_LMR))]) and iTrial < nTrials:
                listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
                stateTraj = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1] #from 1- to 0-based indexing
                events = np.array(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names)
                events = events[np.logical_and([n.startswith('Port') for n in events],[n.endswith('In') for n in events])]
                dic_events = {e:ts for e in events for ts in np.ravel(np.array(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[e].item()))}
                LMR = list(self.params.Ports_LMR)
                if LMR[0] == 'X' and 'Lin' in stateTraj:
                    try:
                        ts_stateLin = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item()[0]
                        LMR[0] = np.array(list(dic_events.keys()))[np.isclose(ts_stateLin,[dic_events[n] for n in dic_events.keys()])].item().split('Port')[1].split('In')[0]
                    except:
                        pass
                if LMR[1] == 'X' and 'Cin' in stateTraj:
                    try:
                        ts_stateCin = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin'].item()[0]
                        LMR[1] = np.array(list(dic_events.keys()))[np.isclose(ts_stateCin,[dic_events[n] for n in dic_events.keys()])].item().split('Port')[1].split('In')[0]
                    except:
                        pass
                if LMR[2] == 'X' and 'Rin' in stateTraj:
                    try:
                        ts_stateRin = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item()[0]
                        LMR[2] = np.array(list(dic_events.keys()))[np.isclose(ts_stateRin,[dic_events[n] for n in dic_events.keys()])].item().split('Port')[1].split('In')[0]
                    except:
                        pass
                self.params.Ports_LMR = "".join(LMR)
                iTrial+=1
            if any([self.params.Ports_LMR[i] == 'X' for i in range(len(self.params.Ports_LMR))]):
                raise RuntimeError("Couldn't figure port assignment (LMR). Aborting.")

        PortL = 'Port' + str(int(self.params.Ports_LMR))[0] + 'In'
        PortC = 'Port' + str(int(self.params.Ports_LMR))[1] + 'In'
        PortR = 'Port' + str(int(self.params.Ports_LMR))[2] + 'In'
        stateTraj = [[]]*nTrials

        for iTrial in range(nTrials):
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
            stateTraj[iTrial] = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1] #from 1- to 0-based indexing

            tsCin[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['wait_Cin'].item()[1]

            if any([PortL in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeL[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortL].item()

            if any([PortC in self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item().dtype.names]) :
                tsPokeC[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()[PortC].item()

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
                waitingTime[iTrial] = tsRwd[iTrial]-tsChoice[iTrial]
            else:
                ndx = np.array([n.startswith('Early') for n in stateTraj[iTrial]])
                ndx = np.logical_or(ndx,np.array([n.startswith('unrewarded') for n in stateTraj[iTrial]]))
                #
                if any(ndx):
                    mystate = np.array(stateTraj[iTrial])[ndx].item()
                    waitingTime[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[mystate].item()[0] - tsChoice[iTrial]

            if any(['stillSampling' in stateTraj[iTrial]]):
                stimDelay[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillSampling'].item()[1] - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin'].item()[0]
            else:
                stimDelay[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin'].item()).item()

            if any(['Lin' in stateTraj[iTrial]]) or any(['Rin' in stateTraj[iTrial]]):
                Sin = stateTraj[iTrial][np.logical_or([n == 'Lin' for n in stateTraj[iTrial]],[n == 'Rin' for n in stateTraj[iTrial]])][0]
                if any(['stillLin' in stateTraj[iTrial]]) or any(['stillRin' in stateTraj[iTrial]]):
                    stillSin = stateTraj[iTrial][np.logical_or([n == 'stillLin' for n in stateTraj[iTrial]],[n == 'stillRin' for n in stateTraj[iTrial]])].item()
                    feedbackDelay[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[stillSin].item()[1] - np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[Sin].item()).min()
                elif any(['Early' + Sin[0] + 'out' in stateTraj[iTrial]]):
                # elif any([n.startswith('Early') for n in stateTraj[iTrial]]):
                    feedbackDelay[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Early' + Sin[0] + 'out'].item()[1] - np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[Sin].item()).min()
                elif any([n.startswith('rewarded_') for n in stateTraj[iTrial]]):
                    rewarded_Sin = stateTraj[iTrial][[n.startswith('rewarded_') for n in stateTraj[iTrial]]].item()
                    feedbackDelay[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[rewarded_Sin].item()[0] - np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()[Sin].item()).min()

            reactionTime[iTrial] = np.nan if np.isclose(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()['Tup'].item(),self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillSampling'].item()[1]).any() else np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillSampling'].item()).item()
            movementTime[iTrial] = np.nan if np.isclose(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()['Tup'].item(),self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['wait_Sin'].item()[1]).any() else np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['wait_Sin'].item()).item()

        isBaitLeft = self.bpod['Custom'].item()['Baited'].item()['Left'].item()
        isBaitRight = self.bpod['Custom'].item()['Baited'].item()['Right'].item()
        isChoiceBaited = np.logical_or(np.logical_and(ChoiceLeft,isBaitLeft),
                                       np.logical_and(ChoiceRight,isBaitRight),
                                      )
        isLeftHi = self.bpod['Custom'].item()['LeftHi'].item().astype(bool)
        assert(isChoiceBaited[Rewarded].all()), "Impossible trials found: unbaited AND rewarded."
        # waitingTime[isChoiceBaited] = np.nan

        self.parsedData = pd.DataFrame({'iTrial': np.arange(nTrials),
                                        'isChoiceLeft': ChoiceLeft, 'isChoiceRight': ChoiceRight, 'isChoiceMiss': ChoiceMiss,'isLeftHi': isLeftHi,
                                        'isRewarded': Rewarded, 'isBrokeFix': FixBroke, 'isEarlyWithdr': EarlyWithdrawal, 'isBaitLeft':isBaitLeft, 'isBaitRight':isBaitRight, 'isChoiceBaited':isChoiceBaited,
                                        'stateTraj': stateTraj,'reactionTime':reactionTime, 'movementTime':movementTime, 'waitingTime': waitingTime, 'StimDelay': stimDelay, 'FeedbackDelay': feedbackDelay,
                                        'tsCin': tsCin, 'tsChoice': tsChoice, 'tsRwd': tsRwd,
                                        'tsPokeL': tsPokeL, 'tsPokeC': tsPokeC, 'tsPokeR': tsPokeR, 'tsState0': tsState0})

        self.parsedData = self.parsedData.set_index('iTrial')

        if not 'pLo' in self.params.index:
            warnings.warn('No params file. Estimating reward probabilities from data.')
            isLeftHi = self.parsedData.isLeftHi
            ndxPrevRwdL = np.hstack((False,np.logical_and(self.parsedData.isRewarded.iloc,self.parsedData.isChoiceLeft)[:-1])).astype(bool)
            ndxPrevRwdR = np.hstack((False,np.logical_and(self.parsedData.isRewarded.iloc,self.parsedData.isChoiceRight)[:-1])).astype(bool)

            baitHi = np.full(self.parsedData.isRewarded.shape,np.nan)
            baitHi[np.logical_and(isLeftHi,ndxPrevRwdL)] = self.parsedData.isBaitLeft.loc[np.logical_and(isLeftHi,ndxPrevRwdL)]
            baitHi[np.logical_and(np.logical_not(isLeftHi),ndxPrevRwdR)] = self.parsedData.isBaitRight.loc[np.logical_and(np.logical_not(isLeftHi),ndxPrevRwdR)]

            baitLo = np.full(self.parsedData.isRewarded.shape,np.nan)
            baitLo[np.logical_and(isLeftHi,ndxPrevRwdR)] = self.parsedData.isBaitRight.loc[np.logical_and(isLeftHi,ndxPrevRwdR)]
            baitLo[np.logical_and(np.logical_not(isLeftHi),ndxPrevRwdL)] = self.parsedData.isBaitLeft.loc[np.logical_and(np.logical_not(isLeftHi),ndxPrevRwdL)]

            self.params.loc['pHi'] = np.nanmean(baitHi)*100
            self.params.loc['pLo'] = pLo = np.nanmean(baitLo)*100

            # ndx = np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight)
            # ndx = np.logical_and(ndx,np.hstack((False,self.parsedData.isRewarded.iloc[:-1])))
            # leftHi = self.bpod['Custom'].item()['LeftHi'].item().astype(bool)
            # ndxHi = np.logical_or(np.logical_and(self.parsedData.isChoiceLeft,leftHi),
            #                       np.logical_and(self.parsedData.isChoiceRight,np.logical_not(leftHi)))
            # ndxHi = np.logical_and(ndx,ndxHi)
            # self.params.loc['pHi'] = self.parsedData.isRewarded.loc[ndxHi].mean()/np.logical_not(self.parsedData.isEarlyWithdr.loc[ndxHi]).mean()
            # ndxLo = np.logical_and(ndx,np.logical_not(ndxHi))
            # self.params.loc['pLo'] = self.parsedData.isRewarded.loc[ndxLo].sum()/np.logical_not(self.parsedData.isEarlyWithdr.loc[ndxLo]).sum()

        self.pred_lauglim()
        self.pred_idobs()
        self.pred_idobs2()

    def pred_lauglim(self,nhist=10):
        ndx = np.logical_not(np.logical_or(self.parsedData.isBrokeFix,self.parsedData.isChoiceMiss))
        tempbhv = self.parsedData.copy().loc[ndx,:]

        tempR = np.zeros(tempbhv.isChoiceLeft.shape)
        tempR[np.logical_and(tempbhv.isRewarded,tempbhv.isChoiceLeft)] = 1
        tempR[np.logical_and(tempbhv.isRewarded,tempbhv.isChoiceRight)] = -1
        R = np.zeros((len(tempR),nhist))

        tempC = np.zeros(tempbhv.isChoiceLeft.shape)
        tempC[tempbhv.isChoiceLeft] = 1
        tempC[tempbhv.isChoiceRight] = -1
        C = np.zeros((len(tempC),nhist))

        for i in range(nhist):
            C[i+1:,i] = tempC[:-i-1]
            R[i+1:,i] = tempR[:-i-1]

        desigm_R = pd.DataFrame(columns=['R_t-{0:0>2}'.format(i) for i in range(1,nhist+1)],index=np.arange(len(ndx)),data=np.nan)
        desigm_R.loc[ndx,:] = R
        desigm_R.dropna(axis=0,inplace=True,how='all')
        desigm_C = pd.DataFrame(columns=['C_t-{0:0>2}'.format(i) for i in range(1,nhist+1)],index=np.arange(len(ndx)),data=np.nan)
        desigm_C.loc[ndx,:] = C
        desigm_C.dropna(axis=0,inplace=True,how='all')

        X = pd.concat((desigm_R,desigm_C),axis=1,sort=False)

        tempy = np.full(tempbhv.isChoiceLeft.shape,np.nan)
        tempy[tempbhv.isChoiceLeft] = 1
        tempy[tempbhv.isChoiceRight] = 0
        y = pd.Series(index=X.index,name='obsChoice',data=tempy,dtype=bool)

        try:
            self.lauglim = linear_model.LogisticRegression(solver='lbfgs').fit(X=X,y=y)
            self.lauglim.dataX = X
            self.lauglim.datay = y
            self.parsedData.loc[:,'lauglim_pLeft'] = np.nan
            self.parsedData.loc[ndx,'lauglim_pLeft'] = self.lauglim.predict_proba(X)[:,1]
            self.parsedData.loc[:,'lauglim_isGreedy'] = np.logical_or(np.logical_and(self.parsedData.lauglim_pLeft>0.5,self.parsedData.isChoiceLeft),
                                                                      np.logical_and(self.parsedData.lauglim_pLeft<0.5,self.parsedData.isChoiceRight))
        except Exception as e:
            warnings.warn('Failed to compute logistic regression P(choice) = f(trial history)')
            print(e)

    def pred_idobs(self):

        pHi=self.params.pHi/100.
        pLo=self.params.pLo/100.
        leftHi=self.bpod['Custom'].item()['LeftHi'].item().astype(bool)
        isChoiceLeft=self.parsedData.isChoiceLeft
        isChoiceRight=self.parsedData.isChoiceRight
        isValid=(isChoiceLeft | isChoiceRight)

        postl = np.full(leftHi.shape,np.nan) # P(Reward|Left Choice)
        postl[0] = 1.
        postr = np.full(leftHi.shape,np.nan) # P(Reward|Right Choice)
        postr[0] = 1.

        for iTrial in range(1,len(leftHi)):
            if not isValid[iTrial-1]:
                postl[iTrial] = postl[iTrial-1]
                postr[iTrial] = postr[iTrial-1]
                continue
            pl = pHi if leftHi[iTrial] else pLo
            pr = pLo if leftHi[iTrial] else pHi

            postl[iTrial] = pl if isChoiceLeft[iTrial-1] else postl[iTrial-1]+(1-postl[iTrial-1])*pl
            postr[iTrial] = pr if isChoiceRight[iTrial-1] else postr[iTrial-1]+(1-postr[iTrial-1])*pr

        self.parsedData.loc[:,'idobs_pLeft'] = postl
        self.parsedData.loc[:,'idobs_pRight'] = postr
        lo = np.log(self.parsedData.idobs_pLeft/self.parsedData.idobs_pRight)
        self.parsedData.loc[:,'idobs_isGreedy'] = np.logical_or(np.logical_and(lo>0,self.parsedData.isChoiceLeft),
                                                          np.logical_and(lo<0,self.parsedData.isChoiceRight))

    def pred_idobs2(self):
        # Takes into account evidence that waiting times provides about whether a reward is available or not
        # if self.params.FeedbackDelaySelection != 3.00000:
        #     return
        if not 'FeedbackDelaySelection' in self.params.index:
            return
        thisDelay = self.bpod['Settings'].item()['GUIMeta'].item()['FeedbackDelaySelection'].item().item()[1][self.params.FeedbackDelaySelection.astype(int)-1]
        pHi=self.params.pHi/100.
        pLo=self.params.pLo/100.
        leftHi=self.bpod['Custom'].item()['LeftHi'].item().astype(bool)
        isChoiceLeft=self.parsedData.isChoiceLeft.copy()
        isChoiceRight=self.parsedData.isChoiceRight.copy()
        isValid=(isChoiceLeft | isChoiceRight)
        isRewarded=self.parsedData.isRewarded

        postl = np.full(leftHi.shape,np.nan) # P(Reward|Left Choice)
        postl[0] = 1.
        postr = np.full(leftHi.shape,np.nan) # P(Reward|Right Choice)
        postr[0] = 1.

        wt_min = self.params.FeedbackDelayMin
        wt_max = self.params.FeedbackDelayMax
        wt_lambda = 1./self.params.FeedbackDelayTau
        for iTrial in range(1,len(leftHi)):
            if not isValid[iTrial-1]:
                postl[iTrial] = postl[iTrial-1]
                postr[iTrial] = postr[iTrial-1]
                continue

            pl = pHi if leftHi[iTrial] else pLo
            pr = pLo if leftHi[iTrial] else pHi

            postl[iTrial] = postl[iTrial-1]+(1-postl[iTrial-1])*pl
            postr[iTrial] = postr[iTrial-1]+(1-postr[iTrial-1])*pr

            if isRewarded[iTrial-1]:
                postl[iTrial] = pl if isChoiceLeft[iTrial-1] else postl[iTrial]
                postr[iTrial] = pr if isChoiceRight[iTrial-1] else postr[iTrial]
            else:
                prior = postl[iTrial-1] if isChoiceLeft[iTrial-1] else postr[iTrial-1]
                assert(isChoiceLeft[iTrial-1] or isChoiceRight[iTrial-1]), "Failed to catch an invalid trial"
                wt = self.parsedData.waitingTime[iTrial-1]
                if thisDelay == 'TruncExp':
                    p_wt_given_r = (np.exp(-wt_lambda*wt) - np.exp(-wt_lambda*wt_max))/(np.exp(-wt_lambda*wt_min) - np.exp(-wt_lambda*wt_max)) # Probability of waiting wt(s) before reward delivery, given that it was a rewarded trial
                elif thisDelay == 'Uniform':
                    p_wt_given_r = (wt_max-wt)/(wt_max-wt_min)
                elif thisDelay == 'Fix':
                    p_wt_given_r = float(self.parsedData.isEarlyWithdr[iTrial])
                else:
                    p_wt_given_r = float(self.parsedData.isEarlyWithdr[iTrial])
                    warnings.warn('Cannot compute p(R|wt) for delay type {}'.format(thisDelay))
                normalizer = 1+prior*(p_wt_given_r-1)
                p_r_given_wt = prior * p_wt_given_r/normalizer

                if isChoiceLeft[iTrial-1]:
                    postl[iTrial] = p_r_given_wt + (1-p_r_given_wt)*pl
                if isChoiceRight[iTrial-1]:
                    postr[iTrial] = p_r_given_wt + (1-p_r_given_wt)*pr
            if np.isnan(postl[iTrial]).any() or np.isnan(postr[iTrial]).any():
                warnings.warn("nan found at trial {}".format(iTrial))

                # print("wt: {:.2f} \t prior:{:.2f} \t p_wt_given_r:{:.2f} \t p_r_given_wt:{:.2f}".format(wt,prior,p_wt_given_r,p_r_given_wt))

        self.parsedData.loc[:,'idobs2_pLeft'] = postl # P(reward | left choice)
        self.parsedData.loc[:,'idobs2_pRight'] = postr # P(reward | right choice)
        lo = np.log(self.parsedData.idobs2_pLeft/self.parsedData.idobs2_pRight)
        self.parsedData.loc[:,'idobs2_isGreedy'] = np.logical_or(np.logical_and(lo>0,self.parsedData.isChoiceLeft),
                                                                 np.logical_and(lo<0,self.parsedData.isChoiceRight))


    def dailyfig(self,filepath_fig='None'): # filepath_fig is fullpath for figure file

        colors_lr=('xkcd:mango','xkcd:grass green')#,)'xkcd:water blue','xkcd:scarlet',
        fs_lab=12
        fs_title=18
        lw=2
        facealpha=.4
        hf, ha = plt.subplots(3,3,figsize=(10,10))

        ## Panel A - Cumulative Trials

        ha[0,0].plot(np.cumsum(self.parsedData.isChoiceRight),np.cumsum(self.parsedData.isChoiceLeft),linewidth=lw)
        ha[0,0].set_aspect(1)
        ha[0,0].set_xlabel('Fraction right choices',fontsize=fs_lab)
        ha[0,0].set_ylabel('Fraction left choices',fontsize=fs_lab)

        ## Panel B - StimDelay

        bins = np.histogram_bin_edges(self.parsedData.StimDelay.dropna(),bins='sturges')
        ha[0,1].hist(self.parsedData.StimDelay.loc[np.logical_not(self.parsedData.isBrokeFix)],color='xkcd:blue',bins=bins,alpha=facealpha,label='valid')
        ha[0,1].hist(self.parsedData.StimDelay.loc[self.parsedData.isBrokeFix],color='xkcd:red',bins=bins,alpha=facealpha,label='brokeFix')
        ha[0,1].legend(fancybox=True, framealpha=0.5)
        ha[0,1].set_xlabel('StimDelay',fontsize=fs_lab)
        ha[0,1].set_ylabel('Trial counts',fontsize=fs_lab)

        ## Panel C - FeedbackDelay

        feedbackDelay = self.parsedData.FeedbackDelay.copy()
        feedbackDelay.loc[self.parsedData.isChoiceBaited] = np.nan
        bins = np.histogram_bin_edges(feedbackDelay.dropna(),bins='sturges')
        ha[0,2].hist(feedbackDelay.loc[np.logical_not(self.parsedData.isEarlyWithdr)],color='xkcd:blue',bins=bins,alpha=facealpha,label='valid')
        ha[0,2].hist(feedbackDelay.loc[self.parsedData.isEarlyWithdr],color='xkcd:red',bins=bins,alpha=facealpha,label='earlyWithdr')
        ha[0,2].legend(fancybox=True, framealpha=0.5)
        ha[0,2].set_xlabel('FeedbackDelay',fontsize=fs_lab)
        if 'FeedbackDelaySelection' in self.bpod['Settings'].item()['GUIMeta'].item().dtype.names:
            thisDelay = self.bpod['Settings'].item()['GUIMeta'].item()['FeedbackDelaySelection'].item().item()[1][self.params.FeedbackDelaySelection.astype(int)-1]
        else :
            thisDelay = 'AutoIncr' if self.bpod['Settings'].item()['GUI'].item()['AutoIncrFeedback'].item() == 1 else 'Fix'
        ha[0,2].set_title(thisDelay)

        ## Panel D - Matching

        df_match = pd.DataFrame({'iBlock':self.bpod['Custom'].item()['BlockNumber'].item(),
                         'isChoiceLeft':self.parsedData.isChoiceLeft,
                         'isChoiceRight':self.parsedData.isChoiceRight,
                         'isRwdLeft':np.logical_and(self.parsedData.isRewarded,self.parsedData.isChoiceLeft),
                         'isRwdRight':np.logical_and(self.parsedData.isRewarded,self.parsedData.isChoiceRight),
                         # 'lo':np.log(self.parsedData.idobs2_pLeft/self.parsedData.idobs2_pRight),
                         'isLeftHi':self.bpod['Custom'].item()['LeftHi'].item()
                        })


        piv_match = df_match.pivot_table(columns='iBlock').T

        ha[1,0].plot([0,1],[0,1],'k',alpha=facealpha)
        for iSide in np.unique(piv_match.isLeftHi.astype(int)):
            ndx = piv_match.isLeftHi.astype(int)==iSide
            ha[1,0].scatter(piv_match.isRwdLeft[ndx]/(piv_match.isRwdLeft[ndx]+piv_match.isRwdRight[ndx]),
                        piv_match['isChoiceLeft'][ndx],
                        color=colors_lr[iSide], label=['leftLo','leftHi'][iSide])
        ha[1,0].legend(fancybox=True, framealpha=0.5)
        ha[1,0].set_xlim([-.1,1.1])
        ha[1,0].set_ylim([-.1,1.1])
        ha[1,0].set_aspect(1)
        ha[1,0].set_xlabel('Fraction left rewards',fontsize=fs_lab)
        ha[1,0].set_ylabel('Fraction left choices',fontsize=fs_lab)

        ## Panel E - Choice Kernel (LauGlim)
        if hasattr(self,'lauglim'):
            nhist=int(len(np.ravel(self.lauglim.coef_))/2)

            ha[1,1].plot([0,nhist],np.ones(2)*self.lauglim.intercept_.item(),label='bias',linewidth=lw,alpha=facealpha)
            ha[1,1].plot(np.ravel(self.lauglim.coef_)[nhist+1:],label='cho',linewidth=lw)
            ha[1,1].plot(np.ravel(self.lauglim.coef_)[:nhist],label='rwd',linewidth=lw)
        ha[1,1].set_xlabel('nTrials back',fontsize=fs_lab)
        ha[1,1].set_ylabel('RegressCoeff',fontsize=fs_lab)
        ha[1,1].legend(fancybox=True, framealpha=0.5)

        ## Panel F - Psychometrics

        nbins = 8
        if 'lauglim_pLeft' in self.parsedData.columns:
            df_psyc_lauglim = pd.DataFrame({'lo':np.log(self.parsedData.lauglim_pLeft/(1-self.parsedData.lauglim_pLeft)),
                                            'isChoiceLeft':self.parsedData.isChoiceLeft})
            df_psyc_lauglim.dropna(inplace=True)
            df_psyc_lauglim.loc[:,'bin'] = np.digitize(df_psyc_lauglim.lo,np.percentile(df_psyc_lauglim.lo,np.linspace(0,100,nbins+1)))
            df_psyc_lauglim.loc[df_psyc_lauglim.bin==df_psyc_lauglim.bin.max(),'bin']=nbins
            piv_psyc_lauglim = df_psyc_lauglim.pivot_table(columns='bin').T
            ha[1,2].scatter(piv_psyc_lauglim.lo,piv_psyc_lauglim.isChoiceLeft,label='lauglim')

        if 'idobs_pLeft' in self.parsedData.columns:
            df_psyc_idobs = pd.DataFrame({'lo':np.log(self.parsedData.idobs_pLeft/self.parsedData.idobs_pRight),
                                          'isChoiceLeft':self.parsedData.isChoiceLeft})
            df_psyc_idobs = df_psyc_idobs.loc[np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight),:].copy()
            df_psyc_idobs.loc[:,'bin'] = np.digitize(df_psyc_idobs.lo,np.unique(np.percentile(df_psyc_idobs.lo,np.linspace(0,100,nbins+1))))
            df_psyc_idobs.loc[df_psyc_idobs.bin==df_psyc_idobs.bin.max(),'bin']=nbins
            piv_psyc_idobs = df_psyc_idobs.pivot_table(columns='bin').T
            ha[1,2].scatter(piv_psyc_idobs.lo,piv_psyc_idobs.isChoiceLeft,label='idobs')

        if 'idobs2_pLeft' in self.parsedData.columns:
            df_psyc_idobs2 = pd.DataFrame({'lo':np.log(self.parsedData.idobs2_pLeft/self.parsedData.idobs2_pRight),
                                           'isChoiceLeft':self.parsedData.isChoiceLeft})
            df_psyc_idobs2 = df_psyc_idobs2.loc[np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight),:].copy()
            df_psyc_idobs2.loc[:,'bin'] = np.digitize(df_psyc_idobs2.lo,np.unique(np.percentile(df_psyc_idobs2.lo,np.linspace(0,100,nbins+1))))
            df_psyc_idobs2.loc[df_psyc_idobs2.bin==df_psyc_idobs2.bin.max(),'bin']=nbins
            piv_psyc_idobs2 = df_psyc_idobs2.pivot_table(columns='bin').T
            ha[1,2].scatter(piv_psyc_idobs2.lo,piv_psyc_idobs2.isChoiceLeft,label='idobs2')

        ha[1,2].set_ylim([-.1,1.1])
        x_hat = np.linspace(ha[1,2].get_xlim()[0],ha[1,2].get_xlim()[1],50)
        ha[1,2].plot(x_hat,1/(1+np.exp(-x_hat)),'k',alpha=facealpha,linewidth=lw)
        ha[1,2].set_xlabel(r'$ log \left( \frac{P_{Left}}{P_{Right}} \right)$',fontsize=fs_lab)
        ha[1,2].set_ylabel('Fraction left choices',fontsize=fs_lab)
        ha[1,2].legend(fancybox=True, framealpha=0.5)

        ## Panel G - Vevaio LauGlim
        nbins = 8

        if 'lauglim_pLeft' in self.parsedData.columns:

            ndx = np.logical_and(self.parsedData.lauglim_isGreedy,
                                 np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isChoiceBaited))
            if ndx.sum() > 0:
                temp_bhv = self.parsedData[ndx].copy()
                df_vevaio_lauglim = pd.DataFrame({'lo':np.log(temp_bhv.lauglim_pLeft/(1-temp_bhv.lauglim_pLeft)),
                                                  'wt':temp_bhv.waitingTime})
                df_vevaio_lauglim.loc[:,'bin'] = np.digitize(df_vevaio_lauglim.lo,np.unique(np.percentile(df_vevaio_lauglim.lo.dropna(),np.linspace(0,100,nbins+1))))
                df_vevaio_lauglim.loc[df_vevaio_lauglim.loc[:,'bin']>nbins,'bin'] = nbins
                ha[2,0].scatter(df_vevaio_lauglim.lo,df_vevaio_lauglim.wt,color='xkcd:green',label='_nolegend_',alpha=.1)
                piv_vevaio_lauglim = df_vevaio_lauglim.pivot_table(columns='bin',aggfunc=np.median).T
                ha[2,0].plot(piv_vevaio_lauglim.lo,piv_vevaio_lauglim.wt,color='xkcd:green',label='greedy',linewidth=lw)
                ha[2,0].legend(fancybox=True, framealpha=0.5)

            ndx = np.logical_and(np.logical_not(self.parsedData.lauglim_isGreedy),
                                 np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
            if ndx.sum() > 0:
                temp_bhv = self.parsedData[ndx].copy()
                df_vevaio_lauglim = pd.DataFrame({'lo':np.log(temp_bhv.lauglim_pLeft/(1-temp_bhv.lauglim_pLeft)),
                                                  'wt':temp_bhv.waitingTime})
                df_vevaio_lauglim.loc[:,'bin'] = np.digitize(df_vevaio_lauglim.lo,np.unique(np.percentile(df_vevaio_lauglim.lo.dropna(),np.linspace(0,100,nbins+1))))
                df_vevaio_lauglim.loc[df_vevaio_lauglim.loc[:,'bin']>nbins,'bin'] = nbins
                ha[2,0].scatter(df_vevaio_lauglim.lo,df_vevaio_lauglim.wt,color='xkcd:red',label='_nolegend_',alpha=.1)
                piv_vevaio_lauglim = df_vevaio_lauglim.pivot_table(columns='bin',aggfunc=np.median).T
                ha[2,0].plot(piv_vevaio_lauglim.lo,piv_vevaio_lauglim.wt,color='xkcd:red',label='notGreedy',linewidth=lw)
                ha[2,0].legend(fancybox=True, framealpha=0.5)
        ha[2,0].set_xlabel(r'$ log \left( \frac{P_{Left}}{P_{Right}} \right)$',fontsize=fs_lab)
        ha[2,0].set_ylabel('waiting time (s)',fontsize=fs_lab)
        ha[2,0].set_title('lauglim',fontsize=fs_title)

        ## Panel H - Vevaio Ideal Observer 1 (ignores waiting time)
        if 'idobs_pLeft' in self.parsedData.columns:
            ndx = np.logical_and(self.parsedData.idobs_isGreedy,
                                 np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isChoiceBaited))
            if ndx.sum() > 0:
                temp_bhv = self.parsedData[ndx].copy()
                df_vevaio_idobs = pd.DataFrame({'lo':np.log(temp_bhv.idobs_pLeft/temp_bhv.idobs_pRight),
                                                  'wt':temp_bhv.waitingTime})
                df_vevaio_idobs.loc[:,'bin'] = np.digitize(df_vevaio_idobs.lo,np.unique(np.percentile(df_vevaio_idobs.lo.dropna(),np.linspace(0,100,nbins+1))))
                df_vevaio_idobs.loc[df_vevaio_idobs.loc[:,'bin']>nbins,'bin'] = nbins
                ha[2,1].scatter(df_vevaio_idobs.lo,df_vevaio_idobs.wt,color='xkcd:green',label='_nolegend_',alpha=.1)
                piv_vevaio_idobs = df_vevaio_idobs.pivot_table(columns='bin',aggfunc=np.median).T
                ha[2,1].plot(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:green',label='greedy',linewidth=lw)
                ha[2,1].legend(fancybox=True, framealpha=0.5)

            if ndx.sum() > 0:
                ndx = np.logical_and(np.logical_not(self.parsedData.idobs_isGreedy),
                                     np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
                ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
                temp_bhv = self.parsedData[ndx].copy()
                df_vevaio_idobs = pd.DataFrame({'lo':np.log(temp_bhv.idobs_pLeft/temp_bhv.idobs_pRight),
                                                  'wt':temp_bhv.waitingTime})
                df_vevaio_idobs.loc[:,'bin'] = np.digitize(df_vevaio_idobs.lo,np.unique(np.percentile(df_vevaio_idobs.lo.dropna(),np.linspace(0,100,nbins+1))))
                df_vevaio_idobs.loc[df_vevaio_idobs.loc[:,'bin']>nbins,'bin'] = nbins
                ha[2,1].scatter(df_vevaio_idobs.lo,df_vevaio_idobs.wt,color='xkcd:red',label='_nolegend_',alpha=.1)
                piv_vevaio_idobs = df_vevaio_idobs.pivot_table(columns='bin',aggfunc=np.median).T
                ha[2,1].plot(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:red',label='notGreedy',linewidth=lw)
                ha[2,1].legend(fancybox=True, framealpha=0.5)
        ha[2,1].set_xlabel(r'$ log \left( \frac{P_{Left}}{P_{Right}} \right)$',fontsize=fs_lab)
        ha[2,1].set_ylabel('waiting time (s)',fontsize=fs_lab)
        ha[2,1].set_title('idobs',fontsize=fs_title)

        ## Panel I - Vevaio Ideal Observer 2 (takes waiting time evidence into account)

        if 'idobs2_pLeft' in self.parsedData.columns:
            ndx = np.logical_and(self.parsedData.idobs2_isGreedy,
                                 np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isChoiceBaited))
            if ndx.sum() > 0:
                temp_bhv = self.parsedData[ndx].copy()
                df_vevaio_idobs2 = pd.DataFrame({'lo':np.log(temp_bhv.idobs2_pLeft/temp_bhv.idobs2_pRight),
                                                  'wt':temp_bhv.waitingTime})
                df_vevaio_idobs2.loc[:,'bin'] = np.digitize(df_vevaio_idobs2.lo,np.unique(np.percentile(df_vevaio_idobs2.lo.dropna(),np.linspace(0,100,nbins+1))))
                df_vevaio_idobs2.loc[df_vevaio_idobs2.loc[:,'bin']>nbins,'bin'] = nbins
                ha[2,2].scatter(df_vevaio_idobs2.lo,df_vevaio_idobs2.wt,color='xkcd:green',label='_nolegend_',alpha=.1)
                piv_vevaio_idobs2 = df_vevaio_idobs2.pivot_table(columns='bin',aggfunc=np.median).T
                ha[2,2].plot(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:green',label='greedy',linewidth=lw)

            if ndx.sum() > 0:
                ndx = np.logical_and(np.logical_not(self.parsedData.idobs2_isGreedy),
                                     np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
                ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
                temp_bhv = self.parsedData[ndx].copy()
                df_vevaio_idobs2 = pd.DataFrame({'lo':np.log(temp_bhv.idobs2_pLeft/temp_bhv.idobs2_pRight),
                                                  'wt':temp_bhv.waitingTime})
                df_vevaio_idobs2.loc[:,'bin'] = np.digitize(df_vevaio_idobs2.lo,np.unique(np.percentile(df_vevaio_idobs2.lo.dropna(),np.linspace(0,100,nbins+1))))
                df_vevaio_idobs2.loc[df_vevaio_idobs2.loc[:,'bin']>nbins,'bin'] = nbins
                ha[2,2].scatter(df_vevaio_idobs2.lo,df_vevaio_idobs2.wt,color='xkcd:red',label='_nolegend_',alpha=.1)
                piv_vevaio_idobs2 = df_vevaio_idobs2.pivot_table(columns='bin',aggfunc=np.median).T
                ha[2,2].plot(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:red',label='notGreedy',linewidth=lw)
                ha[2,2].legend(fancybox=True, framealpha=0.5)
            ha[2,2].set_xlabel(r'$ log \left( \frac{P_{Left}}{P_{Right}} \right)$',fontsize=fs_lab)
            ha[2,2].set_ylabel('waiting time (s)',fontsize=fs_lab)
            ha[2,2].set_title('idobs2',fontsize=fs_title)

        plt.suptitle(self.fname)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if filepath_fig != 'None':
            os.makedirs(os.path.split(filepath_fig)[0],exist_ok=True)
            plt.savefig(filepath_fig)
            plt.close()

        """

        if not dfInt.empty:
            bins = np.arange(np.percentile(dfInt['interval'],99))
            for iArm in set(dfInt['armNo']):
                dfInt_arm=dfInt[dfInt['armNo']==iArm]
                x=dfInt_arm.interval
                x=np.clip(x,0,bins.max())
                ha[1,0].hist(x,bins=bins,cumulative=False,density=False,histtype='step',color=colors[iArm],lw=lw)
                ha[1,0].hist(x,bins=bins,cumulative=False,density=False,histtype='stepfilled',alpha=facealpha,color=colors[iArm],edgecolor='None')


        """



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

        haHist.hist(hisPkL, bins='sturges', histtype='step', color='xkcd:mango', density=False)
        haHist.hist(hisPkR, bins='sturges', histtype='step', color='xkcd:darkish green', density=False)
        haHist.hist(hisPkC, bins='sturges', histtype='step', color='xkcd:scarlet', density=False)
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
