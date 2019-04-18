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
        stimDelay = np.full(nTrials,np.nan)
        feedbackDelay = np.full(nTrials,np.nan)
        tsState0 = tsState0 - tsState0[0]

        """
        Feedback = np.full(nTrials,False)#<---
        FixDur = np.full(nTrials,np.nan)#<---
        MT = np.full(nTrials,np.nan)#<---
        ST = np.full(nTrials,np.nan)#<---
        tsErrTone = np.full(nTrials,np.nan)
        """
        for iTrial in range(nTrials):
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
                else:
                    pass

            #
            # else:
            #     feedbackDelay[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin'].item()).item()
            #
            # if any(['stillSampling' in stateTraj[iTrial]]):
            #     feedbackDelay[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillSampling'].item()[1] - self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin'].item()[0]
            # else:
            #     feedbackDelay[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Cin'].item()).item()
            #     # if ChoiceLeft[iTrial]:
            #     #     waitingTime[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item())
            #     # if ChoiceRight[iTrial]:
            #     #     waitingTime[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item())
            #
            # # if any([n.startswith('start_') for n in stateTraj[iTrial]]):
            # #     if 'start_Lin' in stateTraj[iTrial]:
            # #         pass
            # #     if 'start_Rin' in stateTraj[iTrial]:
            # #         pass
            #
            # if any(['stillRin' in stateTraj[iTrial]]):
            #     waitingTime[iTrial] = np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillRin'].item())[-1]-np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item())[0]
            # if any(['stillLin' in stateTraj[iTrial]]):
            #     waitingTime[iTrial] = np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillLin'].item())[-1]-np.ravel(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Lin'].item())[0]
            # # if waitingTime[iTrial] < 0:
            # #     print(iTrial)
            # #     [print(i,n) for i, n in enumerate(stateTraj[iTrial])]
            # #     break
            # #     # print(stateTraj[iTrial]) #[[n.startswith('still') for n in stateTraj[iTrial]]])
            # #     # if ChoiceLeft[iTrial]:
            # #     #     waitingTime[iTrial] = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillLin'].item()[1] - \
            # #     #     self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['stillLin'].item()[0]
            # #     # if ChoiceRight[iTrial]:
            # #     #     waitingTime[iTrial] = np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['Rin'].item())



        # assert all(np.logical_xor(ChoiceRight,ChoiceLeft))

        self.parsedData = pd.DataFrame({'iTrial': np.arange(nTrials),
                                        'isChoiceLeft': ChoiceLeft, 'isChoiceRight': ChoiceRight, 'isChoiceMiss': ChoiceMiss,
                                        'isRewarded': Rewarded, 'isBrokeFix': FixBroke, 'isEarlyWithdr': EarlyWithdrawal,
                                        'stateTraj': stateTraj, 'WT': waitingTime, 'StimDelay': stimDelay, 'FeedbackDelay': feedbackDelay,
                                        'tsCin': tsCin, 'tsChoice': tsChoice, 'tsRwd': tsRwd,
                                        'tsPokeL': tsPokeL, 'tsPokeC': tsPokeC, 'tsPokeR': tsPokeR, 'tsState0': tsState0})

        self.parsedData = self.parsedData.set_index('iTrial')
        self.pred_lauglim()
        self.pred_idobs()
        self.pred_idobs2()

    def pred_lauglim(self,nhist=10):
        ndx = np.logical_and(np.logical_not(self.parsedData.isBrokeFix),
                             np.logical_not(self.parsedData.isChoiceMiss))
        # tempbhv = self.parsedData.copy().loc[ndx,:]

        y = np.full(self.parsedData.isChoiceLeft.shape,np.nan)
        y[self.parsedData.isChoiceLeft] = 1
        y[self.parsedData.isChoiceRight] = 0

        tempR = np.zeros(self.parsedData.isChoiceLeft.shape)
        tempR[np.logical_and(self.parsedData.isRewarded,self.parsedData.isChoiceLeft)] = 1
        tempR[np.logical_and(self.parsedData.isRewarded,self.parsedData.isChoiceRight)] = -1
        R = np.zeros((len(tempR),nhist))

        tempC = np.zeros(self.parsedData.isChoiceLeft.shape)
        tempC[self.parsedData.isChoiceLeft] = 1
        tempC[self.parsedData.isChoiceRight] = -1
        C = np.zeros((len(tempC),nhist))

        for i in range(nhist):
            C[i+1:,i] = tempC[:-i-1]
            R[i+1:,i] = tempR[:-i-1]

        X = np.hstack([R,C])
        X = X[np.logical_not(np.isnan(y)),:]
        y = y[np.logical_not(np.isnan(y))]

        self.parsedData.loc[:,'lauglim_pLeft'] = np.nan
        try:
            self.lauglim = linear_model.LogisticRegression(solver='lbfgs').fit(X=X,y=y)
            self.parsedData.loc[ndx,'lauglim_pLeft'] = self.lauglim.predict_proba(X)[:,1]
        except Exception as e:
            print(e)
        self.parsedData.loc[:,'lauglim_isGreedy'] = np.logical_or(np.logical_and(self.parsedData.lauglim_pLeft>0.5,self.parsedData.isChoiceLeft),
                                                                  np.logical_and(self.parsedData.lauglim_pLeft<0.5,self.parsedData.isChoiceRight))

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
        thisDelay = self.bpod['Settings'].item()['GUIMeta'].item()['FeedbackDelaySelection'].item().item()[1][self.params.FeedbackDelaySelection.astype(int)]
        pHi=self.params.pHi/100.
        pLo=self.params.pLo/100.
        leftHi=self.bpod['Custom'].item()['LeftHi'].item().astype(bool)
        isChoiceLeft=self.parsedData.isChoiceLeft
        isChoiceRight=self.parsedData.isChoiceRight
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
                wt = self.parsedData.WT[iTrial-1]
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

                # print("wt: {:.2f} \t prior:{:.2f} \t p_wt_given_r:{:.2f} \t p_r_given_wt:{:.2f}".format(wt,prior,p_wt_given_r,p_r_given_wt))

        self.parsedData.loc[:,'idobs2_pLeft'] = postl # P(reward | left choice)
        self.parsedData.loc[:,'idobs2_pRight'] = postr # P(reward | right choice)
        lo = np.log(self.parsedData.idobs2_pLeft/self.parsedData.idobs2_pRight)
        self.parsedData.loc[:,'idobs2_isGreedy'] = np.logical_or(np.logical_and(lo>0,self.parsedData.isChoiceLeft),
                                                                 np.logical_and(lo<0,self.parsedData.isChoiceRight))


    def dailyfig(self):

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

        bins = np.histogram_bin_edges(self.parsedData.StimDelay.dropna())
        ha[0,1].hist(self.parsedData.StimDelay.loc[np.logical_not(self.parsedData.isBrokeFix)],color='xkcd:blue',bins=bins,alpha=facealpha,label='valid')
        ha[0,1].hist(self.parsedData.StimDelay.loc[self.parsedData.isBrokeFix],color='xkcd:red',bins=bins,alpha=facealpha,label='brokeFix')
        ha[0,1].legend(fancybox=True, framealpha=0.5)
        ha[0,1].set_xlabel('StimDelay',fontsize=fs_lab)
        ha[0,1].set_ylabel('Trial counts',fontsize=fs_lab)

        ## Panel C - FeedbackDelay

        bins = np.histogram_bin_edges(self.parsedData.FeedbackDelay.dropna())

        ha[0,2].hist(self.parsedData.FeedbackDelay.loc[np.logical_not(self.parsedData.isEarlyWithdr)],color='xkcd:blue',bins=bins,alpha=facealpha,label='valid')
        ha[0,2].hist(self.parsedData.FeedbackDelay.loc[self.parsedData.isEarlyWithdr],color='xkcd:red',bins=bins,alpha=facealpha,label='earlyWithdr')
        ha[0,2].legend(fancybox=True, framealpha=0.5)
        ha[0,2].set_xlabel('FeedbackDelay',fontsize=fs_lab)

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
        nhist=int(len(np.ravel(self.lauglim.coef_))/2)

        ha[1,1].plot([0,nhist],np.ones(2)*self.lauglim.intercept_.item(),label='bias',linewidth=lw,alpha=facealpha)
        ha[1,1].plot(np.ravel(self.lauglim.coef_)[nhist+1:],label='cho',linewidth=lw)
        ha[1,1].plot(np.ravel(self.lauglim.coef_)[:nhist],label='rwd',linewidth=lw)
        ha[1,1].set_xlabel('nTrials back',fontsize=fs_lab)
        ha[1,1].set_ylabel('RegressCoeff',fontsize=fs_lab)
        ha[1,1].legend(fancybox=True, framealpha=0.5)

        ## Panel F - Psychometrics

        nbins = 8

        df_psyc_lauglim = pd.DataFrame({'lo':np.log(self.parsedData.lauglim_pLeft/(1-self.parsedData.lauglim_pLeft)),
                                        'isChoiceLeft':self.parsedData.isChoiceLeft})
        df_psyc_lauglim.dropna(inplace=True)
        df_psyc_lauglim.loc[:,'bin'] = np.digitize(df_psyc_lauglim.lo,np.percentile(df_psyc_lauglim.lo,np.linspace(0,100,nbins+1)))
        df_psyc_lauglim.loc[df_psyc_lauglim.bin==df_psyc_lauglim.bin.max(),'bin']=nbins
        piv_psyc_lauglim = df_psyc_lauglim.pivot_table(columns='bin').T

        df_psyc_idobs = pd.DataFrame({'lo':np.log(self.parsedData.idobs_pLeft/self.parsedData.idobs_pRight),
                                      'isChoiceLeft':self.parsedData.isChoiceLeft})
        df_psyc_idobs = df_psyc_idobs.loc[np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight),:].copy()
        df_psyc_idobs.loc[:,'bin'] = np.digitize(df_psyc_idobs.lo,np.percentile(df_psyc_idobs.lo,np.linspace(0,100,nbins+1)))
        df_psyc_idobs.loc[df_psyc_idobs.bin==df_psyc_idobs.bin.max(),'bin']=nbins
        piv_psyc_idobs = df_psyc_idobs.pivot_table(columns='bin').T

        ha[1,2].scatter(piv_psyc_lauglim.lo,piv_psyc_lauglim.isChoiceLeft,label='lauglim')
        ha[1,2].scatter(piv_psyc_idobs.lo,piv_psyc_idobs.isChoiceLeft,label='idobs')

        if 'idobs2_pLeft' in self.parsedData.columns:
            df_psyc_idobs2 = pd.DataFrame({'lo':np.log(self.parsedData.idobs2_pLeft/self.parsedData.idobs2_pRight),
                                           'isChoiceLeft':self.parsedData.isChoiceLeft})
            df_psyc_idobs2 = df_psyc_idobs2.loc[np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight),:].copy()
            df_psyc_idobs2.loc[:,'bin'] = np.digitize(df_psyc_idobs2.lo,np.percentile(df_psyc_idobs2.lo,np.linspace(0,100,nbins+1)))
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

        nbins = 6

        ndx = np.logical_and(self.parsedData.lauglim_isGreedy,
                             np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
        ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
        temp_bhv = self.parsedData[ndx].copy()
        df_vevaio_lauglim = pd.DataFrame({'lo':abs(np.log(temp_bhv.lauglim_pLeft/(1-temp_bhv.lauglim_pLeft))),
                                          'wt':temp_bhv.WT})
        df_vevaio_lauglim.loc[:,'bin'] = np.digitize(df_vevaio_lauglim.lo,np.unique(np.percentile(df_vevaio_lauglim.lo,np.linspace(0,100,nbins+1))))
        df_vevaio_lauglim.loc[df_vevaio_lauglim.loc[:,'bin']>nbins,'bin'] = nbins
        ha[2,0].scatter(df_vevaio_lauglim.lo,df_vevaio_lauglim.wt,color='xkcd:green',label='_nolegend_',alpha=.1)
        piv_vevaio_lauglim = df_vevaio_lauglim.pivot_table(columns='bin').T
        ha[2,0].plot(piv_vevaio_lauglim.lo,piv_vevaio_lauglim.wt,color='xkcd:green',label='greedy',linewidth=lw)

        ndx = np.logical_and(np.logical_not(self.parsedData.lauglim_isGreedy),
                             np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
        ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
        temp_bhv = self.parsedData[ndx].copy()
        df_vevaio_lauglim = pd.DataFrame({'lo':abs(np.log(temp_bhv.lauglim_pLeft/(1-temp_bhv.lauglim_pLeft))),
                                          'wt':temp_bhv.WT})
        df_vevaio_lauglim.loc[:,'bin'] = np.digitize(df_vevaio_lauglim.lo,np.unique(np.percentile(df_vevaio_lauglim.lo,np.linspace(0,100,nbins+1))))
        df_vevaio_lauglim.loc[df_vevaio_lauglim.loc[:,'bin']>nbins,'bin'] = nbins
        ha[2,0].scatter(df_vevaio_lauglim.lo,df_vevaio_lauglim.wt,color='xkcd:red',label='_nolegend_',alpha=.1)
        piv_vevaio_lauglim = df_vevaio_lauglim.pivot_table(columns='bin').T
        ha[2,0].plot(piv_vevaio_lauglim.lo,piv_vevaio_lauglim.wt,color='xkcd:red',label='notGreedy',linewidth=lw)
        ha[2,0].set_xlabel(r'$ | log \left( \frac{P_{Left}}{P_{Right}} \right)|$',fontsize=fs_lab)
        ha[2,0].set_ylabel('waiting time (s)',fontsize=fs_lab)
        ha[2,0].set_title('lauglim',fontsize=fs_title)
        ha[2,0].legend(fancybox=True, framealpha=0.5)

        ## Panel H - Vevaio Ideal Observer 1 (ignores waiting time)

        ndx = np.logical_and(self.parsedData.idobs_isGreedy,
                             np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
        ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
        temp_bhv = self.parsedData[ndx].copy()
        df_vevaio_idobs = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs_pLeft/temp_bhv.idobs_pRight)),
                                          'wt':temp_bhv.WT})
        df_vevaio_idobs.loc[:,'bin'] = np.digitize(df_vevaio_idobs.lo,np.unique(np.percentile(df_vevaio_idobs.lo,np.linspace(0,100,nbins+1))))
        df_vevaio_idobs.loc[df_vevaio_idobs.loc[:,'bin']>nbins,'bin'] = nbins
        ha[2,1].scatter(df_vevaio_idobs.lo,df_vevaio_idobs.wt,color='xkcd:green',label='_nolegend_',alpha=.1)
        piv_vevaio_idobs = df_vevaio_idobs.pivot_table(columns='bin').T
        ha[2,1].plot(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:green',label='greedy',linewidth=lw)

        ndx = np.logical_and(np.logical_not(self.parsedData.idobs_isGreedy),
                             np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
        ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
        temp_bhv = self.parsedData[ndx].copy()
        df_vevaio_idobs = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs_pLeft/temp_bhv.idobs_pRight)),
                                          'wt':temp_bhv.WT})
        df_vevaio_idobs.loc[:,'bin'] = np.digitize(df_vevaio_idobs.lo,np.unique(np.percentile(df_vevaio_idobs.lo,np.linspace(0,100,nbins+1))))
        df_vevaio_idobs.loc[df_vevaio_idobs.loc[:,'bin']>nbins,'bin'] = nbins
        ha[2,1].scatter(df_vevaio_idobs.lo,df_vevaio_idobs.wt,color='xkcd:red',label='_nolegend_',alpha=.1)
        piv_vevaio_idobs = df_vevaio_idobs.pivot_table(columns='bin').T
        ha[2,1].plot(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:red',label='notGreedy',linewidth=lw)
        ha[2,1].set_xlabel(r'$ | log \left( \frac{P_{Left}}{P_{Right}} \right)|$',fontsize=fs_lab)
        ha[2,1].set_ylabel('waiting time (s)',fontsize=fs_lab)
        ha[2,1].set_title('idobs',fontsize=fs_title)
        ha[2,1].legend(fancybox=True, framealpha=0.5)
        # nbins = 4
        #
        # ndx = np.logical_and(self.parsedData.idobs_isGreedy,
        #                      np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
        # ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
        # temp_bhv = self.parsedData[ndx].copy()
        # df_vevaio_idobs = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs_pLeft/temp_bhv.idobs_pRight)),
        #                                 'wt':temp_bhv.WT})
        # df_vevaio_idobs.loc[:,'bin'] = np.digitize(df_vevaio_idobs.lo,np.unique(np.percentile(df_vevaio_idobs.lo,np.linspace(0,100,nbins+1))))
        # piv_vevaio_idobs = df_vevaio_idobs.pivot_table(columns='bin').T
        # # display(piv_vevaio_idobs)
        # ha[2,1].scatter(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:green',label='greedy')
        # ha[2,1].plot(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:green',label='_nolegend_')
        #
        # ndx = np.logical_and(np.logical_not(self.parsedData.idobs_isGreedy),
        #                      np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
        # temp_bhv = self.parsedData[ndx].copy()
        # df_vevaio_idobs = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs_pLeft/temp_bhv.idobs_pRight)),
        #                                 'wt':temp_bhv.WT})
        # df_vevaio_idobs.loc[:,'bin'] = np.digitize(df_vevaio_idobs.lo,np.unique(np.percentile(df_vevaio_idobs.lo,np.linspace(0,100,nbins+1))))
        # if np.sum(df_vevaio_idobs.bin==df_vevaio_idobs.bin.max()) < 2:
        #     df_vevaio_idobs.loc[df_vevaio_idobs.bin==df_vevaio_idobs.bin.max(),'bin']=df_vevaio_idobs.bin.max()-1
        # piv_vevaio_idobs = df_vevaio_idobs.pivot_table(columns='bin').T
        # # display(piv_vevaio_idobs)
        # ha[2,1].scatter(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:red',label='notGreedy')
        # ha[2,1].plot(piv_vevaio_idobs.lo,piv_vevaio_idobs.wt,color='xkcd:red',label='_nolegend_')
        #
        # # ha[2,1].xlabel(r'$abs \left(log \left[ \frac{P_{Left}}{P_{Right}} \right] \right)$',fontsize=20)
        # ha[2,1].set_xlabel(r'$ | log (P_{Left}) -  log (P_{Right})|$',fontsize=fs_lab)
        # ha[2,1].set_ylabel('waiting time (s)',fontsize=fs_lab)
        # ha[2,1].set_title('Ideal Observer 1',fontsize=fs_title)
        # ha[2,1].legend(fancybox=True, framealpha=0.5)

        ## Panel I - Vevaio Ideal Observer 2 (takes waiting time evidence into account)

        if 'idobs2_pLeft' in self.parsedData.columns:
            ndx = np.logical_and(self.parsedData.idobs2_isGreedy,
                                 np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
            temp_bhv = self.parsedData[ndx].copy()
            df_vevaio_idobs2 = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs2_pLeft/temp_bhv.idobs2_pRight)),
                                              'wt':temp_bhv.WT})
            df_vevaio_idobs2.loc[:,'bin'] = np.digitize(df_vevaio_idobs2.lo,np.unique(np.percentile(df_vevaio_idobs2.lo,np.linspace(0,100,nbins+1))))
            df_vevaio_idobs2.loc[df_vevaio_idobs2.loc[:,'bin']>nbins,'bin'] = nbins
            ha[2,2].scatter(df_vevaio_idobs2.lo,df_vevaio_idobs2.wt,color='xkcd:green',label='_nolegend_',alpha=.1)
            piv_vevaio_idobs2 = df_vevaio_idobs2.pivot_table(columns='bin').T
            ha[2,2].plot(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:green',label='greedy',linewidth=lw)

            ndx = np.logical_and(np.logical_not(self.parsedData.idobs2_isGreedy),
                                 np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            ndx = np.logical_and(ndx,np.logical_not(self.parsedData.isRewarded))
            temp_bhv = self.parsedData[ndx].copy()
            df_vevaio_idobs2 = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs2_pLeft/temp_bhv.idobs2_pRight)),
                                              'wt':temp_bhv.WT})
            df_vevaio_idobs2.loc[:,'bin'] = np.digitize(df_vevaio_idobs2.lo,np.unique(np.percentile(df_vevaio_idobs2.lo,np.linspace(0,100,nbins+1))))
            df_vevaio_idobs2.loc[df_vevaio_idobs2.loc[:,'bin']>nbins,'bin'] = nbins
            ha[2,2].scatter(df_vevaio_idobs2.lo,df_vevaio_idobs2.wt,color='xkcd:red',label='_nolegend_',alpha=.1)
            piv_vevaio_idobs2 = df_vevaio_idobs2.pivot_table(columns='bin').T
            ha[2,2].plot(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:red',label='notGreedy',linewidth=lw)
            ha[2,2].set_xlabel(r'$ | log \left( \frac{P_{Left}}{P_{Right}} \right)|$',fontsize=fs_lab)
            ha[2,2].set_ylabel('waiting time (s)',fontsize=fs_lab)
            ha[2,2].set_title('idobs2',fontsize=fs_title)
            ha[2,2].legend(fancybox=True, framealpha=0.5)

            # nbins = 4
            #
            # temp_bhv = self.parsedData[self.parsedData.idobs2_isGreedy].copy()
            # df_vevaio_idobs2 = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs2_pLeft/temp_bhv.idobs2_pRight)),
            #                                 'wt':temp_bhv.WT})
            # df_vevaio_idobs2.loc[:,'bin'] = np.digitize(df_vevaio_idobs2.lo,np.unique(np.percentile(df_vevaio_idobs2.lo,np.linspace(0,100,nbins+1))))
            # piv_vevaio_idobs2 = df_vevaio_idobs2.pivot_table(columns='bin').T
            # # display(piv_vevaio_idobs2)
            # ha[2,2].scatter(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:green',label='greedy')
            # ha[2,2].plot(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:green',label='_nolegend_')
            #
            # ndx = np.logical_and(np.logical_not(self.parsedData.idobs2_isGreedy),
            #                      np.logical_or(self.parsedData.isChoiceLeft,self.parsedData.isChoiceRight))
            # temp_bhv = self.parsedData[ndx].copy()
            # df_vevaio_idobs2 = pd.DataFrame({'lo':abs(np.log(temp_bhv.idobs2_pLeft/temp_bhv.idobs2_pRight)),
            #                                 'wt':temp_bhv.WT})
            # df_vevaio_idobs2.loc[:,'bin'] = np.digitize(df_vevaio_idobs2.lo,np.unique(np.percentile(df_vevaio_idobs2.lo,np.linspace(0,100,nbins+1))))
            # if np.sum(df_vevaio_idobs2.bin==df_vevaio_idobs2.bin.max()) < 2:
            #     df_vevaio_idobs2.loc[df_vevaio_idobs2.bin==df_vevaio_idobs2.bin.max(),'bin']=df_vevaio_idobs2.bin.max()-1
            # piv_vevaio_idobs2 = df_vevaio_idobs2.pivot_table(columns='bin').T
            # # display(piv_vevaio_idobs2)
            # ha[2,2].scatter(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:red',label='notGreedy')
            # ha[2,2].plot(piv_vevaio_idobs2.lo,piv_vevaio_idobs2.wt,color='xkcd:red',label='_nolegend_')
            #
            # # ha[2,2].xlabel(r'$abs \left(log \left[ \frac{P_{Left}}{P_{Right}} \right] \right)$',fontsize=20)
            # ha[2,2].set_xlabel(r'$ | log (P_{Left}) -  log (P_{Right})|$',fontsize=fs_lab)
            # ha[2,2].set_ylabel('waiting time (s)',fontsize=fs_lab)
            # ha[2,2].set_title('Ideal Observer 2',fontsize=fs_title)
            # ha[2,2].legend(fancybox=True, framealpha=0.5)

        plt.suptitle(self.fname)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
