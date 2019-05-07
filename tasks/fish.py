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

        self.psypy = pd.read_csv(filepath)
        if 'delay_skip.rt' not in self.psypy.columns:
            self.psypy.loc[:,'delay_skip.rt'] = np.nan
        self.fname = os.path.split(filepath)[1]

        # self.bpod = mysess['SessionData']
        # try:
        #     self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        # except Exception as e:
        #     print(e)
        #     self.params = pd.Series()
        #     warnings.warn('Could not load settings.')
        self.params = pd.Series()
        self.params.loc['pHi'] = max(self.psypy.pL.max(),self.psypy.pR.max())
        self.params.loc['pLo'] = min(self.psypy.pL.min(),self.psypy.pR.min())
        self.params.loc['FeedbackDelayMin'] = self.psypy.loc[:,'delayDurMin'].drop_duplicates()
        self.params.loc['FeedbackDelayMax'] = self.psypy.loc[:,'delayDurMax'].drop_duplicates()
        self.params.loc['FeedbackDelayTau'] = self.psypy.loc[:,'delayDurMean'].drop_duplicates()

        self.parse()
    #
    def parse(self):
        nTrials = self.psypy.shape[0]
        tsState0 = self.psypy.loc[:,'tsState0']
        ChoiceLeft = self.psypy.loc[:,'choice.keys'] == 'left'
        ChoiceRight = self.psypy.loc[:,'choice.keys'] == 'right'
        ChoiceMiss = np.logical_not(np.logical_or(ChoiceLeft,ChoiceRight))
        Rewarded = self.psypy.loc[:,'rewarded'] == True
        FixBroke = np.full(nTrials,False)
        EarlyWithdrawal = self.psypy.loc[:,'earlySout'] == True
        waitingTime = np.nansum(self.psypy.loc[:,['delayDur','stillSampl_skip.rt']].values,axis=1)
        ndxNan = np.logical_not(np.isnan(self.psypy.loc[:,'delay_skip.rt']))
        waitingTime[ndxNan] = self.psypy.loc[ndxNan,'delay_skip.rt']
        stimDelay = np.full(nTrials,np.nan)
        feedbackDelay = np.full(nTrials,np.nan)
        assert(not np.isscalar(tsState0)), "Session is only 1 trial long. Aborting."
        assert(len(tsState0) > 20), "Session is only {} trials long. Aborting.".format(len(tsState0))
        tsState0 = tsState0 - tsState0[0]

        isChoiceBaited = np.logical_or(np.logical_and(ChoiceLeft,self.psypy.loc[:,'baitL']),
                                       np.logical_and(ChoiceRight,self.psypy.loc[:,'baitR']),
                                      )
        assert(isChoiceBaited[Rewarded].all()), "Impossible trials found: unbaited AND rewarded."
        # waitingTime[isChoiceBaited] = np.nan

        self.parsedData = pd.DataFrame({'iTrial': np.arange(nTrials),
                                        'isChoiceLeft': ChoiceLeft, 'isChoiceRight': ChoiceRight, 'isChoiceMiss': ChoiceMiss,
                                        'isRewarded': Rewarded, 'isBrokeFix': FixBroke, 'isEarlyWithdr': EarlyWithdrawal, 'isChoiceBaited':isChoiceBaited,
                                        'WT': waitingTime, 'StimDelay': stimDelay, 'FeedbackDelay': feedbackDelay,'tsState0': tsState0})

        self.parsedData = self.parsedData.set_index('iTrial')

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
        leftHi=self.psypy.leftHi.values
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
        pHi=self.params.pHi/100.
        pLo=self.params.pLo/100.
        leftHi=self.psypy.leftHi.values
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
                wt = self.parsedData.WT[iTrial-1]
                p_wt_given_r = (np.exp(-wt_lambda*wt) - np.exp(-wt_lambda*wt_max))/(np.exp(-wt_lambda*wt_min) - np.exp(-wt_lambda*wt_max)) # Probability of waiting wt(s) before reward delivery, given that it was a rewarded trial
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

        feedbackDelay = self.parsedData.WT.copy()
        feedbackDelay.loc[self.parsedData.isChoiceBaited] = np.nan
        bins = np.histogram_bin_edges(feedbackDelay.dropna(),bins='sturges')
        ha[0,2].hist(feedbackDelay.loc[np.logical_not(self.parsedData.isEarlyWithdr)],color='xkcd:blue',bins=bins,alpha=facealpha,label='valid')
        ha[0,2].hist(feedbackDelay.loc[self.parsedData.isEarlyWithdr],color='xkcd:red',bins=bins,alpha=facealpha,label='earlyWithdr')
        ha[0,2].legend(fancybox=True, framealpha=0.5)
        ha[0,2].set_xlabel('FeedbackDelay',fontsize=fs_lab)

        ## Panel D - Matching

        df_match = pd.DataFrame({'iBlock':np.cumsum(np.hstack((False,np.diff(self.psypy.pL)!=0))),
                         'isChoiceLeft':self.parsedData.isChoiceLeft,
                         'isChoiceRight':self.parsedData.isChoiceRight,
                         'isRwdLeft':np.logical_and(self.parsedData.isRewarded,self.parsedData.isChoiceLeft),
                         'isRwdRight':np.logical_and(self.parsedData.isRewarded,self.parsedData.isChoiceRight),
                         # 'lo':np.log(self.parsedData.idobs2_pLeft/self.parsedData.idobs2_pRight),
                         'isLeftHi':self.psypy.loc[:,'leftHi']
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
                                                  'wt':temp_bhv.WT})
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
                                                  'wt':temp_bhv.WT})
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
                                                  'wt':temp_bhv.WT})
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
                                                  'wt':temp_bhv.WT})
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
                                                  'wt':temp_bhv.WT})
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
                                                  'wt':temp_bhv.WT})
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
