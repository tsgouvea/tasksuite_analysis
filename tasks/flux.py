import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib as mp
import matplotlib.pyplot as plt

class parseSess:

    def __init__(self,filepath):
        if not os.path.exists(filepath):
            print('File not found: ' + bfilepath)
            raise OSError(filepath)
        mysess = sio.loadmat(filepath, squeeze_me=True)
        self.fname = os.path.split(filepath)[1]
        self.bpod = mysess['SessionData']
        try:
            self.params = pd.Series({n: np.asscalar(self.bpod['Settings'].item()['GUI'].item()[n]) for n in self.bpod['Settings'].item()['GUI'].item().dtype.names})
        except Exception as e:
            print('Couldn\'t load session parameters')

        self.parse()

    def parse(self):
        nArms = len(str(self.bpod['Settings'].item()['GUI'].item()['Ports_ABC'].item()))
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsTrialStart = self.bpod['TrialStartTimestamp'].item()
        tsTrialStart = tsTrialStart-tsTrialStart[0] if not np.isscalar(tsTrialStart) else 0

        dfSetup = pd.DataFrame({'tsSetup': [], 'arm': [], 'iTrial': []})
        dfPokes = pd.DataFrame({'tsPoke': [], 'arm': [], 'iTrial': []})
        dfRwd = pd.DataFrame({'tsRwd': [], 'arm': [], 'iTrial': [], 'n': []})

        for iTrial in range(1,len(nTrials)):
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
            stateTraj = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1]
            events = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()

            for iArm in range(nArms):
                if any(['GlobalTimer' + str(iArm+1) + '_End' in events.dtype.names]) :
                    x = tsTrialStart[iTrial] + events['GlobalTimer' + str(iArm+1) + '_End'].item()
                    if x.size == 1: x = [x]
                    dfSetup = dfSetup.append(pd.DataFrame({'tsSetup': x, 'arm': int(iArm), 'iTrial': int(iTrial)}))

                if any(['Port' + str(iArm+1) + 'In' in events.dtype.names]) :
                    x = tsTrialStart[iTrial] + events['Port' + str(iArm+1) + 'In'].item()
                    if x.size == 1: x = [x]
                    dfPokes = dfPokes.append(pd.DataFrame({'tsPoke': x, 'arm': int(iArm), 'iTrial': int(iTrial)}))

                if any(['water_' + 'ABC'[iArm] in stateTraj]) :
                    x = tsTrialStart[iTrial] + self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['water_' + 'ABC'[iArm]].item()[0]
                    n = dfRwd['n'].iloc[-1]+1 if sum(dfRwd['arm']==int(iArm))>0 and dfRwd['arm'].iloc[-1]==int(iArm) else 1
                    if x.size == 1: x = [x]
                    dfRwd = dfRwd.append(pd.DataFrame({'tsRwd': x, 'arm': int(iArm), 'iTrial': int(iTrial), 'n': n}))

        dfPokes.arm = dfPokes.arm.astype(int)
        dfSetup.arm = dfSetup.arm.astype(int)
        dfRwd.arm = dfRwd.arm.astype(int)
        dfPokes.iTrial = dfPokes.iTrial.astype(int)
        dfSetup.iTrial = dfSetup.iTrial.astype(int)
        dfRwd.iTrial = dfRwd.iTrial.astype(int)
        dfRwd.n = dfRwd.n.astype(int)
        dfSetup = dfSetup.set_index('iTrial')
        dfPokes = dfPokes.set_index('iTrial')
        dfRwd = dfRwd.set_index('iTrial')

        dfSetup = dfSetup.sort_values('tsSetup')
        dfPokes = dfPokes.sort_values('tsPoke')
        dfPokes['isSwitch'] = np.insert(dfPokes['arm'].iloc[1:].values != dfPokes['arm'].iloc[:-1].values,0, np.True_)

        dfRwd = dfRwd.sort_values('tsRwd')

        dfPokes['isRwded'] = np.full(len(dfPokes),False)

        for iTrial in dfRwd.index:
            ndx=np.isclose(dfRwd.loc[iTrial].tsRwd,dfPokes.tsPoke)
            ndx=np.logical_and(ndx,dfRwd.loc[iTrial].arm==dfPokes.arm)
            ndx=np.logical_and(ndx,dfPokes.index==iTrial)
            try:
                assert(sum(ndx)>=1)
                dfPokes.iloc[np.arange(len(ndx))[ndx][0],dfPokes.columns.get_loc('isRwded')]=True
            except:
                warnings.warn('Reward could not be assigned to poke response. Trial #%d of session %s' % (iTrial,self.fname))

        # for iRew in range(len(dfRwd)):
        #     a = abs(dfPokes['tsPoke']-dfRwd.iloc[iRew,dfRwd.columns.get_loc('tsRwd')])
        #     element, index = min(list(zip(a, range(len(a)))))
        #     dfPokes.iloc[index,dfPokes.columns.get_loc('isRwded')]=True

        # dfPokes['tinceR'] = np.nan
        # dfPokes['tinceC'] = np.nan
        #
        # dfPokes['tinceR0'] = np.nan
        # dfPokes['tinceR1'] = np.nan
        # dfPokes['tinceR2'] = np.nan
        #
        # for row in np.arange(len(dfPokes))[dfPokes['isSwitch'].values]:
        #     ndxC = np.logical_and(dfPokes['arm']==dfPokes.iloc[row,dfPokes.columns.get_loc('arm')],dfPokes['tsPoke']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
        #     dfPokes.iloc[row,dfPokes.columns.get_loc('tinceC')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfPokes['tsPoke'][ndxC].max()
        #
        #     ndxR = np.logical_and(dfRwd['arm']==dfPokes.iloc[row,dfPokes.columns.get_loc('arm')],dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
        #     dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR].max()
        #
        #     ndxR0 = np.logical_and(dfRwd['arm']==0,dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
        #     dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR0')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR0].max()
        #     ndxR1 = np.logical_and(dfRwd['arm']==1,dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
        #     dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR1')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR1].max()
        #     ndxR2 = np.logical_and(dfRwd['arm']==2,dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
        #     dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR2')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR2].max()

        self.dfSetup = dfSetup
        self.dfPokes = dfPokes
        self.dfRwd = dfRwd

    def dailyfig(self):

        dicInt = {'armNo':[],'interval':[],'ndx':[]}

        for iArm in set(self.dfRwd['arm']):
            dfRwd = self.dfRwd.loc[self.dfRwd['arm']==iArm]
            dfSetup = self.dfSetup.loc[self.dfSetup['arm']==iArm]
            ndx=dfRwd.index
            for iRew in range(len(ndx)-1):
                tsRwd=dfRwd['tsRwd'].loc[ndx[iRew]]
                dicInt['armNo'].append(iArm)
                dicInt['interval'].append(dfSetup['tsSetup'].loc[dfSetup['tsSetup']>tsRwd].min()-tsRwd)
                dicInt['ndx'].append(ndx[iRew])

        dfInt = pd.DataFrame(dicInt)

        dicPR = {'armNo':[],'tsRwd':[],'tsPoke':[],'tinceR':[], 'isRwd':[],'iTrial':[]}

        valendo = 0
        for iArm in set(self.dfRwd['arm']):
            valendo = max(valendo,self.dfRwd['tsRwd'][self.dfRwd['arm']==iArm].min())

        for iArm in set(self.dfPokes['arm']):
            ndx = np.logical_and(self.dfPokes['arm']==iArm,self.dfPokes['isSwitch'])
            ndx = np.logical_and(ndx,self.dfPokes['tsPoke']>valendo)
            dfPokes = self.dfPokes[ndx]
            dfRwd = self.dfRwd[self.dfRwd['arm']==iArm]

            for iPoke in range(len(dfPokes)):
                dicPR['armNo'].append(iArm)
                dicPR['isRwd'].append(dfPokes.iloc[iPoke].isRwded)
                dicPR['iTrial'].append(dfPokes.index[iPoke])
                tsPoke = dfPokes.iloc[iPoke].tsPoke
                tsRwd = dfRwd.tsRwd[dfRwd.tsRwd<dfPokes.iloc[iPoke].tsPoke].max()
                dicPR['tsPoke'].append(tsPoke)
                dicPR['tsRwd'].append(tsRwd)
                dicPR['tinceR'].append(tsPoke-tsRwd)

        #     break


        dfTince=pd.DataFrame(dicPR)
        dfTince=dfTince.set_index('iTrial')
        dfTince.sort_index()

        dicLing = {'armNo':[],'lingert':[],'iTrial':[],'nRew':[]}

        tsPoke = self.dfPokes['tsPoke']
        arm = self.dfPokes['arm']

        MeanABC=self.params.index[[n.startswith('Mean') for n in self.params.index]]
        ndxSwi=np.where(self.dfPokes['isSwitch'])[0]

        for i in range(len(ndxSwi)-1):
            assert(arm.iloc[ndxSwi[i+1]-1]==arm.iloc[ndxSwi[i]])
            dicLing['armNo'].append(arm.iloc[ndxSwi[i]])
            dicLing['lingert'].append(tsPoke.iloc[ndxSwi[i+1]-1]-tsPoke.iloc[ndxSwi[i]])
            dicLing['nRew'].append(self.dfPokes['isRwded'].iloc[ndxSwi[i]:ndxSwi[i+1]].astype(int).sum())
            dicLing['iTrial'].append(self.dfPokes.index[ndxSwi[i]])

        dfLing = pd.DataFrame(dicLing)
        dfLing = dfLing.set_index('iTrial')

        fracTime=np.full(3,np.nan)
        fracRew=np.full(3,np.nan)
        for iArm in set(dfInt['armNo']):
            dfLing_arm=dfLing[dfLing['armNo']==iArm]
            fracTime[iArm]=dfLing_arm['lingert'].sum()/dfLing['lingert'].sum()
            fracRew[iArm]=dfLing_arm['nRew'].sum()/dfLing['nRew'].sum()

        dfLeav=self.dfRwd[np.append(np.diff(self.dfRwd.n)<=0,False)]

        ## Panel A - Rwd-Setup Latency

        colors=('xkcd:water blue','xkcd:scarlet','xkcd:mango')#,'xkcd:grass green')
        fs_lab=12
        lw=2
        facealpha=.2
        hf, ha = plt.subplots(2,3,figsize=(10,6))
        bins = np.arange(np.percentile(dfInt['interval'],99))
        for iArm in set(dfInt['armNo']):
            dfInt_arm=dfInt[dfInt['armNo']==iArm]
            x=dfInt_arm.interval
            x=np.clip(x,0,bins.max())
            ha[0,0].hist(x,bins=bins,cumulative=False,density=False,histtype='step',color=colors[iArm],lw=lw)
            ha[0,0].hist(x,bins=bins,cumulative=False,density=False,histtype='stepfilled',alpha=facealpha,color=colors[iArm],edgecolor='None')

        ha[0,0].set_ylabel('Counts',fontsize=fs_lab)
        ha[0,0].set_xlabel('Reward availability time (s)',fontsize=fs_lab)

        ## Panel B - P($Rwd \mid $ response time)

        for iArm in set(dfTince['armNo']):
            ndx = dfTince['armNo']==iArm
            x=dfTince[ndx]['tinceR']
            y=dfTince[ndx]['isRwd']
        #     x=1-np.exp(-1*lambdas[listArmJ[0]]*x)

            ndx=np.digitize(x,np.percentile(x,np.linspace(0,100,11)))
            x = [x[ndx == i].mean() for i in range(1, len(set(ndx)))]
            y = [y[ndx == i].mean() for i in range(1, len(set(ndx)))]

            y_hat = 1-np.exp(float(self.params['Mean' + 'ABC'[iArm]])**-1*np.array(x)*-1) if self.params.VI else x > self.params['Mean' + 'ABC'[iArm]]

            ha[1,0].scatter(x,y,c=colors[iArm])
            ha[1,0].plot(x,y,c=colors[iArm])
            ha[1,0].plot(x,y_hat,c=colors[iArm],linestyle='--')

        ha[1,0].set_xlabel('Latency to respond (s)',fontsize=fs_lab)
        ha[1,0].set_ylabel('p ( reward )',fontsize=fs_lab)

        ## Panel C - visit duration histogram

        bins=np.arange(np.percentile(dfLing['lingert'],99))

        for iArm in set(dfLing['armNo']):

            dfLing_arm=dfLing[dfLing['armNo']==iArm]
            x=dfLing_arm.lingert
            x=np.clip(x,0,bins.max())

            ha[0,1].hist(x,bins=bins,cumulative=False,density=False,histtype='step',color=colors[iArm],lw=2)
            ha[0,1].hist(x,bins=bins,cumulative=False,density=False,histtype='stepfilled',alpha=facealpha,color=colors[iArm],edgecolor='None')

        ha[0,1].set_xlabel('Visit duration (s)',fontsize=fs_lab)
        ha[0,1].set_ylabel('Counts',fontsize=fs_lab)

        ## Panel D - Matching

        ha[1,1].plot([0,1],[0,1],c='xkcd:gray',lw=2, alpha=.3)

        ha[1,1].plot(np.sort(fracRew),fracTime[np.argsort(fracRew)],c='xkcd:gray',alpha=.2)

        for iArm in set(dfInt['armNo']):
            dfInt_arm=dfInt[dfInt['armNo']==iArm]
            dfLing_arm=dfLing[dfLing['armNo']==iArm]
            ha[1,1].scatter(fracRew[iArm],fracTime[iArm],c=colors[iArm])

        ha[1,1].set_xlabel('Frac rewards',fontsize=fs_lab)
        ha[1,1].set_ylabel('Frac time spent',fontsize=fs_lab)
        ha[1,1].set_xlim(-.1,1.1)
        ha[1,1].set_ylim(-.1,1.1)

        ## Panel E - Hazard rate of leaving

        for iArm in list(set(self.dfRwd.arm)):

            dfArm_all=self.dfRwd[self.dfRwd.arm==iArm]
            dfArm_lea=dfLeav[dfLeav.arm==iArm]

            n=np.unique(dfArm_all.n.values)

            y=np.full(n.shape,np.nan)

            for iRew in range(len(n)):
                y[iRew] = np.sum(dfArm_lea.n==n[iRew]) / np.sum(dfArm_all.n==n[iRew])

            ha[0,2].scatter(n,y,c=colors[iArm])
            ha[0,2].plot(n,y,c=colors[iArm],alpha=.5)

        ha[0,2].set_ylabel("Hazard rate of leaving",fontsize=fs_lab)
        ha[0,2].set_xlabel("# consecutive rewards",fontsize=fs_lab)
        ha[0,2].set_ylim(-.1,1.1)
        # ha[2,0].xlim(np.array([1.1,-.1])*self.params.rewFirst)

        ## Panel F - Cumulative reward

        for iArm in list(set(dfLeav.arm)):
            iArm=int(iArm)
            df=self.dfRwd[self.dfRwd.arm==iArm]
        #     df['tsRwd']=df.tsRwd-self.dfRwd.tsRwd.iloc[0]
            ha[1,2].plot(df.tsRwd/60,np.cumsum(np.full(df.tsRwd.values.shape,self.params['rewFirst']))/1000,c=colors[iArm])

        ha[1,2].plot(self.dfRwd.tsRwd/60,np.cumsum(np.full(self.dfRwd.tsRwd.values.shape,self.params['rewFirst']))/1000,c='xkcd:black')

        ha[1,2].set_ylabel("Total reward ($mL$)",fontsize=fs_lab)
        ha[1,2].set_xlabel("Time (min)",fontsize=fs_lab)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(self.fname.split('.')[0],fontsize=fs_lab*1.3)

    def tince(self): # creates dataframe with Time sINCE last reward/response at moment of each ('isSwitch') response
        dfTince = pd.DataFrame({'arm':self.dfPokes['arm'][self.dfPokes['isSwitch']].values, 'ts':self.dfPokes['tsPoke'][self.dfPokes['isSwitch']].values,
                     'sinceA':np.nan,'sinceB':np.nan,'sinceC':np.nan})

        tsRwdA = self.dfRwd['tsRwd'][self.dfRwd['arm']==0]
        tsRwdB = self.dfRwd['tsRwd'][self.dfRwd['arm']==1]
        tsRwdC = self.dfRwd['tsRwd'][self.dfRwd['arm']==2]

        for iArm in range(3):
            tsRwd = self.dfRwd['tsRwd'][self.dfRwd['arm']==iArm]
            for iResp in range(len(dfTince)):
                if any(tsRwd < dfTince['ts'].iloc[iResp]):
                    dfTince['since' + 'ABC'[iArm]][iResp] = dfTince['ts'].iloc[iResp] - max(tsRwd[tsRwd < dfTince['ts'].iloc[iResp]])
        self.dfTinceR = dfTince
