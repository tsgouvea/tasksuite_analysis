import os

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib as mp
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
            print('Couldn\'t load session parameters')

        self.parse()

    def parse(self):
        nArms = len(str(self.bpod['Settings'].item()['GUI'].item()['Ports_ABC'].item()))
        nTrials = np.arange(np.asscalar(self.bpod['nTrials']))+1
        tsTrialStart = self.bpod['TrialStartTimestamp'].item()
        tsTrialStart = tsTrialStart-tsTrialStart[0] if not np.isscalar(tsTrialStart) else 0

        dfPokes = pd.DataFrame({'tsPoke': [], 'arm': [], 'iTrial': []})
        dfRwd = pd.DataFrame({'tsRwd': [], 'arm': [], 'iTrial': [], 'n': []})

        for iTrial in range(min(5000,len(nTrials))):
            try:
                listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
                stateTraj = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1]
                events = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()
            except Exception as e:
                print(e)
#             print(stateTraj)

#             print(events.dtype.names)
            for iArm in range(nArms):
                if any(['Port' + str(iArm+1) + 'In' in events.dtype.names]) :
                    x = tsTrialStart[iTrial] + events['Port' + str(iArm+1) + 'In'].item()
                    if x.size == 1: x = [x]
                    dfPokes = dfPokes.append(pd.DataFrame({'tsPoke': x, 'arm': int(iArm), 'iTrial': int(iTrial)}))

                if any(['water_' + 'ABC'[iArm] in stateTraj]) :
                    x = tsTrialStart[iTrial] + self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['water_' + 'ABC'[iArm]].item()[0]
                    x = [x] if x.size == 1 else x
                    try:
                        n = dfRwd['n'].iloc[-1]+1 if sum(dfRwd['arm']==int(iArm))>0 and dfRwd['arm'].iloc[-1]==int(iArm) else 1
                    except:
                        display(dfRwd)
                        print(sum(dfRwd['arm']==int(iArm))>0, dfRwd['arm'].iloc[-1])
                        break
                    dfRwd = dfRwd.append(pd.DataFrame({'tsRwd': x, 'arm': int(iArm), 'iTrial': int(iTrial), 'n': n}))

        dfRwd['rwdLast']=np.nan
        dfRwd['rwdNext']=np.nan
        for iArm in list(set(dfRwd.arm)):
            iArm=int(iArm)
            indxArm=np.arange(len(dfRwd))[dfRwd.arm==iArm]
            ratio=self.params.rewLast/self.params.rewFirst
            den=self.params['rewN_' + 'ABC'[iArm]]
            dfRwd.iloc[indxArm,dfRwd.columns.get_loc('rwdNext')]=np.ceil(ratio**(dfRwd.iloc[indxArm].n.values/den)*self.params.rewFirst)
            dfRwd.iloc[indxArm,dfRwd.columns.get_loc('rwdLast')]=np.ceil(ratio**((dfRwd.iloc[indxArm].n.values-1)/den)*self.params.rewFirst)

        dfPokes.arm = dfPokes.arm.astype(int)
        dfRwd.arm = dfRwd.arm.astype(int)
        dfPokes.iTrial = dfPokes.iTrial.astype(int)
        dfRwd.iTrial = dfRwd.iTrial.astype(int)
        dfRwd.n = dfRwd.n.astype(int)
        dfRwd.rwdLast = dfRwd.rwdLast.astype(int)
        dfRwd.rwdNext = dfRwd.rwdNext.astype(int)
        dfPokes = dfPokes.set_index('iTrial')
        dfRwd = dfRwd.set_index('iTrial')

        dfPokes = dfPokes.sort_values('tsPoke')
        dfPokes['isSwitch'] = np.insert(dfPokes['arm'].iloc[1:].values != dfPokes['arm'].iloc[:-1].values,0, np.True_)

        dfRwd = dfRwd.sort_values('tsRwd')

        dfPokes['isRwded'] = np.full(len(dfPokes),False)
        tsRwdCol=dfRwd.columns.get_loc('tsRwd')
        for iRew in range(len(dfRwd)):
            a = abs(dfPokes['tsPoke']-dfRwd.iloc[iRew,tsRwdCol])
            element, index = min(list(zip(a, range(len(a)))))
            dfPokes.iloc[index,dfPokes.columns.get_loc('isRwded')]=True

        self.dfPokes = dfPokes
        self.dfRwd = dfRwd

    def dailyfig(self):
        colors=('xkcd:water blue','xkcd:scarlet','xkcd:mango')#,'xkcd:grass green')
        fs_lab=12
        lw=2
        facealpha=.2
        setN=np.array([self.params['rewN_' + 'ABC'[i]] for i in range(3)])
        dfLeav=self.dfRwd[np.append(np.diff(self.dfRwd.n)<=0,False)]
        hf, ha = plt.subplots(2,3,figsize=(10,6))

        # panel A
        for iArm in list(set(dfLeav.arm)):
            iArm=int(iArm)
            ndxArm=dfLeav.arm==iArm
            x=dfLeav[ndxArm].rwdNext.values
            ha[0,0].hist(x,bins=np.arange(self.params.rewFirst+1),cumulative=False,density=True,histtype='step',color=colors[iArm],lw=lw)
            ha[0,0].hist(x,bins=np.arange(self.params.rewFirst+1),cumulative=False,density=True,histtype='stepfilled',alpha=facealpha,color=colors[iArm],lw=lw)
        # for iArm in list(set(dfLeav.arm)):
        #     iArm=int(iArm)
        #     ha[0,0].hist(dfLeav[dfLeav.arm==iArm].rwdNext,bins=np.linspace(0,self.params.rewFirst,11),cumulative=False,density=True,histtype='stepfilled',alpha=facealpha,color=colors[int(iArm)],lw=lw)
        #     ha[0,0].hist(dfLeav[dfLeav.arm==iArm].rwdNext,bins=np.linspace(0,self.params.rewFirst,11),cumulative=False,density=True,histtype='step',color=colors[int(iArm)],lw=lw)
        ha[0,0].plot(np.full(2,1)*self.dfRwd.rwdLast.sum()/(self.dfRwd.tsRwd.values[-1]-self.dfRwd.tsRwd.values[0])*self.params.IRI,np.array([0, 0.4]),linestyle='--',alpha=.7,color='xkcd:black')
        ha[0,0].set_ylabel("$P(x \mid$ leave $)$",fontsize=fs_lab)
        ha[0,0].set_xlabel("$x = $ next reward ($\mu L$)",fontsize=fs_lab)
        ha[0,0].set_xlim(np.array([1.1,-.1])*self.params.rewFirst)

        # panel B
        for iArm in list(set(self.dfRwd.arm)):

            dfArm_all=self.dfRwd[self.dfRwd.arm==iArm]
            dfArm_lea=dfLeav[dfLeav.arm==iArm]

            set_next=np.unique(dfArm_all.rwdNext.values)

            y=np.full(set_next.shape,np.nan)

            for iRew in range(len(set_next)):
                y[iRew] = np.sum(dfArm_lea.rwdNext==set_next[iRew]) / np.sum(dfArm_all.rwdNext==set_next[iRew])

            ha[1,0].scatter(set_next,y,c=colors[iArm])
            ha[1,0].plot(set_next,y,c=colors[iArm],alpha=.5)
        ha[1,0].plot(np.full(2,1)*self.dfRwd.rwdLast.sum()/(self.dfRwd.tsRwd.values[-1]-self.dfRwd.tsRwd.values[0])*self.params.IRI,np.array([0, 1]),linestyle='--',alpha=.7,color='xkcd:black')
        ha[1,0].set_ylabel("Hazard rate of leaving",fontsize=fs_lab)
        ha[1,0].set_xlabel("Next reward ($\mu L$)",fontsize=fs_lab)
        ha[1,0].set_xlim(np.array([1.1,-.1])*self.params.rewFirst)

        # panel C
        for iArm in list(set(dfLeav.arm)):
            iArm=int(iArm)
            ndxArm=dfLeav.arm==iArm
            x=dfLeav[ndxArm].n.values
            x=x.clip(1,setN.max()*1.5)
            ha[0,1].hist(x,bins=np.arange(setN.max()*1.5+1),cumulative=False,density=True,histtype='step',color=colors[iArm],lw=lw)
            ha[0,1].hist(x,bins=np.arange(setN.max()*1.5+1),cumulative=False,density=True,histtype='stepfilled',alpha=facealpha,color=colors[iArm],lw=lw)
        # ha[0,1].set_ylabel("P ( Leaving )",fontsize=fs_lab)
        ha[0,1].set_xlabel("$x = $ # consecutive rewards",fontsize=fs_lab)
        ha[0,1].set_xlim(np.array([-.1,1.])*setN.max()*1.5)

        # panel D
        listTrials=np.random.permutation(dfLeav.index.values)[0:min(500,len(dfLeav))]
        for iTrial in listTrials:
            iArm=int(self.dfRwd.loc[iTrial].arm)
            ndx=np.logical_and(self.dfRwd.arm==iArm,self.dfRwd.n==1)
            ndx=np.logical_and(ndx,self.dfRwd.index.values<=iTrial)
            jTrial=self.dfRwd.index.values[ndx].max()
            x=np.arange(iTrial-jTrial+1)
            y=self.dfRwd.loc[jTrial:iTrial].rwdNext
            y=y+np.random.rand(1)
            ha[1,1].scatter(x,y,c=colors[iArm],alpha=.5)
            ha[1,1].plot(x,y,c='xkcd:black',alpha=.05)
        ha[1,1].plot(np.array([0,setN.max()]),np.full(2,1)*self.dfRwd.rwdLast.sum()/(self.dfRwd.tsRwd.values[-1]-self.dfRwd.tsRwd.values[0])*self.params.IRI,linestyle='--',alpha=.7,color='xkcd:black')
        ha[1,1].set_ylabel("Next reward ($\mu L$)",fontsize=fs_lab)
        ha[1,1].set_xlabel("# consecutive rewards",fontsize=fs_lab)
        ha[1,1].set_xlim(np.array([-.1,1.])*setN.max()*1.5)

        # panel E, F
        for iArm in list(set(dfLeav.arm)):
            iArm=int(iArm)
            df=self.dfRwd[self.dfRwd.arm==iArm]
            df['tsRwd']=df.tsRwd-self.dfRwd.tsRwd.iloc[0]
            ha[0,2].plot(df.tsRwd/60,np.cumsum(df.rwdLast)/1000,c=colors[iArm],lw=lw)
            ha[1,2].scatter(df.tsRwd/60,df.rwdLast,c=colors[iArm],alpha=.25)
        ha[0,2].plot(self.dfRwd.tsRwd/60,np.cumsum(self.dfRwd.rwdLast)/1000,c='xkcd:black',lw=lw)
        ha[0,2].set_ylabel("Total reward ($mL$)",fontsize=fs_lab)
        ha[0,2].set_xlabel("Time (min)",fontsize=fs_lab)
        ha[1,2].plot(np.array([0,self.dfRwd.tsRwd.values[-1]-self.dfRwd.tsRwd.values[0]])/60,np.full(2,1)*self.dfRwd.rwdLast.sum()/(self.dfRwd.tsRwd.values[-1]-self.dfRwd.tsRwd.values[0])*self.params.IRI,linestyle='--',alpha=.7,color='xkcd:black')
        ha[1,2].set_ylabel("Last reward ($\mu L$)",fontsize=fs_lab)
        ha[1,2].set_xlabel("Time (min)",fontsize=fs_lab)

        # panel F
        # ha[1,2].set_axis_off()
        # for iArm in list(set(dfLeav.arm)):
        #     iArm=int(iArm)
        #     df=self.dfRwd[self.dfRwd.arm==iArm]
        #     df['tsRwd']=df.tsRwd-self.dfRwd.tsRwd.iloc[0]
        #     ha[1,2].scatter(df.tsRwd/60,df.n,c=colors[iArm],alpha=.1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(self.fname.split('.')[0],fontsize=fs_lab*1.3)
