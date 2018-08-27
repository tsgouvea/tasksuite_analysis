import os

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

        # dfSetup = pd.DataFrame({'tsSetup': [], 'arm': [], 'iTrial': []})
        dfPokes = pd.DataFrame({'tsPoke': [], 'arm': [], 'iTrial': []})
        dfRwd = pd.DataFrame({'tsRwd': [], 'arm': [], 'iTrial': []})
        dfTrials = pd.DataFrame({'arm': [], 'iTrial': [], 'latency': []})

        for iTrial in range(len(nTrials)):
            listStates = self.bpod['RawData'].item()['OriginalStateNamesByNumber'].item()[iTrial]
            stateTraj = listStates[self.bpod['RawData'].item()['OriginalStateData'].item()[iTrial]-1]
            events = self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['Events'].item()
            setup=list('setup000')

            for iArm in range(nArms):

                if any(['Port' + str(iArm+1) + 'In' in events.dtype.names]) :
                    x = tsTrialStart[iTrial] + events['Port' + str(iArm+1) + 'In'].item()
                    if x.size == 1: x = [x]
                    dfPokes = dfPokes.append(pd.DataFrame({'tsPoke': x, 'arm': int(iArm), 'iTrial': int(iTrial)}))

                # if any(['GlobalTimer' + str(iArm+1) + '_End' in events.dtype.names]) :
    #                 x = tsTrialStart[iTrial] + events['GlobalTimer' + str(iArm+1) + '_End'].item()
    #                 if x.size == 1: x = [x]
    #                 dfSetup = dfSetup.append(pd.DataFrame({'tsSetup': x, 'arm': int(iArm), 'iTrial': int(iTrial)}))
    #
                if any(['water_' + 'ABC'[iArm] in stateTraj]) :
                    x = tsTrialStart[iTrial] + self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()['water_' + 'ABC'[iArm]].item()[0]
                    if x.size == 1: x = [x]
                    dfRwd = dfRwd.append(pd.DataFrame({'tsRwd': x, 'arm': int(iArm), 'iTrial': int(iTrial)}))

                    setup[5+iArm]='1'
                    y=np.diff(self.bpod['RawEvents'].item()['Trial'].item()[iTrial]['States'].item()["".join(setup)].item()).item()
                    dfTrials = dfTrials.append(pd.DataFrame({'arm': int(iArm), 'iTrial': int(iTrial), 'latency': [y]}))
    #
        dfPokes.arm = dfPokes.arm.astype(int)
        dfPokes.iTrial = dfPokes.iTrial.astype(int)
        dfPokes = dfPokes.set_index('iTrial')
        dfPokes = dfPokes.sort_values('tsPoke')
        dfPokes['isSwitch'] = np.insert(dfPokes['arm'].iloc[1:].values != dfPokes['arm'].iloc[:-1].values,0, np.True_)

        dfRwd.arm = dfRwd.arm.astype(int)
        dfRwd.iTrial = dfRwd.iTrial.astype(int)
        dfRwd = dfRwd.set_index('iTrial')
        dfRwd = dfRwd.sort_values('tsRwd')

        dfTrials.arm = dfTrials.arm.astype(int)
        dfTrials.iTrial = dfTrials.iTrial.astype(int)
        dfTrials = dfTrials.set_index('iTrial')

    #     dfSetup.arm = dfSetup.arm.astype(int)
    #     dfSetup.iTrial = dfSetup.iTrial.astype(int)
    #     dfSetup = dfSetup.set_index('iTrial')
    #
    #     dfSetup = dfSetup.sort_values('tsSetup')

        dfPokes['isRwded'] = np.full(len(dfPokes),False)
        for iRew in range(len(dfRwd)):
            a = abs(dfPokes['tsPoke']-dfRwd.iloc[iRew,dfRwd.columns.get_loc('tsRwd')])
            element, index = min(list(zip(a, range(len(a)))))
            dfPokes.iloc[index,dfPokes.columns.get_loc('isRwded')]=True
    #
    #     dfPokes['tinceR'] = np.nan
    #     dfPokes['tinceC'] = np.nan
    #
    #     dfPokes['tinceR0'] = np.nan
    #     dfPokes['tinceR1'] = np.nan
    #     dfPokes['tinceR2'] = np.nan
    #
    #     for row in np.arange(len(dfPokes))[dfPokes['isSwitch'].values]:
    #         ndxC = np.logical_and(dfPokes['arm']==dfPokes.iloc[row,dfPokes.columns.get_loc('arm')],dfPokes['tsPoke']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
    #         dfPokes.iloc[row,dfPokes.columns.get_loc('tinceC')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfPokes['tsPoke'][ndxC].max()
    #
    #         ndxR = np.logical_and(dfRwd['arm']==dfPokes.iloc[row,dfPokes.columns.get_loc('arm')],dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
    #         dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR].max()
    #
    #         ndxR0 = np.logical_and(dfRwd['arm']==0,dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
    #         dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR0')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR0].max()
    #         ndxR1 = np.logical_and(dfRwd['arm']==1,dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
    #         dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR1')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR1].max()
    #         ndxR2 = np.logical_and(dfRwd['arm']==2,dfRwd['tsRwd']<dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')])
    #         dfPokes.iloc[row,dfPokes.columns.get_loc('tinceR2')] =  dfPokes.iloc[row,dfPokes.columns.get_loc('tsPoke')] - dfRwd['tsRwd'][ndxR2].max()
    #
    #     self.dfSetup = dfSetup
        self.dfPokes = dfPokes
        self.dfRwd = dfRwd
        self.dfTrials = dfTrials
    #
    # def dailyfig(self):
    #
    #     his = [[[] for i in range(3)] for j in range(3)]
    #     hisR = [[[] for i in range(3)] for j in range(3)]
    #
    #     mp.rc('xtick', labelsize=15)
    #     mp.rc('ytick', labelsize=15)
    #     colors=('xkcd:water blue','xkcd:scarlet','xkcd:mango')
    #
    #     hf, ha = plt.subplots(3,4,figsize=(9,6),frameon=False)
    #     for iArm in range(3):
    #         ndxi = self.dfRwd['arm']==iArm
    #         for jArm in range(3):
    #             ndxj = np.logical_and(self.dfPokes['arm'].values==jArm,self.dfPokes['isSwitch'])
    #             ndxjR = np.logical_and(ndxj,self.dfPokes['isRwded'])
    #             for iRwd in self.dfRwd['tsRwd'][ndxi]:
    #                 his[iArm][jArm] = np.hstack((his[iArm][jArm],self.dfPokes['tsPoke'][ndxj]-iRwd))
    #                 hisR[iArm][jArm] = np.hstack((hisR[iArm][jArm],self.dfPokes['tsPoke'][ndxjR]-iRwd))
    #             ha[jArm][iArm].hist(hisR[iArm][jArm],range=(1,2*self.params['Int' + 'ABC'[iArm]]), bins=20, color=colors[jArm], density=False)
    #             ha[jArm][iArm].hist(his[iArm][jArm],range=(1,2*self.params['Int' + 'ABC'[iArm]]), bins=20, histtype='step', color='xkcd:black', density=False)
    #             if jArm == 2:
    #                 ha[jArm][iArm].set_xlabel('Time (s) from reward @' + 'ABC'[iArm], fontsize=7)
    #             if iArm == 0:
    #                 ha[jArm][iArm].set_ylabel('# responses @' + 'ABC'[jArm], fontsize=7)
    #
    #     for jArm in range(3):
    #         ndxj = np.logical_and(self.dfPokes['arm'].values==jArm,self.dfPokes['isSwitch'])
    #         ha[0][3].plot(self.dfPokes['tsPoke'][ndxj].values/60.,np.cumsum(ndxj[ndxj].values),color=colors[jArm])
    #         ha[0][3].set_xlabel('Time (min)', fontsize=7)
    #         ha[0][3].set_ylabel('Cumsum responses', fontsize=7)
    #     plt.suptitle(self.fname)
    #     plt.tight_layout()
    #
    # def tince(self): # creates dataframe with Time sINCE last reward/response at moment of each ('isSwitch') response
    #     dfTince = pd.DataFrame({'arm':self.dfPokes['arm'][self.dfPokes['isSwitch']].values, 'ts':self.dfPokes['tsPoke'][self.dfPokes['isSwitch']].values,
    #                  'sinceA':np.nan,'sinceB':np.nan,'sinceC':np.nan})
    #
    #     tsRwdA = self.dfRwd['tsRwd'][self.dfRwd['arm']==0]
    #     tsRwdB = self.dfRwd['tsRwd'][self.dfRwd['arm']==1]
    #     tsRwdC = self.dfRwd['tsRwd'][self.dfRwd['arm']==2]
    #
    #     for iArm in range(3):
    #         tsRwd = self.dfRwd['tsRwd'][self.dfRwd['arm']==iArm]
    #         for iResp in range(len(dfTince)):
    #             if any(tsRwd < dfTince['ts'].iloc[iResp]):
    #                 dfTince['since' + 'ABC'[iArm]][iResp] = dfTince['ts'].iloc[iResp] - max(tsRwd[tsRwd < dfTince['ts'].iloc[iResp]])
    #     self.dfTinceR = dfTince
