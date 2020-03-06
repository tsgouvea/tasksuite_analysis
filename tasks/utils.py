import pickle, os
import numpy as np
import pandas as pd

def tidy_trials(fun,datadir=None,fnameout='df_tidy_trials.pickle',cols=None,dirout=None,forcerun=False):
    datadir = os.getcwd() if datadir is None else datadir
    dirout = datadir if dirout is None else dirout
    fnameout = fnameout + '.pickle' if not fnameout.endswith('.pickle') else fnameout
    if os.path.isfile(os.path.join(dirout,fnameout)) and not forcerun:
        with open(os.path.join(dirout,fnameout),'rb') as fhandle:
            df_tidy_trials = pickle.load(fhandle)
        print('Reading dataframe from file.')
        return df_tidy_trials
    print('Computing dataframe from scratch.')
    for root, dirs, files in os.walk(datadir):
        dirs.sort(reverse=False)
        files.sort(reverse=False)
        ndxFiles = np.array([n.endswith('.mat') for n in files])
        if any(ndxFiles):
            for file in np.array(files)[ndxFiles]:
                try:
                    bhv = fun.parseSess(os.path.join(root,file))
                    temp = bhv.parsedData.copy() if cols is None else bhv.parsedData.loc[:,cols].copy()
                    temp.loc[:,'subj'] = bhv.bpod['Custom'].item()['Subject'].item()
                    temp.loc[:,'sess'] = bhv.fname
                    df_tidy_trials = pd.concat((df_tidy_trials,temp)) if 'df_tidy_trials' in locals() else temp
                except Exception as e:
                    print(file)
                    print(e)
    with open(os.path.join(dirout,fnameout),'wb') as fhandle:
        pickle.dump(df_tidy_trials,fhandle)
    return df_tidy_trials
