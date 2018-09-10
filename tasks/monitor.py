
# coding: utf-8

# In[12]:


import os
import re

import numpy as np
import tasks as suite
import matplotlib.pyplot as plt

# path=os.path.join(os.sep,'Users','thiago','Programs','BpodUser','Data')
# path='C:\Users\Thiago\BpodUser\Data'
path = os.path.join('C:', os.sep, 'Users','Thiago','BpodUser','Data')

dictasks = {'Flux':'flux','FluxSeries':'fluxseries'}

for key in dictasks.keys():
    print('import tasks.%s' % dictasks[key])
    exec('import tasks.%s' % dictasks[key])

latest={'dirpath':[],'filename':[],'mtime':0}
for (dirpath, dirnames, filenames) in os.walk(path):
    ndxFile1=np.array([n.endswith('.mat') for n in filenames])
    if not ndxFile1.any():
        continue
    ndxFile2=np.full(ndxFile1.shape,False)
    for key in dictasks.keys():
        ndxFile2=np.logical_or(ndxFile2,np.array([key in n for n in filenames]))
    ndxFile=np.logical_and(ndxFile1,ndxFile2)
    if ndxFile.any():
        for filename in filenames:
            mtime=os.path.getmtime(os.path.join(dirpath,filename))
            if mtime > latest['mtime']:
                latest['mtime']=mtime
                latest['dirpath']=dirpath
                latest['filename']=filename

while True:
    for (dirpath, dirnames, filenames) in os.walk(path):
        ndxFile1=np.array([n.endswith('.mat') for n in filenames])
        if not ndxFile1.any():
            continue
        ndxFile2=np.full(ndxFile1.shape,False)
        for key in dictasks.keys():
            ndxFile2=np.logical_or(ndxFile2,np.array([key in n for n in filenames]))
        ndxFile=np.logical_and(ndxFile1,ndxFile2)
        if ndxFile.any():
            for filename in filenames:
                mtime=os.path.getmtime(os.path.join(dirpath,filename))
                if mtime >= latest['mtime']:
                    latest['mtime']=mtime
                    latest['dirpath']=dirpath
                    latest['filename']=filename

                    path = os.path.join(latest['dirpath'],latest['filename'])

                    for key in dictasks.keys():
                        try:
                            s='f=suite.%s.parseSess' % dictasks[key]
                            exec(s)
                            sess=f(path)
                            sess.dailyfig()
                        except Exception as e:
                            print(e)
