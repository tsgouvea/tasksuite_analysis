import os
import re

import numpy as np
import tasks as suite
import matplotlib.pyplot as plt

dictasks = {'Flux':'flux','FluxSeries':'fluxseries'}

for key in dictasks.keys():
    print('import tasks.%s' % dictasks[key])
    exec('import tasks.%s' % dictasks[key])

class spec:
    def __init__(self):
        mypath=os.path.expanduser("~")
        for (dirpath, dirnames, filenames) in os.walk(mypath):
            if dirpath.count(os.sep) - mypath.count(os.sep) > 2:
                continue
            if 'BpodUser' in dirnames:
                break
        self.path = os.path.join(dirpath,'BpodUser','Data')
        self.latest={'dirpath':[],'filename':[],'mtime':0,'task':'none','changed':False}
        self.dictasks = dictasks

    def update(self):
        self.latest['changed']=False
        for (dirpath, dirnames, filenames) in os.walk(self.path):
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
                    if mtime > self.latest['mtime']:
                        print('updating')
                        self.latest['mtime']=mtime
                        self.latest['dirpath']=dirpath
                        self.latest['filename']=filename
                        self.latest['task']=np.array(list(dictasks.keys()))[[n in filename for n in dictasks.keys()]].item()
                        self.latest['changed']=True

    def plot(self):
        s = 'suite.{}.parseSess("{}")'.format(self.dictasks[self.latest['task']],os.path.join(self.latest['dirpath'],self.latest['filename']))
        self.sess = eval(s)
        try:
            self.sess.dailyfig()
        except Exception as e:
            print(e)
