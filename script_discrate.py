import os

import scipy.io as sio
import matplotlib.pyplot as plt

from tasks import parser
from tasks import dailyfig


listSess = ['AZ025_Discrate_Feb16_2018_Session1','M11_Discrate_Feb06_2018_Session1','TG021_Discrate_Feb10_2018_Session1']
fig = []*len(listSess)

for iSess in range(len(listSess)) :

    mysess = sio.loadmat(os.path.join('.','examples','discrate',listSess[iSess] + '.mat'), squeeze_me=True)

    sessData = parser.discrate(mysess)

    dailyfig.discrate(sessData)

    plt.savefig(listSess[iSess] + '.eps')
#print(sessData.parsedData)

#data = sessData.parsedData
