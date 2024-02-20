from sklearn.mixture import GaussianMixture
from pickle import FALSE
import preprocess as pp
import pandas as pd
import numpy as np
import re
import vbhmm
import scipy.special as spg
import os
import pickle
import vhem
import plotss as pt

import numpy as np
import matplotlib.pyplot as plt
hmms= []

#for i in range(11,56):
 #   if i in [6,7,8,9,10,22,34,35,36,37,38,39,40,41]:
  #      continue
   # filepath = os.path.join('Dataset/',f"Look{str(i).zfill(3)}.asc")
    #scanpath = pp.preprocess_file(filepath)
    #print(len(scanpath))
    #hmm = vbhmm.learn(scanpath[:500], 3)
    #with open(f'HmmP{str(i).zfill(3)}.pkl','wb') as file:
     #   pickle.dump(hmm,file)
    #hmms.append(hmm)
for i in range(1,50):
    if i in [6,7,8,9,10,22,34,35,36,37,38,39,40,41]:
        continue
    with open(f'HmmP{str(i).zfill(3)}.pkl','rb') as file:
        hmm=pickle.load(file)
    hmms.append(hmm)
img = plt.imread('img_test.jpeg')
group = vhem.vhem_cluster(hmms,3)
with open('group_hmm11.pkl', 'wb') as file:
    pickle.dump(group,file)

#print only groups
for i in range(3):
    pt.hmm_plot(group['hmms'][i],'g',i)

for i in range(3):
    pt.hmm_plot(group['hmms'][i],'g',i)
    for j in group['groups'][i]:
        pt.hmm_plot(hmms[j],'f',j)
