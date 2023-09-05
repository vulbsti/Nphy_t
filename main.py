import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import blink
import argparse

#import pyedflib as pb    # uncomment incase of errors using EDF

'''
# when using EDF lib
data= mne.io.read_raw_edf("eeg/e/8-0.5_edited.edf",preload=True)
    OR
data= pb.EdfReader("eeg/e/8-0.5_edited.edf")   
annotation=pb.EdfReader.readAnnotations("eeg/e/8-0.5_edited.edf") #MNE shows codec error while reading Annotations
'''

parser = argparse.ArgumentParser(description="take folder path to fif as input")
parser.add_argument("-f", "--file", type=str, required=True, help="Folder Path To fif files")
parser.add_argument("-fx", "--filexp", type=str, required=True, help="Folder Path To fif files")
args = parser.parse_args()
fname=args.file
file_exp=args.filexp



data= mne.io.read_raw_fif(fname,preload=True) 
annotation= mne.read_annotations(+fname)  #returns tuple as MNE object
anot=annotation.to_data_frame()
dt=data.info['meas_date'].time()
ot=blink.time2sec(dt)
anotti= list(anot[:]['onset'])
anot_d=list((anot[:]['duration']))
anot_ke=list((anot[:]['description']))
anot_t=[]
for i in anotti:
    dt = i.time()
    dts=blink.time2sec(dt)
    dts=dts-ot
    anot_t.append(dts)

labels=data.ch_names
sfreq=250
time_step=1/sfreq
dr=data.get_data()



fil_data=blink.filteration(dr)


raw=mne.io.RawArray(fil_data,info=data.info)
raw.set_annotations(annotation)

'''
arguments p_blinks_t and p_blinks_val are for when a correlation is to used in a Supervised System
both variables must be numpy array of shape (n,3) where each coloum represents onset, peak and settlement time and corresponding 
amplitude values. 
A default value of zero is set to compute in an Unsupervised Fashion 

'''
onset,val = blink.compute(fil_data,chan=0)

for i in range(len(onset)): 
    if onset[i,2]-onset[i,0] > 0.1 and onset[i,2]-onset[i,0] < 0.6 :
        raw.annotations.append(onset[i,0],onset[i,2]-onset[i,0],description='BLNK_P')


mne.export.export_raw(file_exp,raw,fmt='edf',physical_range='auto',add_ch_type=False, overwrite=True)


'''  
In case of accuracy measurement for single file  
ls=[]
for i in range(2):
    onset,val=blink.compute(fil_data,chan=i)
    #onset,val=blink.eliminate(onset,val)
    if len(onset)!= 0:
        acc,accuracy,false_pre,precision,recall,F1s=blink.metrics(onset,anot_t,anot_ke)
        l=[accuracy,false_pre,precision,recall,F1s]
        print(acc,false_pre)
        ls.append(l)
    else :
        ls.append([0,0,0,0])
    print(len(onset))
'''                 




