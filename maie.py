import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import blink
import csv
import glob
import argparse
'''
Reading fif via txt files
f=input("Enter the file name: ")
file1 = open("eeg/"+f, 'r')
Lines = file1.readlines()
'''



parser = argparse.ArgumentParser(description="take folder path to fif as input")
parser.add_argument("-fp", "--file", type=str, required=True, help="Folder Path To fif files")
args = parser.parse_args()
filepath=args.file
files=glob.glob(filepath+"/*.fif")

ls=[]
for fname in files:
    '''
    #when reading txt
    #for fname in Lines: 
    #fname=fname.replace('\n','')
    #fname=fname.replace('\n','')
    #data= mne.io.read_raw_fif("eeg/e/" +fname+ ".fif",preload=True) 
    #import pyedflib as pb    # uncomment incase of errors using EDF
    '''

 
    
    data= mne.io.read_raw_fif(fname,preload=True) 
    annotation= mne.read_annotations(fname)  #returns tuple as MNE object
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
       
    
    for i in range(2):
        onset,val=blink.compute(fil_data,chan=i)
        onset,val=blink.eliminate(onset,val)
        if len(onset)!= 0 :
            acc,accuracy,false_pre,precision,recall,F1s=blink.metrics(onset,anot_t,anot_ke)
            l=[fname,acc,accuracy,false_pre,precision,len(onset),len(anot_t),i]
            #print(f"{l[1]} ,{l[3]} ,{l[2]} ,{l[5]}, {l[6]}")
            ls.append(l)
        else :
            ls.append([0,0,0,0,0,0,0])
    
    
   
with open(filepath+"metric.csv", 'w', newline='') as file:
        writer =csv.writer(file)
        writer.writerow(['filename','accuracy_rate','accuracy','false_pre','precision','predicted_t','anot_to','chan'])
        writer.writerows(ls)
