import mne
import os
import glob


filepath=input("edf directory path: ")

files=glob.glob(filepath+"/*.edf")
new_fpath=filepath+"\\fif"
if not os.path.exists(new_fpath):
    os.mkdir(new_fpath)

print (files)
for file in files:
    raw=mne.io.read_raw_edf(file,preload=True)
    
    filen=file.replace(filepath,"")
    filen=filen.replace(".edf",".fif")
    raw.save(new_fpath+filen,overwrite=True)
