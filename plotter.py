import mne
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="take folder path to fif as input")
parser.add_argument("-f", "--file", type=str, required=True, help="Folder Path To fif files")

args = parser.parse_args()
fname=args.file

raw=mne.io.read_raw_edf(fname, preload=True)

raw.plot()
plt.show()
