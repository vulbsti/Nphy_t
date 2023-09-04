# Nphy_t
EEG EYE BLINK CLASSIFIER 


## Usage
main.py taken in arguments -f for filename to work with and -fx for location where to export new annotated edf file
>Example
  ~~~
  python main.py -f eeg/test.fif -fx eeg/
  ~~~

## Metrics
Metrics are based on standard formula for accuracy, precision, F1.
Since all blinks labels havent been annotated
Precision is calculated based on no of predictions in closed eye duration. Any predictions are considered false.
Accuracy is predicted based on How many blinks have been detected that are labelled.

### Reference Resource 
[BLINK Algorithm](uhttps://par.nsf.gov/servlets/purl/10321749rl) used as reference for building a Blink algorithm based on temporal characterstics on Eye blinks


### Warnings
edf files are not supported use edftofif.py to convert edf files to fif and use them. 
MNE shows codec errors while performing certain functions on EDF
If still need to use edf use import pyedflib for reading annotations

