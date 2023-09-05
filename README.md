# Nphy_t
EEG EYE BLINK CLASSIFIER 


## Usage
main.py taken in arguments -f for filename to work with and -fx for location where to export new annotated edf file

~~~
pip install -r requirements.txt
~~~
>Example
  ~~~
  python main.py -f eeg/test.fif -fx eeg/
  ~~~
For Batch-operation and metric calculation use mnbat.py. The module reads all fif files in mentioned directory. And predicted values are stored in a metrics.csv file in the same directory. with chan 0 representing Fp1 and chan 1 representing Fp2. 
  ~~~
  python main_batch.py -fp "fif directory path"
  ~~~

## Metrics
Metrics are based on standard formula for accuracy, precision, F1.
Since all blinks labels havent been annotated
Precision is calculated based on no of predictions in closed eye duration. Any predictions are considered false.
Accuracy is predicted based on How many blinks have been detected that are labelled.
  #### An F1 score is calculated based on these Values
>  Accuracy = (Total Blink labels found in duration of predicted blink label)/Total Blink Labels Annotated

>  Precision = (Total Predictions - Predictions Made in Closed Eye Interval)/Total Predictions

>  Recall =  (Total Predictions that can be counted as Accurate)/(Accurate Predictions + False Predictions)

  

### Reference Resource 
[BLINK Algorithm](https://par.nsf.gov/servlets/purl/10321749) used as reference for building a detecting eye_blinks based on temporal characterstics on Eye blinks
This depends on a single channel data. Fp1 and Fp2 both are great channels for detecting eye_blinks. However Fp1 showed a greater accuracy hence it was selected as preffered channel 
Although using relationship matrix for potential blinks in both channels can provide more refined selection of blink duration.


### Warnings
edf files are not supported use edfTofif.py to convert edf files to fif and use them. 
>  edfTofif detects all edf files in a directory and saves them in fif format in a sub-directory called fif
MNE shows codec errors while performing certain functions on EDF
If still need to use edf use import pyedflib for reading annotations

> Incase of batch_files if metrics.csv is not present in directory add one
