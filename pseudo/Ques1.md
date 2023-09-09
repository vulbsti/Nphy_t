# Algorithm for Blink Detection

Step 1 
 * Input the data using MNE methods and extract the dataset in a numpy format with a (n, n_chan) shape
Step 2 
  * Pass the data in the form of numpy array in a filteration method that returns data with same shape.
   Filteration method applies a notch filter of 50Hz and a bandpass butterworth 6th order filter of 1-50Hz

Step 3 
   * pass the data samples for a selected channel in a peak_det function with inputs as time  of the occurence (index of data / samplefreq) and value of data at the index.
    A for loop with range(dataset_len) is used to find local crests and troughs crossing a certain threshold in entire channel length.

Step 4 
    * The array of local minimas with their time_indexes is passed to a stable_points function along with the selected channel data and an array contaning standard deviations along the
     window length at each of the indexes present in the data_len.
     The stable Points function returns arrays with p_blink_t and p_blinks_val  each with shape (n,3). p_blinks_t contains 

* Step 5
      
     
