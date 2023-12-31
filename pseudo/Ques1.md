# Algorithm for Blink Detection

Step 1> 
Input the data using MNE methods and extract the dataset in a numpy format with a (n,   n_chan) shape

Step 2>  
Pass the data in the form of a numpy array in a filtration method that returns data with the same shape.
Filtration method applies a notch filter of 50Hz and a bandpass butterworth 6th order filter of 1-50Hz

 Step 3> 
pass the data samples for a selected channel in a peak_det function with inputs as time  of the occurrence (index of data / samplefreq) and value of data at the index. A for loop with range(dataset_len) is used to find local crests and troughs crossing a certain threshold (100uV) in the entire channel length.

Step 4> 
The array of local minimas with their time_indexes is passed to a stable_points function along with the selected channel data and an array containing standard deviations along the  window length at each of the indexes present in the data_len.
The stable Points function returns arrays with p_blink_t (blink onset, peak and end time) and p_blinks_val (value at those time indexes)  each with shape (n,3).

Step 5>
    Next step is to compute correlation between each of the potential blink_intervals. and create a correlation matrix
    ~~~
    for i in range (total_p_blinks):
     for j in range (i+1,total_p_blinks):
        signal A = data_signal from p_blink_t[i,0] *samplefreq to p_blinks_t[i,3] * samplefreq
        signal B = data_signal p_blink_t[j,0] *samplefreq to p_blinks_t[j,3] * samplefreq
        find correlation coefficient between signal A and signal B
        add correcoeff to correlation matrix[i,j]
        power ratio = standard deviation(signal A)/standard deviation(signal B)
        add power ratio to power martix pow[i,j]
    ~~~ 
     
Step 6>
Sort the correlation and power matrix based on decreasing values, and eliminate any of the values from the matrix that are less than correlation threshold. This is done to choose the best templates for blink.

Step 7>
Find the ratio of Blink_correlation_matrix / Blink_power_matrix. And create a hierarchical cluster, based on this ratio. We choose this ratio to normalize the correlation values with the square of the amplitudes.
Hence creating a normalized value for each distance pair.
All relationships measured are based on a linear relationship between sample points.

Step 8>
Finally create a flat cluster with two groups and assign each of the cluster created using the previous method to any one of the group. 
Divide the blink_variables ie data_samples in blink duration to  those groups, and find the mean for those variables in each group. And select the best group based on greater mean criteria. Reduce the p_blink_t array to only the selected indexes from the above grouping


Step 9>
Perform  the above steps on each of the interested channels and run a metrics function to calculate accuracy and false prediction around each channel. Select the channel with best accuracy. 
(Note since a consistent high accuracy was found with channel Fp1 over Fp2 hence it was selected as default channel for detecting Eye Blinks) 

Step 10>
  Plot the reduced p_blinks_t using MNE annotation methods onto the raw dataset.

  
