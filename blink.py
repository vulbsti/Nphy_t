
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import cohen_kappa_score
from datetime import datetime
import time

fs=250 #Hz
thresh=100e-6  #peak to peak threshold for eye blink.
corr_threshold_2 =0.6   # used for blink template matching    
std_threshold_window =int(5*fs) # total window size for whose minimum standard deviation is to be taken, around peak.
cofa=6  #cofactor reduction for stable points value =2.5*cofa of min std_window

#calculate the a windowed standard deviation at each index of sample
def windowed_std(data_sig, fs):
    # Find running std for window length around the data_point of interest
    data_sig=np.array(data_sig)
    std_length = 125 # 0.5 seconds
    data_len=len(data_sig)
    running_std = np.zeros([data_len])
    idx = 0
    while(idx < (data_len - std_length)):
        running_std[idx] = np.std(data_sig[idx:(idx + std_length)])
        idx = idx + 1
    running_std[idx:data_len] = running_std[idx-1]

            
    return running_std

#using these dictionary type variable to store values based computed using peak_detection function
def args_init(delta_uV):
    args = {}
    args['mintab'], args['maxtab'] = [], []
    args['mn'], args['mx'] = float("inf"), -1*float("inf")
    args['mnpos'], args['mxpos'] = None, None
    args['min_left'], args['min_right'] = [], []
    args['lookformax'] = True
    args['delta'] = delta_uV
    return args
    
#finds and maximas and minimas in the dataset and stores values in array if they cross the threshold of 100 microvolts.
#the threshold is based on peak to peak difference in amplitudes.
def peakdet(time, value, args):
    foundMin = False
    if value > args['mx']:
        args['mx'] = value
        args['mxpos'] = time
    if value < args['mn']:
        args['mn'] = value
        args['mnpos'] = time
    if args['lookformax']:
        if value < args['mx'] - args['delta']:
            args['maxtab'].append([args['mxpos'], args['mx']])
            args['mn'] = value
            args['mnpos'] = time
            args['lookformax'] = False
    else:
        if value > args['mn'] + args['delta']:
            args['mintab'].append([args['mnpos'], args['mn']])
            args['min_left'].append([-1, -1])
            args['min_right'].append([-1, -1])
            args['mx'] = value
            args['mxpos'] = time
            args['lookformax'] = True                
            foundMin = True
    return foundMin

# function to convert a time object variable into seconds for computing onset time of dataset and annotations. 
def time2sec(time_var):
    '''
    if type(time_var) != datetime.time :
        dt=time_var.time()
    else:    
        
    '''
    dt=time_var  
    dts=dt.hour*3600 +dt.minute*60 +dt.second + +dt.microsecond*1e-6
    return dts

# this functions using in_built MNE filteration methods to apply a notch filter at 50Hz and a butterworth IIR bandpass filter 
# the bandpass filter has cutoff at 1Hz and 50Hz with 6th order.
def filteration(data,sfreq=250):
    dataf=mne.filter.notch_filter(data, Fs=250, freqs=50,method='iir') #dataf is a numpy array
    dataf=mne.filter.filter_data(dataf, sfreq=sfreq, l_freq=1, h_freq= 50, picks=None, 
                                filter_length='auto', l_trans_bandwidth='auto', 
                                h_trans_bandwidth='auto', n_jobs=None, method='iir',
                                iir_params={
                                 'ftype': 'butter',
                                 'order': 6,
                                }, copy=True, phase='zero', verbose=None) 

# Function narrows down the peak points to values of interest based on the stable_point theory.
# Parameters for narrowing are choosen such that most potential blinks are accounted for at the cost of false positives. 
# A balance trade off has been made based on the cummulative analysis of datasets hence been finetuned for the device in consideration
def find_expoints(peaks_arr, data_sig,std_win):
    # Parameters
    offset_t = 0.00 # in seconds
    win_size = 25
    win_shift = 10
    search_maxlen_t = 3 # in seconds

    data_sig=np.array(data_sig)
    offset_idx = int(offset_t*fs)   #index subtraction in case of time fragmented data
    search_maxlen_f = int(search_maxlen_t*fs)
    iters = int(search_maxlen_f/win_shift)  #about 37 iterations to account for multiple blinks in a row
    
    data_len = len(data_sig)
    p_blinks_t, p_blinks_val = [], []
    for idx in range(len(peaks_arr)):
        # x_indR and x_indL are starting points for left and right window they 
        x_indR = int(fs*peaks_arr[idx,0]) + offset_idx   
        x_indL = int(fs*peaks_arr[idx,0]) - offset_idx
        start_index = max(0, int(fs*peaks_arr[idx,0]) - std_threshold_window)
        end_index = min( int(fs*peaks_arr[idx,0]) + std_threshold_window, data_len)
        stable_threshold = min(std_win[start_index:end_index])    #2*min(running_std[start_index:end_index])
        min_val = peaks_arr[idx,1]
        max_val = min_val
        found1, found2 = 0, 0
        state1, state2 = 0, 0

        for i in range(iters):
            if(x_indR + win_size) > data_len:
                break
            if(x_indL < 0):
                break
            if np.std(data_sig[x_indR:x_indR+win_size]) < cofa*stable_threshold and state1==1  and data_sig[x_indR]>min_val:
                found1 = 1
                max_val = max(data_sig[x_indR],max_val)
            if np.std(data_sig[x_indL:x_indL+win_size]) < cofa*stable_threshold and state2==1 and data_sig[x_indL + win_size]>min_val:
                found2 = 1
                max_val = max(data_sig[x_indL + win_size],max_val)
            if np.std(data_sig[x_indR:x_indR+win_size]) > stable_threshold and state1==0:  #change 1.5 stable threshold
                state1 = 1
            if np.std(data_sig[x_indL:x_indL+win_size]) > stable_threshold and state2==0:
                state2 = 1
            if (found1==1) and data_sig[x_indR] < (max_val + 2*min_val)/6:
                found1=0
            if (found2==1) and data_sig[x_indL + win_size] < (max_val + 2*min_val)/6:
                found2=0
            if (found1==0):
                x_indR = x_indR + win_shift
            if (found2==0):
                x_indL = x_indL - win_shift
            if found1==1 and found2==1:
                break
        if found1==1 and found2==1:
            if (x_indL + win_size)/fs > peaks_arr[idx,0]:
                p_blinks_t.append([(x_indL)/fs, peaks_arr[idx,0], x_indR/fs])
                p_blinks_val.append([data_sig[x_indL], peaks_arr[idx,1], data_sig[x_indR]])         
            else:
                p_blinks_t.append([(x_indL + win_size)/fs, peaks_arr[idx,0], x_indR/fs])
                p_blinks_val.append([data_sig[x_indL + win_size], peaks_arr[idx,1], data_sig[x_indR]])
            

    p_blinks_t = np.array(p_blinks_t)        
    p_blinks_val = np.array(p_blinks_val)
    x=np.where(p_blinks_val[:,2]>2*thresh)   # update 7/9 accounting for high threshold artifacts
    p_blinks_t=np.delete(p_blinks_t,x,axis=0)
    p_blinks_val=np.delete(p_blinks_val,x,axis=0)
    
    return p_blinks_t, p_blinks_val

# method computes a self-correlation matrix for each and every potential eye_blink detected with every other potential blink in the selected channel data.
# Also computes a power matrix to compute relation between amplitudes. That is used to normalise data when forming clusters.
def compute_correlation(p_blinks_t, data_sig, fs=250):
    total_p_blinks = len(p_blinks_t)
    corr_matrix = np.ones([total_p_blinks, total_p_blinks])
    pow_matrix = np.ones([total_p_blinks, total_p_blinks])
    for idx_i in range(total_p_blinks):
        for idx_j in range(idx_i+1,total_p_blinks):

            blink_i_left = data_sig[int(fs*p_blinks_t[idx_i,0]):int(fs*p_blinks_t[idx_i,1])]
            blink_i_right = data_sig[int(fs*p_blinks_t[idx_i,1]):int(fs*p_blinks_t[idx_i,2])]

            blink_j_left = data_sig[int(fs*p_blinks_t[idx_j,0]):int(fs*p_blinks_t[idx_j,1])]
            blink_j_right = data_sig[int(fs*p_blinks_t[idx_j,1]):int(fs*p_blinks_t[idx_j,2])]
            
            #simply using this for managaing shape_size for different blink intervals in case i and j are of differing lengths
            #no need to use this if all the blink intervals are of same length
            left_interp = interp1d(np.arange(blink_i_left.size), blink_i_left)
            compress_left = left_interp(np.linspace(0,blink_i_left.size-1, blink_j_left.size))
            right_interp = interp1d(np.arange(blink_i_right.size), blink_i_right)
            compress_right = right_interp(np.linspace(0,blink_i_right.size-1, blink_j_right.size))

            sigA = np.concatenate((compress_left, compress_right))
            sigB = np.concatenate((blink_j_left, blink_j_right))
            
            corr = np.corrcoef(sigA, sigB)[0,1]
            
            corr_matrix[idx_i, idx_j] = corr
            corr_matrix[idx_j, idx_i] = corr
            
            if np.std(sigA) > np.std(sigB):
                pow_ratio = np.std(sigA)/np.std(sigB)
            else:
                pow_ratio = np.std(sigB)/np.std(sigA)
            
            pow_matrix[idx_i, idx_j] = pow_ratio
            pow_matrix[idx_j, idx_i] = pow_ratio
            

    return corr_matrix, pow_matrix

# create clusters based on correlation/power values and select cluster with max mean for choosing the best eye_blink clustered data.
def corr_match(data,chan,p_blinks_t,p_blinks_val,corr_matrix,pow_matrix):
    s_fc = (sum(corr_matrix))
    sort_idx = sorted(range(len(s_fc)), key=lambda k: s_fc[k])

    t = corr_matrix[sort_idx[-1],:] > corr_threshold_2        
    blink_index1 = set([i for i, x in enumerate(t) if x])
    t = corr_matrix[sort_idx[-2],:] > corr_threshold_2        
    blink_index2 = set([i for i, x in enumerate(t) if x])
    t = corr_matrix[sort_idx[-3],:] > corr_threshold_2        
    blink_index3 = set([i for i, x in enumerate(t) if x])

    blink_index = list(blink_index1.union(blink_index2).union(blink_index3))

    blink_template_corrmat = corr_matrix[np.ix_(blink_index,blink_index)]
    blink_template_powmat = pow_matrix[np.ix_(blink_index,blink_index)]
    blink_templates_corrWpower = blink_template_corrmat/blink_template_powmat

    blink_var = []
    for idx in blink_index:
        blink_var.append(np.var(data[chan,int(fs*p_blinks_t[idx,0]):int(fs*p_blinks_t[idx,2])]))


    Z = linkage(blink_templates_corrWpower, 'complete', 'correlation')
    groups = fcluster(Z,2,'maxclust')

    grp_1_blinks_var = [blink_var[i] for i, x in enumerate(groups==1) if x]
    grp_2_blinks_var = [blink_var[i] for i, x in enumerate(groups==2) if x]

    if np.mean(grp_1_blinks_var) > np.mean(grp_2_blinks_var) and np.mean(grp_1_blinks_var)/np.mean(grp_2_blinks_var) > 10:
        blink_index = [blink_index[i] for i, x in enumerate(groups==1) if x]
    elif np.mean(grp_2_blinks_var) > np.mean(grp_1_blinks_var) and np.mean(grp_2_blinks_var)/np.mean(grp_1_blinks_var) > 10:
        blink_index = [blink_index[i] for i, x in enumerate(groups==2) if x]


    final_blinks_t = p_blinks_t[blink_index,:]
    final_blinks_val = p_blinks_val[blink_index,:]
    
    return final_blinks_t , final_blinks_val



    return dataf


def eliminate(onset,val):
    # UCL + harvard research fastest eyeblink duration of 100ms and 400ms slow blink, exceptions are not taken into consideration
    #shorting onsets based on this.
    x=np.where(onset[:,2]-onset[:,0]<0.1)
    onset=np.delete(onset,x,axis=0)
    val=np.delete(val,x,axis=0)
    x=np.where(onset[:,2]-onset[:,0]>0.7)
    onset=np.delete(onset,x,axis=0)
    val=np.delete(val,x,axis=0)
    return onset, val

# main function that computes the eye_blink data for a specific channel. 
# P_blinks array can be provided in case a supervised version has to be generated for eye_blink detection
def compute(data,chan,p_blinks_t=0,p_blinks_val=0):
    
    std_win= windowed_std(data[chan],fs)
    pref=args_init(thresh)
    time=np.array(range(0,len(data[chan])))/fs #time_stamp for each index in data in seconds.
    
    if p_blinks_t==0:
    
        for i in range(len(data[chan,:])): peakdet(time[i],data[chan,i],pref)    #for when you need range and time for entire

        peaks=np.array(pref['mintab'])
    
        p_blinks_t, p_blinks_val = find_expoints(peaks, data[chan],std_win)

    corr_matrix, pow_matrix = compute_correlation(p_blinks_t, data[chan], fs)
    onset,val=corr_match(data,chan,p_blinks_t,p_blinks_val,corr_matrix,pow_matrix)
    return onset, val

  

# metrics are based on pre_annotated data. It doesnt computed any accuracy if the data provided doesnt contain annotated eye_blinks.
# the validity of these metrics is subject to pre_labelled data.
def metrics(onset,anot_t,anot_ke):
    acc=0
    false_p=0
    for t in anot_t:
        for i in range(len(onset)):
            if (onset[i,0]-0.4<t and onset[i,2]>t):
                acc=acc+1
                break
    close_st= anot_ke.index('CLOSED EYE START')
    close_en= anot_ke.index('CLOSED EYE END')
    #print (anot_t[close_st])
    #print(anot_t[close_en])
    
    for i in range(len(onset)):
        if onset[i,1]>anot_t[close_st]+1 and onset[i,1]<anot_t[close_en]-1:
            false_p=false_p+1
      
    if acc==0:
        recall=999999
        F1s=999999
        precision=999999
        accuracy=999999
    else:
        accuracy= acc/(len(anot_ke)-2)      #assuming here that only annotation labels to not be considered are present close and open eye
        precision = (len(onset)-false_p)/len(onset)
        recall=acc/(acc+false_p)
        F1s= (2*precision*recall)/(precision+recall)
    return acc,accuracy,false_p,precision,recall,F1s
            










        
        
        



