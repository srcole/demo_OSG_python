"""
This library contains metrics to quantify the shape of a waveform
1. threshold_amplitude - only look at a metric while oscillatory amplitude is above a set percentile threshold
2. rdratio: Ratio of rise time and decay time
3. symPT: symmetry between peak and trough
4. symRD: symmetry between rise and decay
5. pt_sharp - calculate sharpness of oscillatory extrema
6. rd_steep - calculate rise and decay steepness
7. ptsr - calculate extrema sharpness ratio
8. rdsr - calculate rise-decay steepness ratio
"""

from __future__ import division
import numpy as np
from misshapen.nonshape import ampT, bandpass_default


def threshold_amplitude(x, metric, samples, percentile, frange, Fs, filter_fn=None, filter_kwargs=None):
    """
    Exclude from analysis the samples in which the amplitude falls below a defined percentile
    
    Parameters
    ----------
    x : numpy array
        raw time series
    metric : numpy array
        series of measures corresponding to time samples in 'samples' (e.g. peak sharpness)
    samples : numpy array
        time samples at which metric was computer (e.g. peaks)
    percentile : float
        percentile cutoff for exclusion (e.g. 10 = bottom 10% excluded)
    frange : [lo, hi]
        frequency range of interest for calculating amplitude
    Fs : float
        Sampling rate (Hz)
        
    Returns
    -------
    metric_new : numpy array
        same as input 'metric' but only for samples above the amplitude threshold
    samples_new : numpy array
        samples above the amplitude threshold
    """
    
    # Do nothing if threshold is 0
    if percentile == 0:
        return metric, samples
    
    # Default filter function
    if filter_fn is None:
        filter_fn = bandpass_default
    if filter_kwargs is None:
        filter_kwargs = {}

    # Calculate amplitude time series and threshold
    amp = ampT(x, frange, Fs, rmv_edge = False, filter_fn=filter_fn, filter_kwargs=filter_kwargs)
    amp = amp[samples]
    amp_threshold = np.percentile(amp, percentile)
    
    # Update samples used
    samples_new = samples[amp>=amp_threshold]
    metric_new = metric[amp>=amp_threshold]
    
    return metric_new, samples_new
    

def rdratio(Ps, Ts):
    """
    Calculate the ratio between rise time and decay time for oscillations
    
    Note: must have the same number of peaks and troughs
    Note: the final rise or decay is unused
    
    Parameters
    ----------
    Ps : numpy arrays 1d
        time points of oscillatory peaks
    Ts : numpy arrays 1d
        time points of osillatory troughs
        
    Returns
    -------
    rdr : array-like 1d
        rise-decay ratios for each oscillation
    """
    
    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')
        
    # Assure Ps and Ts are numpy arrays
    if type(Ps)==list or type(Ts)==list:
        print 'Converted Ps and Ts to numpy arrays'
        Ps = np.array(Ps)
        Ts = np.array(Ts)
        
    # Calculate rise and decay times
    if Ts[0] < Ps[0]:
        riset = Ps[:-1] - Ts[:-1]
        decayt = Ts[1:] - Ps[:-1]
    else:
        riset = Ps[1:] - Ts[:-1]
        decayt = Ts[:-1] - Ps[:-1]
            
    # Calculate ratio between each rise and decay time
    rdr = riset / decayt.astype(float)
    
    return riset, decayt, rdr
    

def symPT(x, Ps, Ts, window_half):
    """
    Measure of asymmetry between oscillatory peaks and troughs

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
    window_half : int
        Number of samples around extrema to analyze, in EACH DIRECTION
        
    Returns
    -------
    sym : array-like 1d
        measure of symmetry between each trough-peak pair
        Result of 0 means the peak and trough are perfectly symmetric
    
    Notes
    -----
    Opt 2: Roemer; The metric should be between 0 and 1
    Inner product of Peak and Trough divided by the squareroot of the product of SSQ_peak and SSQ_trough
    
    I'll need to fine tune this to make it more complicated and less susceptible to noise
    """
    
    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')
      
    E = len(Ps)
    sym = np.zeros(E)
    for e in range(E):
        # Find region around each peak and trough. Make extrema be 0
        peak = x[Ps[e]-window_half:Ps[e]+window_half+1] - x[Ps[e]]
        peak = -peak
        trough = x[Ts[e]-window_half:Ts[e]+window_half+1] - x[Ts[e]]
        
        # Compare the two measures
        peakenergy = np.sum(peak**2)
        troughenergy = np.sum(trough**2)
        energy = np.max((peakenergy,troughenergy))
        diffenergy = np.sum((peak-trough)**2)
        sym[e] = diffenergy / energy

    return sym
    

def symRD(x, Ts, window_full):
    """
    Measure of asymmetry between oscillatory peaks and troughs

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ts : array-like 1d
        time points of oscillatory troughs
    window_full : int
        Number of samples after peak to analyze for decay and before peak to analyze for rise
        
    Returns
    -------
    sym : array-like 1d
        measure of symmetry between each rise and decay
    """
      
    T = len(Ts)
    sym = np.zeros(T)
    for t in range(T):
        # Find regions for the rise and the decay
        rise = x[Ts[t]:Ts[t]+window_full+1] - x[Ts[t]]
        decay = x[Ts[t]-window_full:Ts[t]+1] - x[Ts[t]]

        # Ensure the minimum value is 0
        rise[rise<0] = 0
        decay[decay<0] = 0

        # Make rises and decays go the same direction
        rise = np.flipud(rise)
        
        # Calculate absolute difference between each point in the rise and decay
        diffenergy = np.sum(np.abs(rise-decay))
        
        # Normalize this difference by the max voltage value at each point
        rise_decay_maxes = np.max(np.vstack((rise,decay)),axis=0)
        energy = np.sum(rise_decay_maxes)
        
        # Compare the two measures
        sym[t] = diffenergy / energy
        
    return sym
    
    
def pt_sharp(x, Ps, Ts, window_half, method='diff'):
    """
    Calculate the sharpness of extrema
    
    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
    window_half : int
        Number of samples in each direction around extrema to use for sharpness estimation
        
    Returns
    -------
    Psharps : array-like 1d
        sharpness of peaks
    Tsharps : array-like 1d
        sharpness of troughs
    
    """
    
    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')
        
    # Calculate the sharpness of each peak
    P = len(Ps)
    Psharps = np.zeros(P)
    for e in range(P):
        if method == 'deriv':
            Edata = x[Ps[e]-window_half: Ps[e]+window_half+1]
            Psharps[e] = np.mean(np.abs(np.diff(Edata)))
        elif method == 'diff':
            Psharps[e] = np.mean((x[Ps[e]]-x[Ps[e]-window_half],x[Ps[e]]-x[Ps[e]+window_half]))
    
    T = len(Ts)
    Tsharps = np.zeros(T)
    for e in range(T):
        if method == 'deriv':
            Edata = x[Ts[e]-window_half: Ts[e]+window_half+1]
            Tsharps[e] = np.mean(np.abs(np.diff(Edata)))
        elif method == 'diff':
            Tsharps[e] = np.mean((x[Ts[e]-window_half]-x[Ts[e]],x[Ts[e]+window_half]-x[Ts[e]]))
    
    return Psharps, Tsharps
    
    
def rd_steep(x, Ps, Ts):
    """
    Calculate the max steepness of rises and decays
    
    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
        
    Returns
    -------
    risesteep : array-like 1d
        max steepness in each period for rise
    decaysteep : array-like 1d
        max steepness in each period for decay
    """
    
    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')
        
    # Calculate rise and decay steepness
    E = len(Ps) - 1
    risesteep = np.zeros(E)
    for t in range(E):
        if Ts[0] < Ps[0]:
            rise = x[Ts[t]:Ps[t]+1]
        else:
            rise = x[Ts[t]:Ps[t+1]+1]
        risesteep[t] = np.max(np.diff(rise))
        
    decaysteep = np.zeros(E)
    for p in range(E):
        if Ts[0] < Ps[0]:
            decay = x[Ps[p]:Ts[p+1]+1]
        else:
            decay = x[Ps[p]:Ts[p]+1]
        decaysteep[p] = -np.min(np.diff(decay))
        
    return risesteep, decaysteep
    
    
def ptsr(Psharp,Tsharp, log = True, polarity = True):
    if polarity:
        sharpnessratio = Psharp/Tsharp
    else:
        sharpnessratio = np.max((Psharp/Tsharp,Tsharp/Psharp))
    if log:
        sharpnessratio = np.log10(sharpnessratio)
    return sharpnessratio
    
    
def rdsr(Rsteep,Dsteep, log = True, polarity = True):
    if polarity:
        steepnessratio = Rsteep/Dsteep
    else:
        steepnessratio = np.max((Rsteep/Dsteep,Dsteep/Rsteep))
    if log:
        steepnessratio = np.log10(steepnessratio)
    return steepnessratio