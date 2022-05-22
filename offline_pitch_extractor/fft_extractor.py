import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift
import math

tol = 1e-14


def log2(x):
    return (math.log10(x) /
            math.log10(2))


def is_power2(n):

    return math.ceil(log2(n)) == math.floor(log2(n))


def peakDetection(mX, t):
    """
    Detect spectral peak locations
    mX: magnitude spectrum, t: threshold
    returns ploc: peak locations
    """

    thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0)  # locations above threshold
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    return ploc


def cleaningSineTracks(tfreq, minTrackLength=3):
    """
    Delete short fragments of a collection of sinusoidal tracks
    tfreq: frequency of tracks
    minTrackLength: minimum duration of tracks in number of frames
    returns tfreqn: output frequency of tracks
    """

    if tfreq.shape[1] == 0:                                 # if no tracks return input
        return tfreq
    nFrames = tfreq[:,0].size                               # number of frames
    nTracks = tfreq[0,:].size                               # number of tracks in a frame
    for t in range(nTracks):                                # iterate over all tracks
        trackFreqs = tfreq[:,t]                               # frequencies of one track
        trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0)  # begining of track contours
                                & (trackFreqs[1:]>0))[0] + 1
        if trackFreqs[0]>0:
            trackBegs = np.insert(trackBegs, 0, 0)
        trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0)   # end of track contours
                                & (trackFreqs[1:] <=0))[0] + 1
        if trackFreqs[nFrames-1]>0:
            trackEnds = np.append(trackEnds, nFrames-1)
        trackLengths = 1 + trackEnds - trackBegs              # lengths of trach contours
        for i,j in zip(trackBegs, trackLengths):              # delete short track contours
            if j <= minTrackLength:
                trackFreqs[i:i+j] = 0
    return tfreq


def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
    """
    Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
    and calling the function TWM with them
    pfreq, pmag: peak frequencies and magnitudes,
    ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
    f0t: f0 of previous frame if stable
    returns f0: fundamental frequency in Hz
    """
    if (minf0 < 0):                                  # raise exception if minf0 is smaller than 0
        raise ValueError("Minimum fundamental frequency (minf0) smaller than 0")

    if (maxf0 >= 10000):                             # raise exception if maxf0 is bigger than 10000Hz
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if (pfreq.size < 3) & (f0t == 0):                # return 0 if less than 3 peaks and not previous f0
        return 0

    f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0] # use only peaks within given range
    if (f0c.size == 0):                              # return 0 if no peaks within range
        return 0
    f0cf = pfreq[f0c]                                # frequencies of peak candidates
    f0cm = pmag[f0c]                                 # magnitude of peak candidates

    if f0t>0:                                        # if stable f0 in previous frame
        shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]   # use only peaks close to it
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc]%f0t
        if maxcfd > f0t/2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd>(f0t/4)): # or the maximum magnitude peak is not a harmonic
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]                         # frequencies of candidates

    if (f0cf.size == 0):                             # return 0 if no peak candidates
        return 0

    f0, f0error = TWM_p(pfreq, pmag, f0cf)        # call the TWM function with peak candidates

    if (f0>0) and (f0error<ef0max):                  # accept and return f0 if below max error allowed
        return f0
    else:
        return 0


def TWM_p(pfreq, pmag, f0c):
    """
    Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
    [better to use the C version of this function: UF_C.twm]
    pfreq, pmag: peak frequencies in Hz and magnitudes,
    f0c: frequencies of f0 candidates
    returns f0, f0Error: fundamental frequency detected and its error
    """

    p = 0.5                                          # weighting by frequency value
    q = 1.4                                          # weighting related to magnitude of peaks
    r = 0.5                                          # scaling related to magnitude of peaks
    rho = 0.33                                       # weighting of MP error
    Amax = max(pmag)                                 # maximum peak magnitude
    maxnpeaks = 10                                   # maximum number of peaks used
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
        harmonic = harmonic+f0c

    ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size) :                    # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP]/f0c[i])
        nharm = (nharm>=1)*nharm + (nharm<1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

    Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
    f0index = np.argmin(Error)                       # get the smallest error
    f0 = f0c[f0index]                                # f0 with the smallest error

    return f0, Error[f0index]


def peakInterp(mX, pX, ploc):
    """
    Interpolate peak values using parabolic interpolation
    mX, pX: magnitude and phase spectrum, ploc: locations of peaks
    returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
    """

    val = mX[ploc]                                          # magnitude of peak bin
    lval = mX[ploc-1]                                       # magnitude of bin at left
    rval = mX[ploc+1]                                       # magnitude of bin at right
    iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
    ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase


def dft_anal(x, w, N):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns mX, pX: magnitude and phase spectrum
    """

    if not is_power2(N):                                 # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if w.size > N:                                        # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N//2)+1                                           # size of positive spectrum, it includes sample 0
    hM1 = (w.size+1)//2                                     # half analysis window size by rounding
    hM2 = w.size//2                                         # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    xw = x*w                                                # window the input sound
    fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)                                      # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX


def sinc(x, N):
    """
    Generate the main lobe of a sinc function (Dirichlet kernel)
    x: array of indexes to compute; N: size of FFT to simulate
    returns y: samples of the main lobe of a sinc function
    """

    y = np.sin(N * x/2) / np.sin(x/2)                  # compute the sinc function
    y[np.isnan(y)] = N                                 # avoid NaN if x == 0
    return y


def gen_bh_lobe(x):
    """
    Generate the main lobe of a Blackman-Harris window
    x: bin positions to compute (real values)
    returns y: main lobe os spectrum of a Blackman-Harris window
    """

    N = 512                                                 # size of fft to use
    f = x*np.pi*2/N                                         # frequency sampling
    df = 2*np.pi/N
    y = np.zeros(x.size)                                    # initialize window
    consts = [0.35875, 0.48829, 0.14128, 0.01168]           # window constants
    for m in range(0,4):                                    # iterate over the four sincs to sum
        y += consts[m]/2 * (sinc(f-df*m, N) + sinc(f+df*m, N))  # sum of scaled sinc functions
    y = y/N/consts[0]                                       # normalize
    return y


def genSpecSines_p(ipfreq, ipmag, ipphase, N, fs):
    """
    Generate a spectrum from a series of sine values
    iploc, ipmag, ipphase: sine peaks locations, magnitudes and phases
    N: size of the complex spectrum to generate; fs: sampling rate
    returns Y: generated complex spectrum of sines
    """

    Y = np.zeros(N, dtype = complex)                 # initialize output complex spectrum
    hN = N//2                                        # size of positive freq. spectrum
    for i in range(0, ipfreq.size):                  # generate all sine spectral lobes
        loc = N * ipfreq[i] / fs                       # it should be in range ]0,hN-1[
        if loc==0 or loc>hN-1: continue
        binremainder = round(loc)-loc;
        lb = np.arange(binremainder-4, binremainder+5) # main lobe (real value) bins to read
        lmag = gen_bh_lobe(lb) * 10**(ipmag[i]/20)       # lobe magnitudes of the complex exponential
        b = np.arange(round(loc)-4, round(loc)+5, dtype='int')
        for m in range(0, 9):
            if b[m] < 0:                                 # peak lobe crosses DC bin
                Y[-b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
            elif b[m] > hN:                              # peak lobe croses Nyquist bin
                Y[b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
            elif b[m] == 0 or b[m] == hN:                # peak lobe in the limits of the spectrum
                Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i]) + lmag[m]*np.exp(-1j*ipphase[i])
            else:                                        # peak lobe in positive freq. range
                Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i])
        Y[hN+1:] = Y[hN-1:0:-1].conjugate()            # fill the negative part of the spectrum
    return Y


def stochasticModelAnal(x, H, N, stocf):
    """
    Stochastic analysis of a sound
    x: input array sound, H: hop size, N: fftsize
    stocf: decimation factor of mag spectrum for stochastic analysis, bigger than 0, maximum of 1
    returns stocEnv: stochastic envelope
    """

    hN = N // 2 + 1  # positive size of fft
    No2 = N // 2  # half of N
    if (hN * stocf < 3):  # raise exception if decimation factor too small
        raise ValueError("Stochastic decimation factor too small")

    if (stocf > 1):  # raise exception if decimation factor too big
        raise ValueError("Stochastic decimation factor above 1")

    if (H <= 0):  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    if not (is_power2(N)):  # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    w = signal.windows.hanning(N)  # analysis window
    x = np.append(np.zeros(No2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(No2))  # add zeros at the end to analyze last sample
    pin = No2  # initialize sound pointer in middle of analysis window
    pend = x.size - No2  # last sample to start a frame
    while pin <= pend:
        xw = x[pin - No2:pin + No2] * w  # window the input sound
        X = fft(xw)  # compute FFT
        mX = 20 * np.log10(abs(X[:hN]))  # magnitude spectrum of positive frequencies
        mY = signal.resample(np.maximum(-200, mX), int(stocf * hN))  # decimate the mag spectrum
        if pin == No2:  # first frame
            stocEnv = np.array([mY])
        else:  # rest of frames
            stocEnv = np.vstack((stocEnv, np.array([mY])))
        pin += H  # advance sound pointer
    return stocEnv


def harmonic_detection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency
    pfreq, pmag, pphase: peak frequencies, magnitudes and phases
    f0: fundamental frequency, nH: number of harmonics,
    hfreqp: harmonic frequencies of previous frame,
    fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
    """

    if f0<=0:                                          # if no f0 return no harmonics
        return np.zeros(nH), np.zeros(nH), np.zeros(nH)
    hfreq = np.zeros(nH)                                 # initialize harmonic frequencies
    hmag = np.zeros(nH)-100                              # initialize harmonic magnitudes
    hphase = np.zeros(nH)                                # initialize harmonic phases
    hf = f0*np.arange(1, nH+1)                           # initialize harmonic frequencies
    hi = 0                                               # initialize harmonic index
    if hfreqp == []:                                     # if no incomming harmonic tracks initialize to harmonic series
        hfreqp = hf
    while (f0>0) and (hi<nH) and (hf[hi]<fs/2):          # find harmonic peaks
        pei = np.argmin(abs(pfreq - hf[hi]))               # closest peak
        dev1 = abs(pfreq[pei] - hf[hi])                    # deviation from perfect harmonic
        dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi]>0 else fs) # deviation from previous frame
        threshold = f0/3 + harmDevSlope * pfreq[pei]
        if dev1<threshold or dev2<threshold:         # accept peak if deviation is small
            hfreq[hi] = pfreq[pei]                           # harmonic frequencies
            hmag[hi] = pmag[pei]                             # harmonic magnitudes
            hphase[hi] = pphase[pei]                         # harmonic phases
        hi += 1                                            # increase harmonic index
    return hfreq, hmag, hphase

def sine_subtraction(x, N, H, sfreq, smag, sphase, fs):
    """
    Subtract sinusoids from a sound
    x: input sound, N: fft-size, H: hop-size
    sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
    returns xr: residual sound
    """

    hN = N//2                                          # half of fft size
    x = np.append(np.zeros(hN),x)                      # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hN))                      # add zeros at the end to analyze last sample
    bh = signal.windows.blackmanharris(N)                            # blackman harris window
    w = bh/ sum(bh)                                    # normalize window
    sw = np.zeros(N)                                   # initialize synthesis window
    sw[hN-H:hN+H] = signal.windows.triang(2*H) / w[hN-H:hN+H]         # synthesis window
    L = sfreq.shape[0]                                 # number of frames, this works if no sines
    xr = np.zeros(x.size)                              # initialize output array
    pin = 0
    for l in range(L):
        xw = x[pin:pin+N]*w                              # window the input sound
        X = fft(fftshift(xw))                            # compute FFT
        Yh = genSpecSines_p(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N, fs)   # generate spec sines
        Xr = X-Yh                                        # subtract sines from original spectrum
        xrw = np.real(fftshift(ifft(Xr)))                # inverse FFT
        xr[pin:pin+N] += xrw*sw                          # overlap-add
        pin += H                                         # advance sound pointer
    xr = np.delete(xr, range(hN))                      # delete half of first window which was added in stftAnal
    xr = np.delete(xr, range(xr.size-hN, xr.size))     # delete half of last window which was added in stftAnal
    return xr


def stochasticResidualAnal(x, N, H, sfreq, smag, sphase, fs, stocf):
    """
    Subtract sinusoids from a sound and approximate the residual with an envelope
    x: input sound, N: fft size, H: hop-size
    sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
    fs: sampling rate; stocf: stochastic factor, used in the approximation
    returns stocEnv: stochastic approximation of residual
    """

    hN = N//2                                             # half of fft size
    x = np.append(np.zeros(hN),x)                         # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hN))                         # add zeros at the end to analyze last sample
    bh = signal.windows.blackmanharris(N)                                # synthesis window
    w = bh/ sum(bh)                                       # normalize synthesis window
    L = sfreq.shape[0]                                    # number of frames, this works if no sines
    pin = 0
    for l in range(L):
        xw = x[pin:pin+N] * w                               # window the input sound
        X = fft(fftshift(xw))                               # compute FFT
        Yh = genSpecSines_p(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N)   # generate spec sines
        Xr = X-Yh                                           # subtract sines from original spectrum
        mXr = 20*np.log10(abs(Xr[:hN]))                     # magnitude spectrum of residual
        mXrenv = signal.resample(np.maximum(-200, mXr), mXr.size*stocf)  # decimate the mag spectrum
        stocEnv = None
        if l == 0:                                          # if first frame
            stocEnv = np.array([mXrenv])
        else:                                               # rest of frames
            stocEnv = np.vstack((stocEnv, np.array([mXrenv])))
        pin += H                                            # advance sound pointer
        return stocEnv


def harmonicDetection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency
    pfreq, pmag, pphase: peak frequencies, magnitudes and phases
    f0: fundamental frequency, nH: number of harmonics,
    hfreqp: harmonic frequencies of previous frame,
    fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
    """

    if (f0<=0):                                          # if no f0 return no harmonics
        return np.zeros(nH), np.zeros(nH), np.zeros(nH)
    hfreq = np.zeros(nH)                                 # initialize harmonic frequencies
    hmag = np.zeros(nH)-100                              # initialize harmonic magnitudes
    hphase = np.zeros(nH)                                # initialize harmonic phases
    hf = f0*np.arange(1, nH+1)                           # initialize harmonic frequencies
    hi = 0                                               # initialize harmonic index
    if hfreqp == []:                                     # if no incomming harmonic tracks initialize to harmonic series
        hfreqp = hf
    while (f0>0) and (hi<nH) and (hf[hi]<fs/2):          # find harmonic peaks
        pei = np.argmin(abs(pfreq - hf[hi]))               # closest peak
        dev1 = abs(pfreq[pei] - hf[hi])                    # deviation from perfect harmonic
        dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi]>0 else fs) # deviation from previous frame
        threshold = f0/3 + harmDevSlope * pfreq[pei]
        if ((dev1<threshold) or (dev2<threshold)):         # accept peak if deviation is small
            hfreq[hi] = pfreq[pei]                           # harmonic frequencies
            hmag[hi] = pmag[pei]                             # harmonic magnitudes
            hphase[hi] = pphase[pei]                         # harmonic phases
        hi += 1                                            # increase harmonic index
    return hfreq, hmag, hphase

def harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope=0.01, minSineDur=.02):
    """
    Analysis of a sound using the sinusoidal harmonic model
    x: input sound; fs: sampling rate, w: analysis window; N: FFT size (minimum 512); t: threshold in negative dB,
    nH: maximum number of harmonics;  minf0: minimum f0 frequency in Hz,
    maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
    harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
    returns xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases
    """

    if minSineDur < 0:  # raise exception if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hN = N // 2  # size of positive spectrum
    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))  # add zeros at the end to analyze last sample
    pin = hM1  # init sound pointer in middle of anal window
    pend = x.size - hM1  # last sample to start a frame
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    w = w / sum(w)  # normalize analysis window
    hfreqp = []  # initialize harmonic frequencies of previous frame
    f0t = 0  # initialize f0 track
    f0stable = 0  # initialize f0 stable
    while pin <= pend:
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = dft_anal(x1, w, N)  # compute dft
        ploc = peakDetection(mX, t)  # detect peak locations
        iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)  # refine peak values
        ipfreq = fs * iploc / N  # convert locations to Hz
        f0t = f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable - f0t) < f0stable / 5.0)):
            f0stable = f0t  # consider a stable f0 if it is close to the previous one
        else:
            f0stable = 0
        hfreq, hmag, hphase = harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs,
                                                harmDevSlope)  # find harmonics
        hfreqp = hfreq
        if pin == hM1:  # first frame
            xhfreq = np.array([hfreq])
            xhmag = np.array([hmag])
            xhphase = np.array([hphase])
        else:  # next frames
            xhfreq = np.vstack((xhfreq, np.array([hfreq])))
            xhmag = np.vstack((xhmag, np.array([hmag])))
            xhphase = np.vstack((xhphase, np.array([hphase])))
        pin += H  # advance sound pointer
    xhfreq = cleaningSineTracks(xhfreq, round(fs * minSineDur / H))  # delete tracks shorter than minSineDur
    return xhfreq, xhmag, xhphase


def fft_extractor(audio, sr=44100, M=2001, H=2*256, t=-90, nH=5, minf0=50, maxf0=1500, f0et=2, harmDevSlope=0.01, minSineDur=0.01, Ns=2048, stocf=0.3):
    """
    Extracts the fundamental frequency given an input sound using the FFT method.
    Args:
        audio: the input sound (list of float)
        sr: the sampling rate (int)
    Returns:
        freq: the estimated fundamental frequency (float)
    """
    """
    Analysis of a sound using the harmonic plus stochastic model
    x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB, 
    nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
    maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
    harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
    returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; stocEnv: stochastic residual
    """
    N = int(2**np.ceil(np.log2(M)))
    w = signal.windows.blackmanharris(M)

    # perform harmonic analysis
    hfreq, hmag, hphase = harmonicModelAnal(audio, sr, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
    # subtract sinusoids from original sound
    xr = sine_subtraction(audio, Ns, H, hfreq, hmag, hphase, sr)
    # perform stochastic analysis of residual
    stocEnv = stochasticModelAnal(xr, H, H * 2, stocf)

    # create figure to plot
    #plt.figure(figsize=(9, 6))

    # frequency range to plot
    maxplotfreq = 2000.0

    # plot spectrogram stochastic component
    # plt.subplot(3, 1, 2)
    numFrames = int(stocEnv[:, 0].size)
    sizeEnv = int(stocEnv[0, :].size)
    frmTime = H * np.arange(numFrames) / float(sr)
    binFreq = (.5 * sr) * np.arange(sizeEnv * maxplotfreq / (.5 * sr)) / sizeEnv
    # plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv[:, :int(sizeEnv * maxplotfreq / (.5 * sr) + 1)]))
    # plt.autoscale(tight=True)

    # plot harmonic on top of stochastic spectrogram
    if (hfreq.shape[1] > 0):
        harms = hfreq * np.less(hfreq, maxplotfreq)
        harms[harms == np.nan] = 0
        numFrames = harms.shape[0]
        frmTime = H * np.arange(numFrames) / float(sr)
        """plt.plot(frmTime, harms[:,0], color='k', ms=3, alpha=1)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.autoscale(tight=True)
        plt.title(f"harmonics + stochastic spectrogram")"""

    # plt.tight_layout()
    # plt.show()

    return harms[:,0]
