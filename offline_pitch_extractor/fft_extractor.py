import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift
import math

tol = 1e-14


def is_power2(n):
    return math.ceil(np.log2(n)) == math.floor(np.log2(n))


def peak_detection(x_mag, th):
    """
    Detect spectral peak locations
    mX: magnitude spectrum, t: threshold
    returns peak_loc: peak locations
    """

    thresh = np.where(np.greater(x_mag[1:-1], th), x_mag[1:-1], 0)      # locations above threshold
    next_minor = np.where(x_mag[1:-1] > x_mag[2:], x_mag[1:-1], 0)     # locations higher than the next one
    prev_minor = np.where(x_mag[1:-1] > x_mag[:-2], x_mag[1:-1], 0)    # locations higher than the previous one
    peak_loc = thresh * next_minor * prev_minor                        # locations fulfilling the three criteria
    peak_loc = peak_loc.nonzero()[0] + 1                               # add 1 to compensate for previous steps
    return peak_loc


def cleaning_sine_tracks(track_freq, min_track_length=3):
    """
    Delete short fragments of a collection of sinusoidal tracks
    tfreq: frequency of tracks
    minTrackLength: minimum duration of tracks in number of frames
    returns tfreqn: output frequency of tracks
    """

    if track_freq.shape[1] == 0:                                   # if no tracks return input
        return track_freq
    n_frames = track_freq[:, 0].size                               # number of frames
    n_tracks = track_freq[0, :].size                               # number of tracks in a frame
    for t in range(n_tracks):                                      # iterate over all tracks
        track_freqs = track_freq[:, t]                             # frequencies of one track
        track_begs = np.nonzero((track_freqs[:n_frames - 1] <= 0)  # beginning of track contours
                                & (track_freqs[1:] > 0))[0] + 1
        if track_freqs[0] > 0:
            track_begs = np.insert(track_begs, 0, 0)
        track_ends = np.nonzero((track_freqs[:n_frames-1] > 0)     # end of track contours
                                & (track_freqs[1:] <=0))[0] + 1
        if track_freqs[n_frames-1] > 0:
            track_ends = np.append(track_ends, n_frames-1)
        track_lengths = 1 + track_ends - track_begs                # lengths of track contours
        for i, j in zip(track_begs, track_lengths):                # delete short track contours
            if j <= min_track_length:
                track_freqs[i:i+j] = 0
    return track_freq


def f0_twm(peak_freq, peak_mag, max_error_f0, min_f0, max_f0, f0_th=0):
    """
    Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
    and calling the function TWM with them
    peak_freq, pmag: peak frequencies and magnitudes,
    max_error_f0: maximum error allowed, min_f0, max_f0: minimum  and maximum f0
    f0t: f0 of previous frame if stable
    returns f0: fundamental frequency in Hz
    """
    if min_f0 < 0:
        raise ValueError(f"Minimum fundamental frequency min_f0={min_f0} smaller than 0")

    if max_f0 >= 10000:
        raise ValueError(f"Maximum fundamental frequency max_f0={max_f0} bigger than 10000Hz")

    if peak_freq.size < 3 and f0_th == 0:
        return 0

    f0_cand = np.argwhere((peak_freq > min_f0) & (peak_freq < max_f0))[:, 0]  # use only peaks within given range

    if f0_cand.size == 0:                              # return 0 if no peaks within range
        return 0
    f0_cand_freq = peak_freq[f0_cand]                  # frequencies of peak candidates
    f0_cand_mag = peak_mag[f0_cand]                    # magnitude of peak candidates

    if f0_th > 0:                                      # if stable f0 in previous frame
        shortlist = np.argwhere(np.abs(f0_cand_freq - f0_th) < f0_th / 2.0)[:, 0]   # use only peaks close to it
        max_cand = np.argmax(f0_cand_mag)
        max_cfd = f0_cand_freq[max_cand] % f0_th
        if max_cfd > f0_th/2:
            max_cfd = f0_th - max_cfd
        if (max_cand not in shortlist) and (max_cfd > (f0_th / 4)):  # or the maximum magnitude peak is not a harmonic
            shortlist = np.append(max_cand, shortlist)
        f0_cand_freq = f0_cand_freq[shortlist]                       # frequencies of candidates

    if f0_cand_freq.size == 0:                                       # return 0 if no peak candidates
        return 0

    f0, f0_error = twm_peaks(peak_freq, peak_mag, f0_cand_freq)          # call the TWM function with peak candidates

    if f0 > 0 and f0_error < max_error_f0:                       # accept and return f0 if below max error allowed
        return f0
    else:
        return 0


def twm_peaks(peak_freq, peak_mag, f0_cand):
    """
    Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
    [better to use the C version of this function: UF_C.twm]
    pfreq, pmag: peak frequencies in Hz and magnitudes,
    f0c: frequencies of f0 candidates
    returns f0, f0Error: fundamental frequency detected and its error
    """
    p = 0.5                                                 # weighting by frequency value
    q = 1.4                                                 # weighting related to magnitude of peaks
    r = 0.5                                                 # scaling related to magnitude of peaks
    rho = 0.33                                              # weighting of MP error
    mag_max = max(peak_mag)                                 # maximum peak magnitude
    max_num_peaks = 10                                      # maximum number of peaks used
    harmonic = np.matrix(f0_cand)
    error_pm = np.zeros(harmonic.size)                      # initialize PM errors
    max_npm = min(max_num_peaks, peak_freq.size)
    for i in range(0, max_npm):                             # predicted to measured mismatch error
        dif_matrix_pm = harmonic.T * np.ones(peak_freq.size)
        dif_matrix_pm = abs(dif_matrix_pm - np.ones((harmonic.size, 1)) * peak_freq)
        freq_distance = np.amin(dif_matrix_pm, axis=1)      # minimum along rows
        peak_loc = np.argmin(dif_matrix_pm, axis=1)
        pond_dif = np.array(freq_distance) * (np.array(harmonic.T)**(-p))
        mag_peaks = peak_mag[peak_loc]
        mag_factor = 10**((mag_peaks-mag_max)/20)
        error_pm = error_pm + (pond_dif + mag_factor*(q*pond_dif-r)).T
        harmonic = harmonic + f0_cand

    error_mp = np.zeros(harmonic.size)                      # initialize MP errors
    max_nmp = min(max_num_peaks, peak_freq.size)
    for i in range(0, f0_cand.size):                        # measured to predicted mismatch error
        num_harm = np.round(peak_freq[:max_nmp] / f0_cand[i])
        num_harm = (num_harm >= 1) * num_harm + (num_harm < 1)
        freq_distance = abs(peak_freq[:max_nmp] - num_harm * f0_cand[i])
        pond_dif = freq_distance * (peak_freq[:max_nmp] ** (-p))
        mag_peaks = peak_mag[:max_nmp]
        mag_factor = 10**((mag_peaks-mag_max)/20)
        error_mp[i] = sum(mag_factor * (pond_dif + mag_factor*(q*pond_dif-r)))

    error = (error_pm[0] / max_npm) + (rho * error_mp / max_nmp)  # total error
    f0index = np.argmin(error)                                    # get the smallest error
    f0 = f0_cand[f0index]                                         # f0 with the smallest error

    return f0, error[f0index]


def peak_interp(mag_x, phase_x, peak_loc):
    """
    Interpolate peak values using parabolic interpolation
    mX, pX: magnitude and phase spectrum, ploc: locations of peaks
    returns ip_loc, ip_mag, ip_phase: interpolated peak location, magnitude and phase values
    """

    val = mag_x[peak_loc]                                              # magnitude of peak bin
    left_val = mag_x[peak_loc - 1]                                     # magnitude of bin at left
    right_val = mag_x[peak_loc + 1]                                    # magnitude of bin at right
    ip_loc = peak_loc + 0.5 * (left_val - right_val) / (left_val - 2 * val + right_val)  # center of parabola
    ip_mag = val - 0.25*(left_val-right_val)*(ip_loc - peak_loc)       # magnitude of peaks
    ip_phase = np.interp(ip_loc, np.arange(0, phase_x.size), phase_x)  # phase of peaks by linear interpolation
    return ip_loc, ip_mag, ip_phase


def dft_analysis(audio, w, fft_size):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns mag_audio_fft, phase_audio_fft: magnitude and phase spectrum
    """

    if not is_power2(fft_size):
        raise ValueError("FFT size (N) is not a power of 2")

    if w.size > fft_size:
        raise ValueError("Window size (M) is bigger than FFT size")

    pos_fft_size = (fft_size // 2) + 1                     # size of positive spectrum, it includes sample 0
    half_window_round = (w.size+1)//2                       # half analysis window size by rounding
    half_window_floor = w.size//2                           # half analysis window size by floor
    fft_buffer = np.zeros(fft_size)                         # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    audio_w = audio * w                                     # window the input sound
    fft_buffer[:half_window_round] = audio_w[half_window_floor:]  # zero-phase window in fft_buffer
    fft_buffer[-half_window_floor:] = audio_w[:half_window_floor]
    audio_fft = fft(fft_buffer)
    abs_audio_fft = abs(audio_fft[:pos_fft_size])          # compute absolute value of positive side
    abs_audio_fft[abs_audio_fft < np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mag_audio_fft = 20 * np.log10(abs_audio_fft)            # magnitude spectrum of positive frequencies in dB

    # for phase calculation set to 0 the small values
    audio_fft[:pos_fft_size].real[np.abs(audio_fft[:pos_fft_size].real) < tol] = 0.0
    audio_fft[:pos_fft_size].imag[np.abs(audio_fft[:pos_fft_size].imag) < tol] = 0.0

    phase_audio_fft = np.unwrap(np.angle(audio_fft[:pos_fft_size]))  # unwrapped phase spectrum of positive frequencies
    return mag_audio_fft, phase_audio_fft


def sinc(x, fft_size):
    """
    Generate the main lobe of a sinc function (Dirichlet kernel)
    x: array of indexes to compute; N: size of FFT to simulate
    returns y: samples of the main lobe of a sinc function
    """

    y = np.sin(fft_size * x / 2) / np.sin(x / 2)              # compute the sinc function
    y[np.isnan(y)] = fft_size                                 # avoid NaN if x == 0
    return y


def gen_bh_lobe(x):
    """
    Generate the main lobe of a Blackman-Harris window
    x: bin positions to compute (real values)
    returns y: main lobe os spectrum of a Blackman-Harris window
    """

    fft_size = 512                                                            # size of fft to use
    f = x * np.pi * 2 / fft_size                                              # frequency sampling
    df = 2 * np.pi / fft_size
    y = np.zeros(x.size)                                                      # initialize window
    consts = [0.35875, 0.48829, 0.14128, 0.01168]                             # window constants
    for m in range(0, 4):                                                     # iterate over the four sincs to sum
        y += consts[m]/2 * (sinc(f-df*m, fft_size) + sinc(f+df*m, fft_size))  # sum of scaled sinc functions
    y = y / fft_size / consts[0]                                              # normalize
    return y


def gen_spec_sines(ip_freq, ip_mag, ip_phase, fft_size, sr):
    """
    Generate a spectrum from a series of sine values
    iploc, ipmag, ipphase: sine peaks locations, magnitudes and phases
    N: size of the complex spectrum to generate; fs: sampling rate
    returns output_spectrum: generated complex spectrum of sines
    """

    output_spectrum = np.zeros(fft_size, dtype=complex)        # initialize output complex spectrum
    half_fft_size = fft_size // 2                              # size of positive freq. spectrum
    for i in range(0, ip_freq.size):                           # generate all sine spectral lobes
        loc = fft_size * ip_freq[i] / sr                       # it should be in range ]0,half_fft_size-1[
        if loc == 0 or loc > half_fft_size-1:
            continue
        bin_remainder = round(loc)-loc
        lb = np.arange(bin_remainder-4, bin_remainder+5)       # main lobe (real value) bins to read
        lobe_mag = gen_bh_lobe(lb) * 10**(ip_mag[i] / 20)      # lobe magnitudes of the complex exponential
        b = np.arange(round(loc)-4, round(loc)+5, dtype='int')
        for m in range(0, 9):
            if b[m] < 0:                                       # peak lobe crosses DC bin
                output_spectrum[-b[m]] += lobe_mag[m]*np.exp(-1j * ip_phase[i])
            elif b[m] > half_fft_size:                         # peak lobe crosses Nyquist bin
                output_spectrum[b[m]] += lobe_mag[m]*np.exp(-1j * ip_phase[i])
            elif b[m] == 0 or b[m] == half_fft_size:           # peak lobe in the limits of the spectrum
                output_spectrum[b[m]] += lobe_mag[m] * np.exp(1j * ip_phase[i]) + lobe_mag[m] * np.exp(-1j*ip_phase[i])
            else:                                              # peak lobe in positive freq. range
                output_spectrum[b[m]] += lobe_mag[m]*np.exp(1j * ip_phase[i])
        # fill the negative part of the spectrum
        output_spectrum[half_fft_size+1:] = output_spectrum[half_fft_size-1:0:-1].conjugate()
    return output_spectrum


def stochastic_model_analysis(audio, hop_size, fft_size, stoch_factor):
    """
    Stochastic analysis of a sound
    x: input array sound, H: hop size, N: fftsize
    stocf: decimation factor of mag spectrum for stochastic analysis, bigger than 0, maximum of 1
    returns stoch_env: stochastic envelope
    """

    pos_fft_size = fft_size // 2 + 1  # positive size of fft
    half_fft_size = fft_size // 2  # half of N
    if pos_fft_size * stoch_factor < 3:  # raise exception if decimation factor too small
        raise ValueError("Stochastic decimation factor too small")

    if stoch_factor > 1:  # raise exception if decimation factor too big
        raise ValueError("Stochastic decimation factor above 1")

    if hop_size <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    if not is_power2(fft_size):  # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    w = signal.windows.hann(fft_size)  # analysis window
    audio = np.append(np.zeros(half_fft_size), audio)  # add zeros at beginning to center first window at sample 0
    audio = np.append(audio, np.zeros(half_fft_size))  # add zeros at the end to analyze last sample
    p_in = half_fft_size  # initialize sound pointer in middle of analysis window
    p_end = audio.size - half_fft_size  # last sample to start a frame
    while p_in <= p_end:
        xw = audio[p_in - half_fft_size:p_in + half_fft_size] * w  # window the input sound
        audio_fft = fft(xw)  # compute FFT
        audio_fft_mag = 20 * np.log10(abs(audio_fft[:pos_fft_size]))  # magnitude spectrum of positive frequencies
        # decimate the mag spectrum
        output_mag = signal.resample(np.maximum(-200, audio_fft_mag), int(stoch_factor * pos_fft_size))
        if p_in == half_fft_size:  # first frame
            stoch_env = np.array([output_mag])
        else:  # rest of frames
            stoch_env = np.vstack((stoch_env, np.array([output_mag])))
        p_in += hop_size  # advance sound pointer
    return stoch_env


def harmonic_detection(peak_freq, peak_mag, peak_phase, f0, num_harm, harm_freq_prev, sr, harm_dev_slope=0.01):
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency
    pfreq, pmag, pphase: peak frequencies, magnitudes and phases
    f0: fundamental frequency, nH: number of harmonics,
    hfreqp: harmonic frequencies of previous frame,
    fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    returns harm_freq, harm_mag, harm_phase: harmonic frequencies, magnitudes, phases
    """

    if f0 <= 0:                                                                 # if no f0 return no harmonics
        return np.zeros(num_harm), np.zeros(num_harm), np.zeros(num_harm)
    harm_freq = np.zeros(num_harm)                                              # initialize harmonic frequencies
    harm_mag = np.zeros(num_harm) - 100                                         # initialize harmonic magnitudes
    harm_phase = np.zeros(num_harm)                                             # initialize harmonic phases
    harm_f = f0*np.arange(1, num_harm + 1)                                      # initialize harmonic frequencies
    harm_idx = 0                                                                # initialize harmonic index
    if not harm_freq_prev:  # if no incoming harmonic tracks initialize to harmonic series
        harm_freq_prev = harm_f
    while (f0 > 0) and (harm_idx < num_harm) and (harm_f[harm_idx] < sr / 2):   # find harmonic peaks
        peak_idx = np.argmin(abs(peak_freq - harm_f[harm_idx]))                 # closest peak
        dev1 = abs(peak_freq[peak_idx] - harm_f[harm_idx])                      # deviation from perfect harmonic
        # deviation from previous frame
        dev2 = (abs(peak_freq[peak_idx] - harm_freq_prev[harm_idx]) if harm_freq_prev[harm_idx] > 0 else sr)
        threshold = f0 / 3 + harm_dev_slope * peak_freq[peak_idx]
        if dev1 < threshold or dev2 < threshold:                                # accept peak if deviation is small
            harm_freq[harm_idx] = peak_freq[peak_idx]                           # harmonic frequencies
            harm_mag[harm_idx] = peak_mag[peak_idx]                             # harmonic magnitudes
            harm_phase[harm_idx] = peak_phase[peak_idx]                         # harmonic phases
        harm_idx += 1                                                           # increase harmonic index
    return harm_freq, harm_mag, harm_phase


def sine_subtraction(x, fft_size, hop_size, sfreq, smag, sphase, fs):
    """
    Subtract sinusoids from a sound
    x: input sound, N: fft-size, H: hop-size
    sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
    returns xr: residual sound
    """

    hN = fft_size // 2                                          # half of fft size
    x = np.append(np.zeros(hN),x)                      # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hN))                      # add zeros at the end to analyze last sample
    bh = signal.windows.blackmanharris(fft_size)                            # blackman harris window
    w = bh/ sum(bh)                                    # normalize window
    sw = np.zeros(fft_size)                                   # initialize synthesis window
    sw[hN - hop_size:hN + hop_size] = signal.windows.triang(2 * hop_size) / w[hN - hop_size:hN + hop_size]         # synthesis window
    L = sfreq.shape[0]                                 # number of frames, this works if no sines
    xr = np.zeros(x.size)                              # initialize output array
    pin = 0
    for l in range(L):
        xw = x[pin:pin + fft_size] * w                              # window the input sound
        X = fft(fftshift(xw))                            # compute FFT
        Yh = gen_spec_sines(fft_size * sfreq[l, :] / fs, smag[l, :], sphase[l, :], fft_size, fs)   # generate spec sines
        Xr = X-Yh                                        # subtract sines from original spectrum
        xrw = np.real(fftshift(ifft(Xr)))                # inverse FFT
        xr[pin:pin + fft_size] += xrw * sw                          # overlap-add
        pin += hop_size                                         # advance sound pointer
    xr = np.delete(xr, range(hN))                      # delete half of first window which was added in stftAnal
    xr = np.delete(xr, range(xr.size-hN, xr.size))     # delete half of last window which was added in stftAnal
    return xr


def stochastic_residual_analysis(x, fft_size, hop_size, sine_freq, sine_mag, sine_phase, sr, stoch_factor):
    """
    Subtract sinusoids from a sound and approximate the residual with an envelope
    x: input sound, N: fft size, H: hop-size
    sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
    fs: sampling rate; stocf: stochastic factor, used in the approximation
    returns stoch_env: stochastic approximation of residual
    """

    half_fft_size = fft_size // 2
    x = np.append(np.zeros(half_fft_size),x)               # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(half_fft_size))               # add zeros at the end to analyze last sample
    w = signal.windows.blackmanharris(fft_size)            # synthesis window
    w = w / sum(w)                                         # normalize synthesis window
    num_frames = sine_freq.shape[0]                        # number of frames, this works if no sines
    pin = 0
    for frame in range(num_frames):
        xw = x[pin:pin + fft_size] * w                      # window the input sound
        x_fft = fft(fftshift(xw))

        # generate spec sines
        y_harm = gen_spec_sines(fft_size * sine_freq[frame, :] / sr, sine_mag[frame, :], sine_phase[frame, :], fft_size)
        x_fft_res = x_fft-y_harm                            # subtract sines from original spectrum
        x_fft_res_mag = 20*np.log10(abs(x_fft_res[:half_fft_size]))
        # decimate the mag spectrum
        x_fft_res_mag_env = signal.resample(np.maximum(-200, x_fft_res_mag), x_fft_res_mag.size * stoch_factor)
        stoch_env = None
        if frame == 0:
            stoch_env = np.array([x_fft_res_mag_env])
        else:
            stoch_env = np.vstack((stoch_env, np.array([x_fft_res_mag_env])))
        pin += hop_size
        return stoch_env


def harmonic_detection_2(peak_freq, peak_mag, peak_phase, f0, num_harm, harm_freq_prev, sr, harm_dev_slope=0.01):
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency
    pfreq, pmag, pphase: peak frequencies, magnitudes and phases
    f0: fundamental frequency, nH: number of harmonics,
    hfreqp: harmonic frequencies of previous frame,
    fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    returns harm_freqs, harm_mag, harm_phase: harmonic frequencies, magnitudes, phases
    """

    if f0 <= 0:  # if no f0 return no harmonics
        return np.zeros(num_harm), np.zeros(num_harm), np.zeros(num_harm)

    harm_freqs = np.zeros(num_harm)                     # initialize harmonic frequencies
    harm_mag = np.zeros(num_harm) - 100                 # initialize harmonic magnitudes
    harm_phase = np.zeros(num_harm)                     # initialize harmonic phases
    harm_freq = f0*np.arange(1, num_harm + 1)           # initialize harmonic frequencies
    harm_idx = 0                                        # initialize harmonic index

    if len(harm_freq_prev) == 0:                        # if no incoming harmonic tracks initialize to harmonic series
        harm_freq_prev = harm_freq

    while (f0 > 0) and (harm_idx < num_harm) and (harm_freq[harm_idx] < sr / 2):          # find harmonic peaks
        pei = np.argmin(abs(peak_freq - harm_freq[harm_idx]))               # closest peak
        dev1 = abs(peak_freq[pei] - harm_freq[harm_idx])                    # deviation from perfect harmonic
        dev2 = (abs(peak_freq[pei] - harm_freq_prev[harm_idx]) if harm_freq_prev[harm_idx] > 0 else sr) # deviation from previous frame
        threshold = f0 / 3 + harm_dev_slope * peak_freq[pei]

        if dev1 < threshold or dev2 < threshold:         # accept peak if deviation is small
            harm_freqs[harm_idx] = peak_freq[pei]                           # harmonic frequencies
            harm_mag[harm_idx] = peak_mag[pei]                             # harmonic magnitudes
            harm_phase[harm_idx] = peak_phase[pei]                         # harmonic phases

        harm_idx += 1                                            # increase harmonic index
    return harm_freqs, harm_mag, harm_phase


def harmonic_model_analysis(
        x, sr, w, fft_size, hop_size, th, num_harm, min_f0, max_f0, f0_eth, harm_dev_slope=0.01, min_sine_dur=.02):
    """
    Analysis of a sound using the sinusoidal harmonic model
    x: input sound; fs: sampling rate, w: analysis window; N: FFT size (minimum 512); t: threshold in negative dB,
    nH: maximum number of harmonics;  minf0: minimum f0 frequency in Hz,
    maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
    harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
    returns xhfreq, x_harm_mag, xhphase: harmonic frequencies, magnitudes and phases
    """

    if min_sine_dur < 0:  # raise exception if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    pos_fft_size = fft_size // 2  # size of positive spectrum
    half_fft_round = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    half_fft_floor = int(math.floor(w.size / 2))  # half analysis window size by floor
    x = np.append(np.zeros(half_fft_floor), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(half_fft_floor))  # add zeros at the end to analyze last sample
    p_in = half_fft_round  # init sound pointer in middle of anal window
    p_end = x.size - half_fft_round  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    harm_freq_prev = []  # initialize harmonic frequencies of previous frame
    f0_stable = 0  # initialize f0 stable
    while p_in <= p_end:
        frame = x[p_in - half_fft_round:p_in + half_fft_floor]  # select frame
        mag_fft, phase_fft = dft_analysis(frame, w, fft_size)  # compute dft
        peak_loc = peak_detection(mag_fft, th)  # detect peak locations
        ip_loc, ip_mag, ip_phase = peak_interp(mag_fft, phase_fft, peak_loc)  # refine peak values
        ip_freq = sr * ip_loc / fft_size  # convert locations to Hz
        f0_track = f0_twm(ip_freq, ip_mag, f0_eth, min_f0, max_f0, f0_stable)  # find f0
        if ((f0_stable == 0) and (f0_track > 0)) \
                or ((f0_stable > 0) and (np.abs(f0_stable - f0_track) < f0_stable / 5.0)):
            f0_stable = f0_track  # consider a stable f0 if it is close to the previous one
        else:
            f0_stable = 0
        harm_freq, harm_mag, harm_phase = \
            harmonic_detection_2(ip_freq, ip_mag, ip_phase, f0_track, num_harm, harm_freq_prev, sr, harm_dev_slope)
        harm_freq_prev = harm_freq
        if p_in == half_fft_round:  # first frame
            x_harm_freq = np.array([harm_freq])
            x_harm_mag = np.array([harm_mag])
            x_harm_phase = np.array([harm_phase])
        else:  # next frames
            x_harm_freq = np.vstack((x_harm_freq, np.array([harm_freq])))
            x_harm_mag = np.vstack((x_harm_mag, np.array([harm_mag])))
            x_harm_phase = np.vstack((x_harm_phase, np.array([harm_phase])))
        p_in += hop_size  # advance sound pointer
    # delete tracks shorter than minSineDur
    x_harm_freq = cleaning_sine_tracks(x_harm_freq, round(sr * min_sine_dur / hop_size))
    return x_harm_freq, x_harm_mag, x_harm_phase


def fft_extractor(
        audio, sr=44100, w_size=2001, hop_size=2 * 256, th=-90, num_harm=5, min_f0=50, max_f0=1500, f0_eth=2,
        harm_dev_slope=0.01, min_sine_dur=0.01, num_sines=2048, stoch_factor=0.3):
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
    x: input sound, fs: sampling rate, w: analysis window; fft_size: FFT size, t: threshold in negative dB, 
    nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
    maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
    harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
    returns hfreq, h_mag, h_phase: harmonic frequencies, magnitude and phases; stoch_env: stochastic residual
    """
    fft_size = int(2 ** np.ceil(np.log2(w_size)))
    w = signal.windows.blackmanharris(w_size)

    # perform harmonic analysis
    h_freq, h_mag, h_phase = \
        harmonic_model_analysis(audio, sr, w, fft_size, hop_size, th, num_harm,
                                min_f0, max_f0, f0_eth, harm_dev_slope, min_sine_dur)
    # subtract sinusoids from original sound
    x_res = sine_subtraction(audio, num_sines, hop_size, h_freq, h_mag, h_phase, sr)
    # perform stochastic analysis of residual
    stoch_env = stochastic_model_analysis(x_res, hop_size, hop_size * 2, stoch_factor)

    # create figure to plot
    #plt.figure(figsize=(9, 6))

    # frequency range to plot
    max_plot_freq = 2000.0

    # plot spectrogram stochastic component
    # plt.subplot(3, 1, 2)
    num_frames = int(stoch_env[:, 0].size)
    size_env = int(stoch_env[0, :].size)
    # frmTime = hop_size * np.arange(num_frames) / float(sr)
    # binFreq = (.5 * sr) * np.arange(size_env * max_plot_freq / (.5 * sr)) / size_env
    # plt.pcolormesh(frmTime, binFreq, np.transpose(stoch_env[:, :int(size_env * max_plot_freq / (.5 * sr) + 1)]))
    # plt.autoscale(tight=True)

    # plot harmonic on top of stochastic spectrogram
    if h_freq.shape[1] > 0:
        harms = h_freq * np.less(h_freq, max_plot_freq)
        harms[harms == np.nan] = 0
        num_frames = harms.shape[0]
        # frmTime = hop_size * np.arange(num_frames) / float(sr)

        """plt.plot(frmTime, harms[:,0], color='k', ms=3, alpha=1)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.autoscale(tight=True)
        plt.title(f"harmonics + stochastic spectrogram")"""

    # plt.tight_layout()
    # plt.show()

    return harms[:, 0]
