description = "a module for building waveforms for pedagogical figures"
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def whiteNoiseT( T, s ):
    """
    generates unit-variance white noise with duration (T) and sampling rate (s)
    unit-variance in frequency domain
    """
    N = T*s
    if N%2:
        raise ValueError("must be an even number of sample points!")
    times = np.arange(0, T, 1.0/s)
    w = np.random.normal(loc=0, scale=1, size=T*s)
    w /= (T/s)**0.5 ### normalization to get the freq-series to be unit-variance

    return times, w

def whiteNoiseF( T, s ):
    """
    generates unit-variance white noise with duration (T) and sampling rate (s)
    unit-varience in frequency domain
    """
    N = T*s
    if N%2:
        raise ValueError("must be an even number of sample points!")
    n = N/2
    ### frequencies
    freqs = np.arange(-s/2.0, s/2.0, 1.0/T)
    ### generate one-sided FreqDoman representation
    w = np.empty( N )
    p = np.empty( N )

    w[0] = w[n] = np.random.normal( loc=0, scale=1, size=1 )
    w[1:n] = np.random.normal( loc=0, scale=1, size=n-1 )
    w[n+1:] = w[1:n][::-1]

    p[0] = p[n] = 0
    p[1:n] = 2*np.pi*np.random.rand( n-1 )
    p[n+1:] = -p[1:n][::-1]

    return freqs, w*np.exp( -1j*p )

def whiteNoise( T, s ):
    """
    generates unit-variance white noise with duration (T) and sampling rate (s)

    returns the white noise in the Frequency Domain (one-sided) and in the Time Domain
    """
    ### generate one-sided FreqDoman representation
    freqs, wFreqDomain = whiteNoiseF( T, s )

    ### generate TimeDomain representation
    wTimeDomain = np.fft.ifft( np.fft.ifftshift(wFreqDomain) ).real ### truncate hanging imaginary numbers
    wTimeDomain *= s ### ifft normalizes the sum by 1/n = 1/(s*T) and we want to normalize by 1/T to approximate the integral
    times = np.arange(0, T, 1.0/s)

    return (freqs, wFreqDomain), (times, wTimeDomain)

#-------------------------------------------------

def sineGaussianT( t, A, fo, tau, to ):
    """
    A * sin(2*pi*fo*(t-to)) * exp( - ((t-to)/tau)**2 )
    """
    return A * np.sin(2*np.pi*fo*(t-to)) * np.exp( - ((t-to)/tau)**2 )

def sineGaussianF( f, A, fo, tau, to ):
    """
    0.5 * pi**0.5 * A *  tau * exp( - (pi*(f-fo)*tau)**2 - 2*pi*i*f*to )
    """
    return 0.5 * np.pi**0.5 * A * tau * np.exp( - (np.pi*(f-fo)*tau)**2 - 2*np.pi*1j*f*to )

def chirpSineGaussianT( t, A, fo, fdot, tau, to ):
    """
    A * sin(2*pi*(fo+fdot*(t-to))*(t-to)) * exp( - ((t-to)/tau)**2 )
    """
    return A * np.sin(2*np.pi*(fo+fdot*(t-to))*(t-to)) * np.exp( - ((t-to)/tau)**2 )

def chirpSineGaussianF( f, A, fo, fdot, tau, to ):
    """
    0.5 * A * tau * ( pi / (1-2*pi*i*fdot*tau**2) )**0.5 * exp( - (pi*(f-fo)*tau)**2/(1-2*pi*i*fdot*tau**2) - 2*pi*i*fo*tau )
    """
    x = 1-2*np.pi*1j*fdot*tau**2
    return 0.5 * A * tau * ( np.pi/x )**0.5 * np.exp( - (np.pi*(f-fo)*tau)**2/x - 2*np.pi*1j*f*to )

