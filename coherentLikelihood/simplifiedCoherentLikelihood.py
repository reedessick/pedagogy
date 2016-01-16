#!/usr/bin/python
usage = "simplifiedCoherentLikelihood.py [--options]"
description = "builds figures to demonstrate a heuristic burst search"
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import waveforms

import numpy as np

import subprocess as sp

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from optparse import OptionParser

#-------------------------------------------------

parser = OptionParser(usage=usage, description=description)

parser.add_option('-v', '--verbose', default=False, action='store_true')

parser.add_option('-T', '--duration', default=10.0, type='float', help='duration of the experiment')
parser.add_option('-s', '--sampling-rate', default=1024, type='int', help='sampling rate of the experiment, should be a power of 2')

parser.add_option('-S', '--SNR', default=15.0, type='float', help='requested SNR for the injection')

parser.add_option('', '--theta', default=45, type='float', help='the polar angle for triangulation. WARNING: the plot shows "theta" but that is measured from the zenith!')
parser.add_option('', '--D-over-c', default=3, type='float', help='the triangulation baseline')

parser.add_option('-f', '--freq', default=10.0, type='float', help='central frequency of the chirpedSineGaussian')
parser.add_option('-F', '--freqDot', default=20, type='float', help='frequency derivative of the chirpedSineGaussian')
parser.add_option('-t', '--tau', default=0.25, type='float', help='time constnat of the chirpedSineGaussian')

parser.add_option('', '--frames-per-sec', default=30, type='int', help='the number of frames per second of the movie')
parser.add_option('', '--num-frames', default=200, type='int', help='the total number of frames in the movie')

parser.add_option('', '--hide-signal', default=False, action='store_true', help='do not show signal in fame*png figures')
parser.add_option('', '--hide-noisy-reconstruction', default=False, action='store_true', help='do not show the reconstructed signal which contains noise')
parser.add_option('', '--hide-noiseless-reconstruction', default=False, action='store_true', help='do not show the reconstructed signal which contains only injections')

parser.add_option('', '--tag', default='', type='string' )
parser.add_option('', '--dpi', default=200, type='int' )
parser.add_option('', '--movie-type', default='mpg', type='string')

parser.add_option('', '--sanity-check', default=False, action='store_true', help='stop after making sanity check plots')

opts, args = parser.parse_args()

if opts.tag:
    opts.tag = "_%s"%opts.tag

N = opts.duration*opts.sampling_rate
if N%2:
    raise ValueError("must have an even number of sample points! %.3f*%.3f=%.3f"%(opts.duration, opts.sampling_rate, N))

#-------------------------------------------------

if opts.verbose:
    print "generating white noise (in the freq domain)"
(freqs, wFreqDom1), (times, wTimeDom1) =  waveforms.whiteNoise( opts.duration, opts.sampling_rate )
(freqs, wFreqDom2), (times, wTimeDom2) =  waveforms.whiteNoise( opts.duration, opts.sampling_rate )

#-------------------------------------------------

dt = opts.D_over_c * np.cos( opts.theta*np.pi/180 )
to = max(opts.duration-opts.D_over_c-3*opts.tau, opts.D_over_c + (opts.duration-2*opts.D_over_c)/2)

if opts.verbose:
    print "generating injection with to=%.3f"%(to)
hTimeDom1 = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, to )
hFreqDom1 = waveforms.chirpSineGaussianF( freqs, 1.0, opts.freq, opts.freqDot, opts.tau, to )

hTimeDom2 = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, to-dt )
hFreqDom2 = waveforms.chirpSineGaussianF( freqs, 1.0, opts.freq, opts.freqDot, opts.tau, to-dt )

#-------------------------------------------------

if opts.verbose:
    print "computing optimal SNR and scaling injection"
### for white-gaussian noise with unit-variance in the frequency domain
snr = ( 2 * np.sum( hFreqDom1.real**2 + hFreqDom1.imag**2 + hFreqDom2.real**2 + hFreqDom2.imag**2 ) / opts.duration )**0.5

scaling = opts.SNR/snr
hTimeDom1 *= scaling
hFreqDom1 *= scaling
hTimeDom2 *= scaling
hFreqDom2 *= scaling

#-------------------------------------------------

if opts.verbose:
    print "compute logBSN as a function of theta"

dataF1 = wFreqDom1 + hFreqDom1
dataT1 = wTimeDom1 + hTimeDom1

dataF2 = wFreqDom2 + hFreqDom2
dataT2 = wTimeDom2 + hTimeDom2

ylim = 1.1*max(np.max(np.abs(dataT2)), np.max(np.abs(dataT1)))
ylim = (-ylim, ylim)

#snr = 2 * np.sum( dataF1.real**2 + dataF1.imag**2 + dataF2.real**2 + dataF2.imag**2 ) / opts.duration + np.fft.ifft( 2 * np.fft.ifftshift( dataF1 * np.conj(dataF2) ) ).real * opts.sampling_rate ### ifft normalizes the sum by 1/n = 1/(s*T) and we want to normalize by 1/T to approximate the integral
#SNR = snr**0.5 ### this is the "coherent snr"

SNR = np.fft.ifft( 2 * np.fft.ifftshift( dataF1 * np.conj(dataF2) ) ).real * opts.sampling_rate ### ifft normalizes the sum by 1/n = 1/(s*T) and we want to normalize by 1/T to approximate the integral

#-------------------------------------------------

if opts.verbose:
    print "plotting sanity check of injection and noise"

fig = plt.figure(figsize=(7,13))

### IFO1 raw data
ax = plt.subplot(3,1,1)

ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 1}$', ha='left', va='top', fontsize=20)

ax.plot( times, dataT1, 'b-', linewidth=1, alpha=0.50, label='$\mathrm{noise_1+signal_1}$' )

if not opts.hide_signal:
    ax.plot( times, hTimeDom1, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{signal_1}$' )

ax.legend(loc='lower left')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$d_1(t)$')

ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
ax.set_ylim(ylim)

### IFO2 raw data
ax = plt.subplot(3,1,2)

ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 2}$', ha='left', va='top', fontsize=20)

ax.plot( times+dt, dataT2, 'r-', linewidth=1, alpha=0.50, label='$\mathrm{shifted\ noise_2+signal_2}$' )

if not opts.hide_signal:
    ax.plot( times+dt, hTimeDom2, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ signal_2}$' )

plt.annotate(s='', xy=(to-dt, np.max(dataT2)), xytext=(to,np.max(dataT2)), arrowprops=dict(arrowstyle='<-'))

ax.legend(loc='lower left')
#ax.set_xlabel('$\mathrm{time}$')
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel('$d_2(t)$')

ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
ax.set_ylim(ylim)

### ray-plot
ax = plt.subplot(3,1,3)

ax.plot( times, SNR, 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )
ax.plot( -times, SNR, 'g-', linewidth=1, alpha=0.5, label=None )

ylim_ray = ax.get_ylim()
ax.plot( [dt]*2, ylim_ray, 'k--', linewidth=1, alpha=0.5 )
ax.set_ylim(ylim_ray)

ax.set_xlabel('$\Delta t$')
ax.set_ylabel('$\mathrm{correlated\ Energy}\sim\\rho^2$')

ax.set_xlim(xmin=opts.D_over_c-(to-dt), xmax=opts.duration-opts.D_over_c-(to-dt))

plt.subplots_adjust(hspace=0.1, wspace=0.1)

figname = "simplifiedSanityCheck%s.png"%(opts.tag)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close( fig )

if opts.sanity_check:
    import sys
    sys.exit(0)

#-------------------------------------------------

if opts.verbose:
    print "making movie frames"

shifts = np.arange(opts.D_over_c-(to-dt), opts.duration-opts.D_over_c-(to-dt), 1.0/opts.sampling_rate)
N = len(shifts)
frame_step = int( 1.0*N / opts.num_frames )

frameNo = 0

### plot an openning frame
fig = plt.figure(figsize=(7,13))

### IFO1 raw data
ax = plt.subplot(3,1,1)

ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 1}$', ha='left', va='top', fontsize=20)

ax.plot( times, dataT1, 'b-', linewidth=1, alpha=0.50, label='$\mathrm{noise_1+signal_1}$' )

if not opts.hide_signal:
    ax.plot( times, hTimeDom1, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{signal_1}$' )

ax.legend(loc='lower left')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$d_1(t)$')

ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
ax.set_ylim(ylim)

### IFO2 raw data
ax = plt.subplot(3,1,2)

ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 2}$', ha='left', va='top', fontsize=20)

ax.plot( times+shifts[0], dataT2, 'r-', linewidth=1, alpha=0.50, label='$\mathrm{shifted\ noise_2+signal_2}$' )

if not opts.hide_signal:
    ax.plot( times+shifts[0], hTimeDom2, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ signal_2}$' )

plt.annotate(s='', xy=(to-dt,np.max(dataT2)), xytext=(to-dt+shifts[0],np.max(dataT2)), arrowprops=dict(arrowstyle='<-'))

ax.legend(loc='lower left')
#ax.set_xlabel('$\mathrm{time}$')
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel('$d_2(t)$')

ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
ax.set_ylim(ylim)

### ray-plot
ax = plt.subplot(3,1,3)

#ax.plot( times, SNR, 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )
#ax.plot( -times, SNR, 'g-', linewidth=1, alpha=0.5, label=None )

#ax.legend(loc='best')
ax.set_xlabel('$\Delta t$')
#ax.set_ylabel('$\\rho(\\tau)$')
ax.set_ylabel('$\mathrm{correlated\ Energy}\sim\\rho^2$')

ax.set_xlim(xmin=opts.D_over_c-(to-dt), xmax=opts.duration-opts.D_over_c-(to-dt))
ax.set_ylim(ymin=1.1*np.min(SNR), ymax=1.1*np.max(SNR))

plt.subplots_adjust(hspace=0.1, wspace=0.1)

figname = "simplifiedFrame%s-%04d.png"%(opts.tag, frameNo)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close(fig)

frameNo += 1

### plot the rest of the frames
ind = 0
while ind < N:

    shift = shifts[ind]

    fig = plt.figure(figsize=(7,13))

    ### IFO1 raw data
    ax = plt.subplot(3,1,1)

    ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 1}$', ha='left', va='top', fontsize=20)

    ax.plot( times, dataT1, 'b-', linewidth=1, alpha=0.50, label='$\mathrm{noise_1+signal_1}$' )

    if not opts.hide_signal:
        ax.plot( times, hTimeDom1, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{signal_1}$' )

    ax.legend(loc='lower left')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('$\mathrm{time}$')
    ax.set_ylabel('$d_1(t)$')

    ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
    ax.set_ylim(ylim)

    ### IFO2 raw data
    ax = plt.subplot(3,1,2)

    ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 2}$', ha='left', va='top', fontsize=20)

    ax.plot( times+shift, dataT2, 'r-', linewidth=1, alpha=0.50, label='$\mathrm{shifted\ noise_2+signal_2}$' )

    if not opts.hide_signal:
        ax.plot( times+shift, hTimeDom2, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ signal_2}$' )

    if shift:
        plt.annotate(s='', xy=(to-dt,np.max(dataT2)), xytext=(to-dt+shift,np.max(dataT2)), arrowprops=dict(arrowstyle='<-'))

    ax.legend(loc='lower left')
#    ax.set_xlabel('$\mathrm{time}$')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$d_2(t)$')

    ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
    ax.set_ylim(ylim)

    ### ray-plot
    ax = plt.subplot(3,1,3)

    truth = -times <= shift
    ax.plot( -times[truth], SNR[truth], 'g-', linewidth=1, alpha=0.5, label=None )
    truth = times <= shift
    ax.plot( times[truth], SNR[truth], 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )

#    ax.legend(loc='best')
    ax.set_xlabel('$\Delta t$')
#    ax.set_ylabel('$\\rho(\\tau)$')
    ax.set_ylabel('$\mathrm{correlated\ Energy}\sim\\rho^2$')

#    ax.set_xlim(xmin=-opts.D_over_c, xmax=opts.D_over_c)
    ax.set_xlim(xmin=opts.D_over_c-(to-dt), xmax=opts.duration-opts.D_over_c-(to-dt))
    ax.set_ylim(ymin=1.1*np.min(SNR), ymax=1.1*np.max(SNR))

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    figname = "simplifiedFrame%s-%04d.png"%(opts.tag, frameNo)
    if opts.verbose:
        print "    %s"%figname
    fig.savefig( figname, dpi=opts.dpi )
    plt.close(fig)

    frameNo += 1
    ind += frame_step

### plot the final frame

shift = shifts[-1]
ind = N

fig = plt.figure(figsize=(15,10))

### IFO1 raw data
ax = plt.subplot(3,1,1)

ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 1}$', ha='left', va='top', fontsize=20)

ax.plot( times, dataT1, 'b-', linewidth=1, alpha=0.50, label='$\mathrm{noise_1+signal_1}$' )

if not opts.hide_signal:
    ax.plot( times, hTimeDom1, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{signal_1}$' )

ax.legend(loc='lower left')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$d_1(t)$')

ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
ax.set_ylim(ylim)

### IFO2 raw data
ax = plt.subplot(3,1,2)

ax.text(opts.D_over_c+0.01*opts.duration, 0.9*ylim[1], '$\mathrm{IFO\ 2}$', ha='left', va='top', fontsize=20)

ax.plot( times+shift, dataT2, 'r-', linewidth=1, alpha=0.50, label='$\mathrm{shifted\ noise_2+signal_2}$' )

if not opts.hide_signal:
    ax.plot( times+shift, hTimeDom2, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ signal_2}$' )

plt.annotate(s='', xy=(to-dt,np.max(dataT2)), xytext=(to-dt+shift,np.max(dataT2)), arrowprops=dict(arrowstyle='<-'))

ax.legend(loc='lower left')
#ax.set_xlabel('$\mathrm{time}$')
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel('$d_2(t)$')

ax.set_xlim(xmin=opts.D_over_c, xmax=opts.duration-opts.D_over_c)
ax.set_ylim(ylim)

### ray-plot
ax = plt.subplot(3,1,3)

truth = -times <= shift
ax.plot( -times[truth], SNR[truth], 'g-', linewidth=1, alpha=0.5, label=None )
truth = times <= shift
ax.plot( times[truth], SNR[truth], 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )

#truth = times[-1]-times < shift
#ax.plot( times[truth]-times[-1], SNR[truth], 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )

#ax.legend(loc='best')
ax.set_xlabel('$\Delta t$')
#ax.set_ylabel('$\\rho(\\tau)$')
ax.set_ylabel('$\mathrm{correlated\ Energy}\sim\\rho^2$')

ax.set_xlim(xmin=opts.D_over_c-(to-dt), xmax=opts.duration-opts.D_over_c-(to-dt))
ax.set_ylim(ymin=1.1*np.min(SNR), ymax=1.1*np.max(SNR))

plt.subplots_adjust(hspace=0.1, wspace=0.1)

figname = "simplifiedFrame%s-%04d.png"%(opts.tag, frameNo)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close(fig)

#-------------------------------------------------

cmd = "ffmpeg -r %d -i simplifiedFrame%s-%s04d.png simplifiedCoherentLikelihood%s.%s"%(opts.frames_per_sec, opts.tag, "%", opts.tag, opts.movie_type)
if opts.verbose:
    print "wrapping into a movie:\n\t%s"%(cmd)

sp.Popen(cmd.split()).wait()
