#!/usr/bin/python
usage = "matchedFilter.py [--options]"
description = "builds figures to demonstrate matched filtering"
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

parser.add_option('-T', '--duration', default=5.0, type='float', help='duration of the experiment')
parser.add_option('-s', '--sampling-rate', default=1024, type='int', help='sampling rate of the experiment, should be a power of 2')

parser.add_option('-S', '--SNR', default=10.0, type='float', help='requested SNR for the injection')

parser.add_option('-f', '--freq', default=300.0, type='float', help='central frequency of the chirpedSineGaussian')
parser.add_option('-F', '--freqDot', default=10.0, type='float', help='frequency derivative of the chirpedSineGaussian')
parser.add_option('-t', '--tau', default=0.5, type='float', help='time constnat of the chirpedSineGaussian')

parser.add_option('', '--frames-per-sec', default=30, type='int', help='the number of frames per second of the movie')
parser.add_option('', '--num-frames', default=300, type='int', help='the total number of frames in the movie')

parser.add_option('', '--hide-signal', default=False, action='store_true', help='do not show signal in fame*png figures')
parser.add_option('', '--tag', default='', type='string' )
parser.add_option('', '--dpi', default=200, type='int' )

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
(freqs, wFreqDom), (times, wTimeDom) =  waveforms.whiteNoise( opts.duration, opts.sampling_rate )

# if opts.verbose:
#     print "generating white noise (in the time domain)"
# timesT, wTimeDomT = waveforms.whiteNoiseT( opts.duration, opts.sampling_rate )
# wFreqDomT = np.fft.fftshift( np.fft.fft( wTimeDomT ) )/opts.sampling_rate

#-------------------------------------------------

to = max(opts.duration-3*opts.tau, opts.duration/2)
if opts.verbose:
    print "generating injection with to=%.3f"%(to)
hTimeDom = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, to )
hFreqDom = waveforms.chirpSineGaussianF( freqs, 1.0, opts.freq, opts.freqDot, opts.tau, to )

#-------------------------------------------------

if opts.verbose:
    print "computing optimal SNR and scaling injection"
### for white-gaussian noise with unit-variance in the frequency domain
snr = ( 2 * np.sum( hFreqDom.real**2 + hFreqDom.imag**2 ) / opts.duration )**0.5

scaling = opts.SNR/snr
hTimeDom *= scaling
hFreqDom *= scaling

#-------------------------------------------------

if opts.verbose:
    print "computing SNR(t)"
data  = hFreqDom + wFreqDom
#dataT = hFreqDom + wFreqDomT

### set up template
tFreqDom = waveforms.chirpSineGaussianF( freqs, 1.0, opts.freq, opts.freqDot, opts.tau, 0 ) ### template with to=0
tFreqDom /= ( 2 * np.sum( tFreqDom.real**2 + tFreqDom.imag**2 ) / opts.duration )**0.5 ### normalize the template

### compute snr(t)
snr  = np.fft.ifft( 2 * np.fft.ifftshift( data * np.conj(tFreqDom) ) ) * opts.sampling_rate ### normalization...
SNR = np.abs(snr)

#snrT = np.fft.ifft( 2 * np.fft.ifftshift( dataT * np.conj(tFreqDom) ) ) * opts.sampling_rate
#SNRT = np.abs(snrT)

# if opts.verbose:
#     print "computing SNR(t) the hard way (in time domain)"
#
# tdata = hTimeDom + wTimeDom
# tdataT = hTimeDom + wTimeDomT
#
# tsnr = np.empty_like( tdata )
# tsnrT = np.empty_like( tdataT )
#
# ### iterate and compute convolution by hand
# for ind, t in enumerate(times):
#     template = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, t ) ### template with to=t
#     template /= ( np.sum( template**2 ) / opts.sampling_rate )**0.5 ### normalize template
# 
#     tsnr[ind]  = np.abs( np.sum( template*tdata ) / opts.sampling_rate )
#     tsnrT[ind] = np.abs( np.sum( template*tdataT ) / opts.sampling_rate )

#-------------------------------------------------

if opts.verbose:
    print "plotting sanity check of injection and noise in Time and Freq Domains"

### time domain

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(1,2,1)

ax.plot( times, wTimeDom+hTimeDom,  'r-', linewidth=1, alpha=0.5, label='$\mathrm{noise+signal}$' )
#ax.plot( times, wTimeDomT+hTimeDom, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{noiseT+signal}$' )

ax.plot( times, hTimeDom, 'k-', linewidth=2, alpha=0.75, label='$\mathrm{signal}$' )

ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$h(t)$')
ax.legend(loc='best')

ax.set_xlim(xmin=0, xmax=opts.duration)

### freq domain

ax = plt.subplot(1,2,2)

ax.plot( freqs, (wFreqDom.real+hFreqDom.real)**2 + (wFreqDom.imag+hFreqDom.imag)**2,  'r-', linewidth=1, alpha=0.5, label='$\mathrm{noise+signal}$' )
#ax.plot( freqs, (wFreqDomT.real+hFreqDom.real)**2 + (wFreqDomT.imag+hFreqDom.imag)**2, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{noiseT+signal}$' )

ax.plot( freqs, hFreqDom.real**2 + hFreqDom.imag**2, 'k-', linewidth=2, alpha=0.75, label='$\mathrm{signal}$' )

ax.set_xlabel('$\mathrm{freq}$')
ax.set_ylabel('$\\tilde{h}(f)$')
ax.legend(loc='best')

ax.set_xlim(xmin=0, xmax=opts.sampling_rate/2)

ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

plt.subplots_adjust(hspace=0.05, wspace=0.05)

figname = "sanityCheck%s.png"%(opts.tag)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close(fig)

#-------------------------------------------------

if opts.verbose:
    print "plotting sanity check of SNR(t)"

fig = plt.figure(figsize=(5,10))

### noise generated in freq-domain
ax = plt.subplot(2,1,1)

ax.plot( times, hTimeDom + wTimeDom, 'r-', linewidth=1, alpha=0.5, label='$\mathrm{noise+signal}$' )
#ax.plot( times, hTimeDom + wTimeDomT, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{noiseT+signal}$' )

ax.plot( times, hTimeDom, 'k-', linewidth=1, alpha=0.75, label='$\mathrm{signal}$' )

ax.legend(loc='best')
ax.set_xlabel('$\mathrm{time}$')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('$h(t)$')

ax.set_xlim(xmin=0, xmax=opts.duration)

### noise generated in time-domain
ax = plt.subplot(2,1,2)

ax.plot( times, SNR,  'r-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )
#ax.plot( times, SNRT, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )

#ax.plot( times, tsnr,  'm-', linewidth=1, alpha=0.5, label='$\mathrm{time-domain}$\n$\mathrm{computation}$' )
#ax.plot( times, tsnrT, 'c-', linewidth=1, alpha=0.5, label='$\mathrm{time-domain}$\n$\mathrm{computation}$' )

ax.legend(loc='best')
ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$\\rho(t)$')

ax.set_xlim(xmin=0, xmax=opts.duration)

ylim = ax.get_ylim()
ax.plot([to]*2, ylim, 'k--', alpha=0.5 )
ax.set_ylim(ylim)

plt.subplots_adjust(hspace=0.05, wspace=0.05)

figname = "sanityCheckSNR%s.png"%(opts.tag)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close(fig)

#-------------------------------------------------

if opts.sanity_check:
    import sys
    sys.exit(0)

#-------------------------------------------------

if opts.verbose:
    print "making movie frames"

N = opts.duration*opts.sampling_rate
frame_step = int( 1.0*N / opts.num_frames )

frameNo = 0

fig = plt.figure(figsize=(5,10))
ax = plt.subplot(2,1,1)

ax.plot( times, hTimeDom + wTimeDom, 'r-', linewidth=1, alpha=0.5, label='$\mathrm{noise+signal}$' )
if not opts.hide_signal:
    ax.plot( times,  hTimeDom, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{signal}$' )

ax.set_xlabel('$\mathrm{time}$')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('$h(t)$')

ax.set_xlim(xmin=0, xmax=opts.duration)

ax = plt.subplot(2,1,2)

ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$\\rho(t)$')

ax.set_xlim(xmin=0, xmax=opts.duration)
ax.set_ylim(ymin=0, ymax=1.1*np.max(SNR))

plt.subplots_adjust(hspace=0.05, wspace=0.05)

figname = "frame%s-%04d.png"%(opts.tag, frameNo)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close( fig )

frameNo += 1

ind = 0
while ind < N:
    fig = plt.figure(figsize=(5,10))
    ax = plt.subplot(2,1,1)

    ax.plot( times, hTimeDom + wTimeDom, 'r-', linewidth=1, alpha=0.5, label='$\mathrm{noise+signal}$' )
    if not opts.hide_signal:
        ax.plot( times,  hTimeDom, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{signal}$' )

    template = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, times[ind] ) ### template with to=t
    template /= ( np.sum( template**2 ) / opts.sampling_rate )**0.5 ### normalize template

    ax.plot( times, template, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{template}$')

    ylim = ax.get_ylim()
    ax.arrow( times[ind], 40, 0, -20, head_width=0.1, head_length=5.0, fc='k', ec='k' )

    ax.set_xlabel('$\mathrm{time}$')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('$h(t)$')

    ax.set_xlim(xmin=0, xmax=opts.duration)

    ax = plt.subplot(2,1,2)

    ax.plot( times[:ind], SNR[:ind], 'k-', linewidth=1, alpha=0.5, label='$\mathrm{time-domain}$\n$\mathrm{computation}$' )
    
    ax.set_xlabel('$\mathrm{time}$')
    ax.set_ylabel('$\\rho(t)$')

    ax.set_xlim(xmin=0, xmax=opts.duration)
    ax.set_ylim(ymin=0, ymax=1.1*np.max(SNR))

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    figname = "frame%s-%04d.png"%(opts.tag, frameNo)
    if opts.verbose:
        print "    %s"%figname
    fig.savefig( figname, dpi=opts.dpi )
    plt.close( fig )

    frameNo += 1
    ind += frame_step
    
fig = plt.figure(figsize=(5,10))
ax = plt.subplot(2,1,1)

ax.plot( times, hTimeDom + wTimeDom, 'r-', linewidth=1, alpha=0.5, label='$\mathrm{noise+signal}$' )
if not opts.hide_signal:
    ax.plot( times,  hTimeDom, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{signal}$' )

template = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, times[-1] ) ### template with to=t
template /= ( np.sum( template**2 ) / opts.sampling_rate )**0.5 ### normalize template

ax.plot( times, template, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{template}$')

ylim = ax.get_ylim()
ax.arrow( times[-1], 40, 0, -20, head_width=0.1, head_length=5.0, fc='k', ec='k' )

ax.set_xlabel('$\mathrm{time}$')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('$h(t)$')

ax.set_xlim(xmin=0, xmax=opts.duration)

ax = plt.subplot(2,1,2)

ax.plot( times, SNR, 'k-', linewidth=1, alpha=0.5, label='$\mathrm{time-domain}$\n$\mathrm{computation}$' )

ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$\\rho(t)$')

ax.set_xlim(xmin=0, xmax=opts.duration)
ax.set_ylim(ymin=0, ymax=1.1*np.max(SNR))

plt.subplots_adjust(hspace=0.05, wspace=0.05)

figname = "frame%s-%04d.png"%(opts.tag, frameNo)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname, dpi=opts.dpi )
plt.close( fig )

#-------------------------------------------------

cmd = "ffmpeg -r %d -i frame%s-%s04d.png matchedFilter%s.mp4"%(opts.frames_per_sec, opts.tag, "%", opts.tag)
if opts.verbose:
    print "wrapping into a movie:\n\t%s"%(cmd)

sp.Popen(cmd.split()).wait()
