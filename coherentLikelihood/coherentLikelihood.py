#!/usr/bin/python
usage = "coherentLikelihood.py [--options]"
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

parser.add_option('-T', '--duration', default=5.0, type='float', help='duration of the experiment')
parser.add_option('-s', '--sampling-rate', default=1024, type='int', help='sampling rate of the experiment, should be a power of 2')

parser.add_option('-S', '--SNR', default=25.0, type='float', help='requested SNR for the injection')

parser.add_option('', '--theta', default=np.pi/4, type='float', help='the polar angle for triangulation')
parser.add_option('', '--D-over-c', default=1.5, type='float', help='the triangulation baseline')

parser.add_option('-f', '--freq', default=10.0, type='float', help='central frequency of the chirpedSineGaussian')
parser.add_option('-F', '--freqDot', default=8.0, type='float', help='frequency derivative of the chirpedSineGaussian')
parser.add_option('-t', '--tau', default=0.5, type='float', help='time constnat of the chirpedSineGaussian')

parser.add_option('', '--frames-per-sec', default=30, type='int', help='the number of frames per second of the movie')
parser.add_option('', '--num-frames', default=300, type='int', help='the total number of frames in the movie')

parser.add_option('', '--hide-signal', default=False, action='store_true', help='do not show signal in fame*png figures')
parser.add_option('', '--tag', default='', type='string' )

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

dt = opts.D_over_c * np.cos( opts.theta )
to = opts.duration/2

if opts.verbose:
    print "generating injection with to=%.3f"%(to)
hTimeDom1 = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, to+dt/2 )
hFreqDom1 = waveforms.chirpSineGaussianF( freqs, 1.0, opts.freq, opts.freqDot, opts.tau, to+dt/2 )

hTimeDom2 = waveforms.chirpSineGaussianT( times, 1.0, opts.freq, opts.freqDot, opts.tau, to-dt/2 )
hFreqDom2 = waveforms.chirpSineGaussianF( freqs, 1.0, opts.freq, opts.freqDot, opts.tau, to-dt/2 )

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

snr = np.fft.ifft( 2 * np.fft.ifftshift( dataF1 * np.conj(dataF2) ) ).real * opts.sampling_rate ### ifft normalizes the sum by 1/n = 1/(s*T) and we want to normalize by 1/T to approximate the integral
SNR = snr**0.5

print "WARNING: may want to include \"diagonal\" compenents of the likelihood or to normalize it to a posterior"

#-------------------------------------------------

if opts.verbose:
    print "plotting sanity check of injection and noise"

fig = plt.figure(figsize=(15,10))

### IFO1 raw data
ax = plt.subplot(2,3,1)

ax.plot( times, dataT1, 'm-', linewidth=1, alpha=0.75, label='$\mathrm{noise_1+signal_1}$' )
ax.plot( times-dt/2, dataT1, 'r-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ noise_1+signal_1}$' )

ax.legend(loc='best')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$d_1(t)$')

ax.set_xlim(xmin=dt/2, xmax=opts.duration-dt/2)
ylim = ax.get_ylim()

### IFO1 strain data
ax = plt.subplot(2,3,2)

ax.plot( times, hTimeDom1, 'm-', linewidth=1, alpha=0.75, label='$\mathrm{signal_1}$' )
ax.plot( times-dt/2, hTimeDom1, 'r-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ signal_1}$' )

print "WARNING: need optimal reconstructed strain for IFO1!"

ax.set_ylim(ylim)

#ax.legend(loc='best')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('$\mathrm{time}$')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.setp(ax.get_yticklabels(), visible=False)
ax.set_ylabel('$h_1(t)$')

ax.set_xlim(xmin=dt/2, xmax=opts.duration-dt/2)

plt.annotate(s='', xy=(to+dt/2,np.min(hTimeDom1)), xytext=(to,np.min(hTimeDom1)), arrowprops=dict(arrowstyle='<-'))
#plt.annotate(s='$\\tau$', xy=(to+dt/4,np.min(hTimeDom1)*1.1), xytext=(to+dt/4,np.min(hTimeDom1)*1.1) )

ylim = ax.get_ylim()
ax.plot( [to]*2, ylim, 'k--', alpha=0.5, linewidth=1 )
ax.set_ylim(ylim)

### IFO2 raw data
ax = plt.subplot(2,3,4)

ax.plot( times, dataT2, 'c-', linewidth=1, alpha=0.75, label='$\mathrm{noise_2+signal_2}$' )
ax.plot( times+dt/2, dataT2, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ noise_2+signal_2}$' )

ax.legend(loc='best')
ax.set_xlabel('$\mathrm{time}$')
ax.set_ylabel('$d_2(t)$')

ax.set_xlim(xmin=dt/2, xmax=opts.duration-dt/2)

ylim = ax.get_ylim()

### IFO2 strain data
ax = plt.subplot(2,3,5)

ax.plot( times, hTimeDom2, 'c-', linewidth=1, alpha=0.75, label='$\mathrm{signal_2}$' )
ax.plot( times+dt/2, hTimeDom2, 'b-', linewidth=1, alpha=0.5, label='$\mathrm{shifted\ signal_2}$' )

print "WARNING: need optimal reconstructed strain for IFO2!"

ax.set_ylim(ylim)

#ax.legend(loc='best')
ax.set_xlabel('$\mathrm{time}$')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.setp(ax.get_yticklabels(), visible=False)
ax.set_ylabel('$h_2(t)$')

ax.set_xlim(xmin=dt/2, xmax=opts.duration-dt/2)

plt.annotate(s='', xy=(to-dt/2,np.max(hTimeDom2)), xytext=(to,np.max(hTimeDom2)), arrowprops=dict(arrowstyle='<-'))
#plt.annotate(s='$\\tau$', xy=(to-dt/4,np.max(hTimeDom2)*1.1), xytext=(to-dt/4,np.max(hTimeDom2)*1.1) )

ylim = ax.get_ylim()
ax.plot( [to]*2, ylim, 'k--', alpha=0.5, linewidth=1 )
ax.set_ylim(ylim)

### ray-plot
ax = plt.subplot(3,3,6)

truth = times<=opts.D_over_c
ax.plot( times[truth], SNR[truth], 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )

truth = times[-1]-times <= opts.D_over_c
ax.plot( times[truth]-times[-1], SNR[truth], 'g-', linewidth=1, alpha=0.5, label='$\mathrm{freq-domain}$\n$\mathrm{computation}$' )

ylim = ax.get_ylim()
ax.plot( [dt]*2, ylim, 'k--', linewidth=1, alpha=0.5 )
ax.set_ylim(ylim)

#ax.legend(loc='best')
ax.set_xlabel('$\\tau$')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.set_ylabel('$\\rho(\\tau)$')

ax.set_xlim(xmin=-opts.D_over_c, xmax=opts.D_over_c)

ax = ax.twiny()
thetas = [-90, -45, -30, -15, 0, 15, 30, 45, 90]
ax.set_xticks([opts.D_over_c*np.sin(theta*np.pi/180) for theta in thetas])
ax.set_xticklabels(["$%d^\circ$"%theta for theta in thetas])

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('$\\theta$')

plt.subplots_adjust(hspace=0.05, wspace=0.05)

figname = "sanityCheck%s.png"%(opts.tag)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname )
plt.close( fig )

if opts.sanity_check:
    import sys
    sys.exit(0)

#-------------------------------------------------

if opts.verbose:
    print "making movie frames"

N = opts.duration*opts.sampling_rate
frame_step = int( 1.0*N / opts.num_frames )

print "WARNING: may want to restrict the range of the timeslides to that which is causal..."

frameNo = 0

### plot an openning frame
fig = plt.figure()

raise StandardError("WRITE ME")

figname = "frame%s-%04d.png"%(opts.tag, frameNo)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname )
plt.close(fig)

frameNo += 1

### plot the rest of the frames
ind = 0
while ind < N:
    fig = plt.figure()

    raise StandardError("WRITE ME")

    figname = "frame%s-%04d.png"%(opts.tag, frameNo)
    if opts.verbose:
        print "    %s"%figname
    fig.savefig( figname )
    plt.close(fig)

    frameNo += 1
    ind += frame_step

### plot the final frame
fig = plt.figure()

raise StandardError("WRITE ME")

figname = "frame%s-%04d.png"%(opts.tag, frameNo)
if opts.verbose:
    print "    %s"%figname
fig.savefig( figname )
plt.close(fig)

#-------------------------------------------------

cmd = "ffmpeg -r %d -i frame%s-%s04d.png coherentLiklihood%s.mp4"%(opts.frames_per_sec, opts.tag, "%", opts.tag)
if opts.verbose:
    print "wrapping into a movie:\n\t%s"%(cmd)

sp.Popen(cmd.split()).wait()
