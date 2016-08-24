#!/usr/bin/python
usage = "strainDemo.py [--options]"
description = "builds figures to demonstrate strain"
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

parser.add_option('', '--dots-per-row', default=10, type='int')
parser.add_option('', '--dots-per-col', default=10, type='int')

parser.add_option('', '--max-strain', default=0.10, type='float', help='the amplitude of the oscillation')
parser.add_option('', '--frequency', default=0.5, type='float', help='the frequency of the oscillation in Hz')

parser.add_option('', '--frames-per-sec', default=30, type='int', help='the number of frames per second of the movie')
parser.add_option('', '--num-frames', default=400, type='int', help='the total number of frames in the movie')

parser.add_option('', '--tag', default='', type='string' )
parser.add_option('', '--dpi', default=200, type='int' )
parser.add_option('', '--movie-type', default=[], action='append', type='string')

opts, args = parser.parse_args()

if opts.tag:
    opts.tag = "_%s"%opts.tag

if not opts.movie_type:
    opts.movie_type.append( 'mpg' )

#-------------------------------------------------

if opts.verbose:
    print "making movie frames"

dt = 1./opts.frames_per_sec

frameNo = 0

xraster = np.linspace(-1, 1, opts.dots_per_row)
yraster = np.linspace(-1, 1, opts.dots_per_col)

xlim = np.array([-1.1, 1.1])*(1+opts.max_strain)
ylim = np.array([-1.1, 1.1])*(1+opts.max_strain)

while frameNo < opts.num_frames:
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0, 0, 1, 1])

    h = opts.max_strain * np.sin( 2*np.pi*frameNo*dt*opts.frequency )

    x, y = np.meshgrid( xraster*(1+h), yraster*(1-h) )

    ax.plot( x, y, marker='o', markerfacecolor='k', markeredgecolor='k', markersize=4, linestyle='none' )

    ax.axis('off')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    figname = "frame%s-%04d.png"%(opts.tag, frameNo)
    if opts.verbose:
        print "    %s"%figname
    fig.savefig( figname, dpi=opts.dpi )
    plt.close( fig )

    frameNo += 1
    
#-------------------------------------------------

for movie_type in opts.movie_type:
    cmd = "ffmpeg -r %d -i frame%s-%s04d.png strainDemo%s.%s"%(opts.frames_per_sec, opts.tag, "%", opts.tag, movie_type)
    if opts.verbose:
        print "wrapping into a movie:\n\t%s"%(cmd)

    sp.Popen(cmd.split()).wait()
