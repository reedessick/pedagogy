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

parser.add_option('', '--cartesian', default=False, action='store_true', help='distribute dots on a rectangular grid')
parser.add_option('', '--dots-per-row', default=11, type='int')
parser.add_option('', '--dots-per-col', default=11, type='int')

parser.add_option('', '--cylindrical', default=False, action='store_true', help='distribute dots in cylindrical pattern')
parser.add_option('', '--dots-per-radius', default=11, type='int')
parser.add_option('', '--dots-per-circumfrence', default=51, type='int')

parser.add_option('', '--random', default=False, action='store_true', help='randomly distribute dots')
parser.add_option('', '--num-dots', default=242, type='int')

parser.add_option('', '--hp_max-strain', default=0.10, type='float', help='the amplitude of the oscillation')
parser.add_option('', '--hp_frequency', default=0.5, type='float', help='the frequency of the oscillation in Hz')
parser.add_option('', '--hp_phase', default=0, type='float')

parser.add_option('', '--hx_max-strain', default=0.10, type='float', help='the amplitude of the oscillation')
parser.add_option('', '--hx_frequency', default=0.5, type='float', help='the frequency of the oscillation in Hz')
parser.add_option('', '--hx_phase', default=0, type='float', help='default is pi/4 relative to hp')

parser.add_option('', '--frames-per-sec', default=30, type='int', help='the number of frames per second of the movie')
parser.add_option('', '--num-frames', default=400, type='int', help='the total number of frames in the movie')

parser.add_option('', '--tag', default='', type='string' )
parser.add_option('', '--dpi', default=200, type='int' )
parser.add_option('', '--movie-type', default=[], action='append', type='string')

opts, args = parser.parse_args()

if np.sum([opts.cartesian, opts.cylindrical, opts.random])!=1:
    raise ValueError('please supply either --cartesian, --cylindrical, or --random\n%s'%usage)

if opts.tag:
    opts.tag = "_%s"%opts.tag

if not opts.movie_type:
    opts.movie_type.append( 'mpg' )

#-------------------------------------------------

if opts.verbose:
    print "making movie frames"

dt = 1./opts.frames_per_sec

frameNo = 0

if opts.cartesian:
    xraster, yraster = np.meshgrid( np.linspace(-1, 1, opts.dots_per_row), np.linspace(-1, 1, opts.dots_per_col) )

elif opts.cylindrical:
    xraster = [0]
    yraster = [0]
    dr = 1./opts.dots_per_radius
    dl = 2*np.pi/opts.dots_per_circumfrence
    r = dr
    while r <= 1:
        for p in np.linspace(0, 2*np.pi, int(2*np.pi*r/dl)):
            xraster.append( r*np.cos(p) )
            yraster.append( r*np.sin(p) )
        r += dr
    xraster = np.array(xraster)
    yraster = np.array(yraster)

elif opts.random:
    xraster, yraster = np.reshape(2*np.random.rand(2*opts.num_dots)-1, (2, opts.num_dots))

else:
    raise ValueError('bad options\n%s'%usage)


xlim = np.array([-1.1, 1.1])*(1+(opts.hp_max_strain**2 + opts.hx_max_strain**2)**0.5)
ylim = xlim

while frameNo < opts.num_frames:
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0, 0, 1, 1])

    t = frameNo*dt

    hp = opts.hp_max_strain * np.sin( 2*np.pi*t*opts.hp_frequency + opts.hp_phase )
    hx = opts.hx_max_strain * np.cos( 2*np.pi*t*opts.hx_frequency + opts.hx_phase )

    dx = hp*xraster + hx*yraster
    dy = hx*xraster - hp*yraster

    ax.plot( xraster+dx, yraster+dy, marker='o', markerfacecolor='k', markeredgecolor='k', markersize=4, linestyle='none' )

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
