#!/usr/bin/python

usage = "worldline [--options]"
description = "make a space-time diagram of light in a Michelson interferometer"
author = "Reed Essick"

#-------------------------------------------------

import subprocess as sp

import numpy as np
from scipy.optimize import newton_krylov as solver

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "figure.subplot.bottom": 0.10,
    "figure.subplot.left": 0.10,
    "figure.subplot.right": 0.95,
    "figure.subplot.top": 0.95,
    "grid.color": 'gray',
#    "image.aspect": 'auto',
#    "image.interpolation": 'nearest',
#    "image.origin": 'lower',
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

from optparse import OptionParser

#-------------------------------------------------

def TofX(X, ho, n, omega, phi):
    '''
    numerically solve
        T = X + ho/(omega*(1-n))*sin(omega*(T-nX)+phi)
    for T
    '''
    return solver(lambda T: X + ho/(omega*(1-n))*(np.sin(omega*(T-n*X)+phi)-np.sin(phi)) - T, X)

#-------------------------------------------------

parser = OptionParser(usage=usage, description=description)

parser.add_option('-v', '--verbose', default=False, action='store_true')

parser.add_option('-o', '--output-dir', default='.', type='string')
parser.add_option('-t', '--tag', default='', type='string')

parser.add_option('', '--L', default=1, type='float')
parser.add_option('', '--f', default=0.1, type='float')

parser.add_option('', '--nx', default=0, type='float')
parser.add_option('', '--ny', default=0, type='float')
parser.add_option('', '--hxx', default=+0.1, type='float')
parser.add_option('', '--hyy', default=-0.1, type='float')

parser.add_option('', '--N', default=1001, type='int')

parser.add_option('', '--alpha', default=0.5, type='float')
parser.add_option('', "--Nphi", default=5, type='float')

parser.add_option('', '--shift-start', default=False, action='store_true')

parser.add_option('', '--Nframes', default=100, type='int')

parser.add_option('', "--dpi", default=100, type='int')

parser.add_option('', '--frames-per-sec', default=30, type='int', help='the number of frames per second of the movie')
parser.add_option('', '--movie-type', default=[], action='append', type='string')

opts, args = parser.parse_args()

if opts.tag:
    opts.tag = "_"+opts.tag

omega = 2*np.pi*opts.f

if not opts.movie_type:
    opts.movie_type.append( 'mpg' )

#-------------------------------------------------

phis = np.linspace(0, 2*np.pi, opts.Nphi+1)
vmin = min(phis)
vmax = max(phis)
if vmin==vmax:
    vmin=-100

m1 = cm.ScalarMappable(
    norm=plt.Normalize(
        vmin=vmin,
        vmax=vmax,
    ),
    cmap=plt.get_cmap('jet'),
)
m1.set_array(phis)

m2 = cm.ScalarMappable(
    norm=plt.Normalize(
        vmin=vmin,
        vmax=vmax,
    ),
    cmap=plt.get_cmap('GnBu'),
)
m2.set_array(phis)

maximum = opts.L

x_out = np.linspace(0, opts.L, opts.N)
x_bck = opts.L-x_out

X = []
Y = []
for phi in phis:
    if opts.verbose:
        print( 'phi = %.3f'%phi )
        print( '    finding T(x) during outbound journey' )

    X_out = np.array([TofX(_, opts.hxx, opts.nx, omega, phi) for _ in x_out])
    Y_out = np.array([TofX(_, opts.hyy, opts.ny, omega, phi) for _ in x_out])

    if opts.shift_start:
        X_out += phi/omega
        Y_out += phi/omega

    if opts.verbose:
        print( '    finding T(x) during the return journey' )
    X_bck = X_out[-1] + np.array([TofX(_, opts.hxx, -opts.nx, omega, phi+omega*(X_out[-1]-opts.nx*opts.L)) for _ in x_out])
    Y_bck = Y_out[-1] + np.array([TofX(_, opts.hyy, -opts.ny, omega, phi+omega*(Y_out[-1]-opts.ny*opts.L)) for _ in x_out])

    X.append(np.concatenate((X_out, X_bck)))
    Y.append(np.concatenate((Y_out, Y_bck)))

X = np.array(X)
Y = np.array(Y)
L = np.concatenate((x_out, x_bck))

times = np.linspace(0, max(np.max(X), np.max(Y)), opts.Nframes)
for ind, t in enumerate(times):
    
    fig = plt.figure()
    ax = fig.gca()

#    ax.plot([0, opts.L, 0], [0, opts.L, 2*opts.L], 'k--')

    for jnd, phi in enumerate(phis):
#        ax.plot(np.interp(times[:ind], Y[jnd,:], L), times[:ind], color=m2.to_rgba(phi))
        ax.plot(np.interp(times[:ind], Y[jnd,:], L), times[:ind], color='grey')
        ax.plot(np.interp(times[:ind], X[jnd,:], L), times[:ind], color=m1.to_rgba(phi))

    ### decorate
    ax.set_xlabel('r/L')
    ax.set_ylabel('t')

    ax.grid(True, which='both')

    ax.set_xlim(xmin=0, xmax=opts.L)
    ax.set_ylim(ymin=0, ymax=times[-1])

#    ax.text(0.1, opts.L, 'x-arm', color=m1.to_rgba(vmax), ha='center', va='bottom', fontsize=14)
#    ax.text(0.1, opts.L, 'y-arm', color=m2.to_rgba(vmax), ha='center', va='top', fontsize=14)

    ### save
    figname = "%s/frame%s-%04d.png"%(opts.output_dir, opts.tag, ind)
    if opts.verbose:
        print( figname )
    fig.savefig(figname, dpi=opts.dpi)
    plt.close(fig)

#------------------------

for movie_type in opts.movie_type:
    cmd = "ffmpeg -r %d -i frame%s-%s04d.png worldline%s.%s"%(opts.frames_per_sec, opts.tag, "%", opts.tag, movie_type)
    if opts.verbose:
        print "wrapping into a movie:\n\t%s"%(cmd)

    sp.Popen(cmd.split()).wait()
