"""
Plotting Functions
Functions that generate the wavelet scalograms.
"""
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

from ..w_transform import HaarTransform, InvHaarTransform

from .plottingtools import _BinData, _findmin, _findmax
from .plottingtools import _NewColorMap, _NSigmaFilter

__all__ = ['wScalogram', 'wScalogram_nsig']

Data_color='#0782B0'
Coeffs_color='#69B4F2'
Firsttrend_color=Coeffs_color
Nsigma_color='#54B959'

def wScalogram(data, hypothesis=None,
               nsigma=None, nsigma_min=None, nsigma_percent=1,
               reconstruction_scaled=False,
               firsttrend=False,
               logscale=True,
               filled=False,
               title=None,
               titlesize=18,
               xlabel=None,
               textsize=14,
               textpos=0.91,
               figsize=(12,12),
               ticksize=11,
               ypad=0,
               markersize=3,
               outputfile=None):
    """
    Function that generates a bar plot of the wavelet coefficients of the data array
    per level.
    Parameters
    ----------
    data : array
    Array to calculate the discrete Haar wavelet transform on.
    firsttrend : bool
    Whether to include the first trend on the scalogram plot.
    filled : bool
    Whether to fill the bars or just show their contour.
    outputfile : string
    Name of the png file to save the plot to. If None, don't print the plot.
    """

    WaveDec_data = HaarTransform(data)
    Ccoeffs = WaveDec_data[:-1]
    FirstTrend = WaveDec_data[-1]
    Level = len(Ccoeffs)

    nlevels = Level if firsttrend==False else Level+1
    nrows = nlevels+1 # the first panel is the data histogram
    if nsigma is not None:
        nrows += 1 # add another panel for the generating function
    ratio = [1.5]
    ratio += [1]*(nrows-1)

    if filled==True:
        histtype='bar'
        coeffs_color=Coeffs_color
        firsttrend_color=Firsttrend_color
    else:
        histtype='step'
        coeffs_color='black'
        firsttrend_color='black'

    if logscale==True:
        scale='log'
    else:
        scale='linear'

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols=1, nrows=nrows,
                           height_ratios=ratio,
                           hspace=0)
    axs = [fig.add_subplot(gs[i,0]) for i in range(nrows)]

    # Fill out top panel
    data_hist, _, data_center, data_width = _BinData(data, bins=2**Level)
    axs[0].bar(data_center, data_hist, align='center',
               width=data_width, color=Data_color)
    axs[0].tick_params(axis='both', labelsize=ticksize)
    axs[0].text(x=textpos, y=.63, s='Data', fontsize=textsize,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                transform=axs[0].transAxes)
    axs[0].set_yscale(scale)

    # If nsigma is provided
    if nsigma is not None:

        nsigCcoeffs = nsigma

        cut = '(No cut)'
        if nsigma_percent is not None:
            cut = str(nsigma_percent*100) + '%'
        if nsigma_min is not None:
            cut = r'$\sigma_{min}$ = ' + str(nsigma_min)

        if hypothesis is not None:
            #TODO: error trap
            DeltaCoeff = _NSigmaFilter(data, hypothesis, nsigma, nsigma_min, nsigma_percent)
            ReconstructedData = InvHaarTransform(DeltaCoeff, normalize=False)
            if reconstruction_scaled is True:
                RecData = np.divide(ReconstructedData, np.sqrt(hypothesis))
            else:
                RecData = ReconstructedData
            rec_hist, _, rec_center, rec_width = _BinData(RecData, bins=2**Level)
            axs[1].plot(rec_center, rec_hist, 'o', markersize=markersize, color='#E67E22',
                        label='Reconstruction ({})'.format(cut))
            axs[1].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
            axs[1].tick_params(axis='x', labelbottom=False)
            axs[1].set_yscale('linear')
            axs[1].legend(edgecolor="black", fancybox=False, borderpad=.2,
                          handletextpad=0.0, handlelength=0, markerscale=0, fontsize=textsize)

    # If firsttrend, fill out the bottom panel with the first trend
    if firsttrend==True:
        bins = 1
        n, binning, patches = axs[-1].hist(x=range(bins), bins=bins, weights=FirstTrend,
                                        histtype=histtype, color=firsttrend_color)
        axs[-1].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
        axs[-1].tick_params(axis='x', labelbottom=False)
        axs[-1].set_xticks(np.linspace(binning[0], binning[-1], 2**(Level-1)))
        axs[-1].set_yscale(scale)
        axs[-1].text(x=textpos, y=.63, s=r'$\ell={%.1i}$'%(0), fontsize=textsize,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                     transform=axs[-1].transAxes)

    # Fill out the rest of the pannels with the wavelet coefficients
    # If signal_only, start two panels below the top panel
    s = 2 if nsigma is not None else 1
    for l in range(Level):
        bins=2**(Level-l-1)
        coeffs = Ccoeffs[l]

        if logscale==True:
            # Plot the positive coefficients
            pos_ix = np.where(Ccoeffs[l]>0)
            pos_coeffs = np.zeros_like(coeffs)
            for i in pos_ix:
                pos_coeffs[i] = coeffs[i]
            axs[l+s].hist(x=range(bins), bins=bins,
                          weights=pos_coeffs, histtype=histtype, color=coeffs_color)

            # Now plot the negative coefficients. The bars are hashed to distinguish the
            # pos and neg coefficients.
            neg_ix = np.where(Ccoeffs[l]<0)
            neg_coeffs = np.zeros_like(coeffs)
            for j in neg_ix:
                neg_coeffs[j] = np.absolute(coeffs[j])
            n, binning, patches = axs[l+s].hist(x=range(bins), bins=bins,
                          weights=neg_coeffs, histtype=histtype, hatch='///', color=coeffs_color)

            axs[l+s].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
            axs[l+s].tick_params(axis='x', labelbottom=False)
            axs[l+s].set_xticks(np.linspace(binning[0], binning[-1], 2**(Level-1) ))

            lev = Level-l-1
            axs[l+s].text(x=textpos, y=.63, s=r'$\ell={%.1i}$'%(lev+1), fontsize=textsize,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                          transform=axs[l+s].transAxes)
            axs[l+s].set_yscale(scale)

        else:
            n, binning, patches = axs[l+s].hist(x=range(bins), bins=bins, weights=coeffs, histtype=histtype, color=coeffs_color)
            axs[l+s].plot(binning, np.zeros_like(binning), color='black', linewidth=0.5)
            axs[l+s].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
            axs[l+s].tick_params(axis='x', labelbottom=False, labelsize=ticksize)
            axs[l+s].set_xticks(np.linspace(binning[0], binning[-1], 2**(Level-1) ))
            lev = Level-l-1
            axs[l+s].text(x=textpos, y=.63, s=r'$\ell={%.1i}$'%(lev+1), fontsize=textsize,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                          transform=axs[l+s].transAxes)
            axs[l+s].set_yscale(scale)

    if title is not None:
        fig.suptitle(title, fontsize=titlesize, y=0.92)
        fig.text(x=0.5, y=0.1, s=xlabel, fontsize=textsize)
    if outputfile is not None:
        plt.savefig(outputfile, bbox_inches='tight')
    plt.show()


def wScalogram_nsig(data, hypothesis=None,
                    nsigma=None, nsigma_min=None, nsigma_percent=1,
                    reconstruction_scaled=False,
                    firsttrend=False,
                    logscale=True,
                    title=None,
                    titlesize=18,
                    xlabel=None,
                    textsize=14,
                    figsize=(12,12),
                    textpos=0.91,
                    ticksize=11,
                    markersize=3,
                    outputfile=None):

    """
    Function that generates a bar plot of the wavelet coefficients of the data array
    per level.
    Parameters
    ----------
    data : array
    Array to calculate the discrete Haar wavelet transform on.
    nsigma : array
    Nsigma array to use as the color-code for the wavelet coefficients.
    firsttrend : bool
    Whether to include the first trend on the scalogram plot.
    logscale : bool
    Whether to use a linear of log scale on the y-axis .
    outputfile : string
    Name of the png file to save the plot to. If None, don't print the plot.
    """

    WaveDec_data = HaarTransform(data)
    Ccoeffs = WaveDec_data[:-1]
    FirstTrend = WaveDec_data[-1]
    Level = len(Ccoeffs)

    if logscale==True:
        scale='log'
    else:
        scale='linear'

    nlevels = Level if firsttrend==False else Level+1
    nrows = nlevels+1 # the first panel is the data histogram
    if nsigma is not None:
        nrows += 1 # add another panel for the generating function
    ratio = [1.5]
    ratio += [1]*(nrows-1)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols=1, nrows=nrows,
                           height_ratios=ratio,
                           hspace=0)
    axs = [fig.add_subplot(gs[i,0]) for i in range(nrows)]
    cbar_axs = fig.add_axes([0.93, 0.15, 0.02, 0.7]) # colorbar axis

    # Fill out top panel
    data_hist, _, data_center, data_width = _BinData(data, bins=2**Level)
    axs[0].bar(data_center, data_hist, align='center', width=data_width, color=Data_color)
    axs[0].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
    axs[0].tick_params(axis='x', labelbottom=False)
    axs[0].set_yscale(scale)
    axs[0].text(x=textpos, y=.63, s='Data', fontsize=textsize,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                transform=axs[0].transAxes)

    # If nsigma function is provided
    if nsigma is not None:

        nsigCcoeffs = nsigma

        cut = '(No cut)'
        if nsigma_percent is not None:
            cut = str(nsigma_percent*100) + '%'
        if nsigma_min is not None:
            cut = r'$\sigma_{min}$ = ' + str(nsigma_min)

        if hypothesis is not None:
            #TODO: error trap
            DeltaCoeff = _NSigmaFilter(data, hypothesis, nsigma, nsigma_min, nsigma_percent)
            ReconstructedData = InvHaarTransform(DeltaCoeff, normalize=False)
            if reconstruction_scaled is True:
                RecData = np.divide(ReconstructedData, np.sqrt(hypothesis))
            else:
                RecData = ReconstructedData
            rec_hist, _, rec_center, rec_width = _BinData(RecData, bins=2**Level)
            axs[1].plot(rec_center, rec_hist, 'o', markersize=markersize, color='#E67E22',
                        label='Reconstruction ({})'.format(cut))
            axs[1].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
            axs[1].tick_params(axis='x', labelbottom=False)
            axs[1].set_yscale('linear')
            axs[1].legend(edgecolor="black", fancybox=False, borderpad=.2,
                          handletextpad=0, handlelength=0, markerscale=0, fontsize=textsize)

    cmap = _NewColorMap()
    binintensity = np.absolute(nsigma)
    sig_min = _findmin(binintensity)
    sig_max = _findmax(binintensity)
    norm = Normalize(vmin=sig_min, vmax=sig_max)

    # If firsttrend, fill out the bottom panel with the first trend
    if firsttrend==True:
        bins=1
        norm_points = norm(binintensity[-1])
        color_points = [cmap(i) for i in norm_points]
        hist, edges, center, width = _BinData(FirstTrend, bins=1)
        axs[-1].bar(center, hist, align='center', width=width, color=color_points)
        axs[-1].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
        axs[-1].tick_params(axis='x', labelbottom=False)
        axs[-1].set_xticks(np.linspace(edges[0], edges[-1], 2**(Level-1)))
        axs[-1].set_yscale(scale)
        axs[-1].text(x=textpos, y=.63, s=r'$\ell={%.1i}$'%(0), fontsize=textsize,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                     transform=axs[-1].transAxes)

    # Now plot the negative coefficients. The bars are hashed to distinguish the
    # pos and neg coefficients.
    s = 2 if nsigma is not None else 1
    for l in range(Level):
        bins=2**(Level-l-1)
        coeffs = Ccoeffs[l]
        norm_points = norm(binintensity[l])
        color_points = [cmap(i) for i in norm_points]

        if logscale==True:
            # Plot the positive coefficients
            pos_ix = np.where(coeffs>0)
            pos_coeffs = np.zeros_like(coeffs)
            for i in pos_ix:
                pos_coeffs[i] = coeffs[i]
            pos_hist, pos_edges, pos_center, pos_width = _BinData(pos_coeffs, bins=bins)
            axs[l+s].bar(pos_center, pos_hist, align='center', width=pos_width, color=color_points)

            # Now plot the negative coefficients. The bars are hashed to distinguish the
            # pos and neg coefficients.
            neg_ix = np.where(Ccoeffs[l]<0)
            neg_coeffs = np.zeros_like(coeffs)
            for j in neg_ix:
                neg_coeffs[j] = np.absolute(coeffs[j])
            neg_hist, neg_edges, neg_center, neg_width = _BinData(neg_coeffs, bins=bins)
            axs[l+s].bar(neg_center, neg_hist, align='center', width=neg_width, color=color_points,
                         hatch='///')

            left_edge=np.minimum(pos_edges[0], neg_edges[0])
            right_edge=np.maximum(pos_edges[-1], neg_edges[-1])
            axs[l+s].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
            axs[l+s].tick_params(axis='x', labelbottom=False)
            axs[l+s].set_xticks(np.linspace(left_edge, right_edge, 2**(Level-1)))

            lev = Level-l-1
            axs[l+s].text(x=textpos, y=.63, s=r'$\ell={%.1i}$'%(lev+1), fontsize=textsize,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                          transform=axs[l+s].transAxes)
            axs[l+s].set_yscale(scale)

        else:
            hist, edges, center, width = _BinData(coeffs, bins=bins)
            axs[l+s].plot(binning, np.zeros_like(binning), color='black', linewidth=0.5)
            axs[l+s].bar(center, hist, align='center', width=width,
                         color=color_points)
            axs[l+s].plot(range(bins), np.zeros(bins), color='black',
                          linewidth=0.5)
            axs[l+s].tick_params(axis='y', bottom=False, labelbottom=False, labelsize=ticksize)
            axs[l+s].tick_params(axis='x', labelbottom=False)
            axs[l+s].set_xticks(np.linspace(edges[0], edges[-1], 2**(Level-1)))
            lev=Level-l-1
            axs[l+s].text(x=textpos, y=.63, s=r'$C_{l=%.1i}$'%(lev), fontsize=textsize,
                          bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
                          transform=axs[l+s].transAxes)
            axs[l+s].set_yscale(scale)

    cbar = ColorbarBase(cbar_axs, cmap=cmap, norm=norm)
    #cbar_axs.text(.5, sig_max, r'$N\sigma$', fontsize=12)
    fig.text(x=0.93, y=.86, s=r'$N\sigma$', fontsize=textsize)

    if title is not None:
        fig.suptitle(title, fontsize=titlesize, y=0.92)
        fig.text(x=0.5, y=0.1, s=xlabel, fontsize=textsize)
    if outputfile is not None:
        plt.savefig(outputfile, bbox_inches='tight')
    plt.show()
