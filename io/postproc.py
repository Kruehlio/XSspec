# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyfits

from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

from ..utils.fitter import onedgaussfit
from ..utils.astro import ergJy, abflux, errs, Jyerg, highabswin

#==============================================================================


def telCor(s2d, arms):
    haws = highabswin()
    for arm in arms:
        print '\tCreating telluric model for arm %s' %arm
        telcor = np.ones(len(s2d.wave[arm]))
        wl = s2d.wave[arm]
        f = open('%s_telcor_%s.txt' %(s2d.object, arm), 'w')
        for i in range(len(wl)):
            for haw in haws:
                if haw[0] < wl[i] and wl[i] < haw[1]: 
                    if (s2d.cont[arm][i] / s2d.oneddata[arm][i]) > 1:
                        telcor[i] = s2d.cont[arm][i] / s2d.oneddata[arm][i]
            f.write('%.3f\t%.3f\n' %(wl[i], telcor[i]))
        f.close()
        s2d.telcorr[arm] = telcor
        s2d.telwave[arm] = s2d.wave[arm]
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))            
        
        ax.plot(s2d.wave[arm], s2d.telcorr[arm], 'black')
        fig.savefig('%s_telcor_%s.pdf' %(s2d.object, arm))

#==============================================================================
    

def applTel(s2d, arms):
    for arm in arms:
        print '\tCorrecting for tellurics arm %s' %arm
        tck = interpolate.InterpolatedUnivariateSpline(s2d.telwave[arm], 
                                           s2d.telcorr[arm])
        tellcor = tck(s2d.wave[arm])
        s2d.oneddata[arm] *= tellcor

#==============================================================================


def fluxCor(s2d, arm, fluxf, countf):
    print '\tConverting counts to flux'
    fluxspec = np.array(pyfits.getdata(fluxf, 0)[0].transpose())
    countspec = np.array(pyfits.getdata(countf, 0)[0].transpose())
    wlkey, wlstart = 'NAXIS%i'%s2d.dAxis[arm], 'CRVAL%i'%s2d.dAxis[arm]
    wlinc, wlpixst = 'CDELT%i'%s2d.dAxis[arm], 'CRPIX%i'%s2d.dAxis[arm]
    head = pyfits.getheader(fluxf, 0)
    pix = np.arange(head[wlkey]) + 1
    wave = (head[wlstart] + (pix - head[wlpixst])*head[wlinc])*s2d.wlmult
    count = np.where(countspec == 0, 1E-5, countspec)
    resp = fluxspec/count
    tck = interpolate.InterpolatedUnivariateSpline(wave, resp)
    respm = tck(s2d.wave[arm])
    fig = plt.figure(figsize = (9,9))
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
    
    ax.set_yscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.plot(s2d.wave[arm], respm, 'o')
    ax.set_xlabel(r'$\rm{Observed\,wavelength\,(\AA)}$')
    ax.set_ylabel(r'$\rm{Response\,function}$')
    fig.savefig('%s_resp_%s.pdf' %(s2d.object, arm))
    s2d.data[arm] = (respm*s2d.data[arm].transpose()).transpose()
    s2d.erro[arm] = (respm*s2d.erro[arm].transpose()).transpose()  
    

#==============================================================================

def scaleSpec(s2d, arm, pband = '', mag = '', err = 1.E-5,
              method = 'median', dls = 2, intsys = 0):
    ''' Calculates the scale factor betwen the spectrum and photometry in 
    a given photometric band, calculates the missing flux in absorption lines '''

    bandwl = {'g': [3856.2, 5347.7], 'r': [5599.5, 6749.0], 'i': [7154.9, 8156.6], 
      'B': [3742.5, 4717.0], 'V': [4920.9, 5980.2], 'R': [5698.9, 7344.4], 
      'I': [7215.0, 8725.7], 'F814': [6884.0, 9659.4], 
      'Y': [9686.0, 10701.8],
      'z': [8250.0, 9530.4], 'J': [11192.6, 13558.2], 'H': [15102.4, 17803.9],
      'F160W': [13996.4, 16870.0], 'K': [20144.5, 23176.1],
      'v': [5068.0, 5868.0]}

    if bandwl.has_key(pband):
        wlsel = (bandwl[pband][0] < s2d.wave[arm]) * (s2d.wave[arm] < bandwl[pband][1]) 
    else:
        wlsel = (pband[0] < s2d.wave[arm]) * (s2d.wave[arm] < pband[1]) 

    print '\tEstimating flux in band %s' %pband
    # First, we select the requested wavelengths for photband
    pb, pbw, pbe = s2d.oneddata[arm][wlsel], s2d.wave[arm][wlsel], s2d.onederro[arm][wlsel]
    bb, bbe = s2d.onedback[arm][wlsel], s2d.skyrms[arm][wlsel]

    # Second, only where telluric absorption is not strong, transmission >0.85
    # And exclude regions of high sky radiance       
    if arm in ['vis', 'nir']:        
        skyexl = s2d.skytel[arm][wlsel] > 0.85
        skyexl *= s2d.skyrad[arm][wlsel] < 5E2
    else:
        skyexl = s2d.skytel[arm][wlsel] > 0.

    pbc, pbwc, pbec = pb[skyexl], pbw[skyexl], pbe[skyexl]
    bbc, bbec = bb[skyexl], bbe[skyexl]
        
    print '\t\tUsing %i data points for flux estimation' %len(bbc)
    
    # Now, calculate flux in background region and trace
    bandfl = ergJy(pbc,pbwc)
    bande = abs(pbec/pbc*bandfl)
    bandback = ergJy(bbc, pbwc)
    bandbacke = abs(ergJy(bbec, pbwc))
    bandbacke[bandbacke<=0] = np.median(bandbacke)
    
    if mag != '':
        magsel = np.ones(1000)*mag
        mag = np.random.normal(magsel, err)   
        fluxcomp = abflux(mag) 
    elif s2d.modfl != '':
        wlsel2 = (bandwl[pband][0] < s2d.modwl) * (s2d.modwl < bandwl[pband][1]) 
        bandflux = np.array(s2d.modfl[wlsel2])
        fluxcomp = np.median(bandflux) * np.ones(1000)
    else:
        print ('\tNeed either a magnitude via mag or a defined ASCII Model')
        raise SystemExit
    print '\t\t\tComparison flux = %.2f \muJy' %(np.median(fluxcomp))

    while True:
        bguess = np.median(bandfl)
        brms = np.std(bandfl)
        clip = (bandfl > (bguess - 6*brms)) * (bandfl < (bguess + 6*brms))
        if clip.all() == True:
            break
        bandfl, bande = bandfl[clip], bande[clip]
        bandback, bandbacke = bandback[clip],  bandbacke[clip]
    #bandfl = smooth(bandfl, window_len=10, window='median')
    #bandback = smooth(bandback, window_len=10, window='median')

    bflints, bbackints, n, corrfs = [], [], 1000, []
    for i in range(n):
        bflint = sum(np.random.normal(bandfl, abs(bande)))/len(bandfl)
        bbacki = sum(np.random.normal(bandback, abs(bandbacke)))/len(bandfl)
        bflints.append(bflint-bbacki)
        bbackints.append(bbacki)
        corrfs.append(np.random.choice(fluxcomp)/(bflint-bbacki))
        
    valmed, valmin, valmax, val2s = errs(bflints)
    print '\t\t\tFlux (%s-band) = %.2f+/-%.2f \muJy' \
            %(pband, valmed, valmax)    
    print '\t\t\tFlux (%s-band) = %.2e+/-%.2e erg/cm^2/s/AA' \
            %(pband, Jyerg(valmed, np.average(bandwl[pband])), 
              Jyerg(valmax, np.average(bandwl[pband])))   
    valmed, valmin, valmax, val2s = errs(bbackints)
    print '\t\t\tBackground (%s-band) = %.2f+/-%.2f \muJy' \
            %(pband, valmed, valmax)
    #valmed, valmin, valmax, val2s = errs(bflux)
    #print '\tFlux (%s-band) w/o Background = %.2f+/-%.2f \muJy' \
    #        %(pband, valmed, valmax) 
    corrf, corrfmin, corrfmax, val2s = errs(corrfs)
    print '\t\t\tCorrection factor (%s arm, %s-band): %.2f+/-%.2f\n' \
            %(arm, pband, corrf, corrfmax)
    s2d.slitcorr[arm] = corrf
    s2d.linepars['sl_%s_%s'%(arm, pband)] = (arm.upper(), pband, corrf, corrfmax)


#==============================================================================


def applyScale(s2d, arms, mdata = 0, usearm = ''):
    ''' Applies the previously defined scaling to the luminosity spectrum
    Uses the same arm, or if not defined the factor for the reddest arm that
    is defined. Can be overridden by usearm parameter. E.g., usearm=vis uses
    the vis factor for all bands in arms'''
    corfarm = {}
    corfa, corfea = [], []
    for arm in arms:
        corfs, corfes = [], []
        for slf in s2d.linepars.keys():
          if slf.startswith('sl'):  
            if s2d.linepars[slf][0] == arm.upper():
                corfs.append(s2d.linepars[slf][2])
                corfes.append(s2d.linepars[slf][3])
        corfa += corfs
        corfea += corfes
        if len(corfs) >= 1:
            corfs, weight = np.array(corfs), 1./np.array(corfes)
            corfarm[arm] = sum(corfs*weight)/sum(weight)
    corfa, weight = np.array(corfa), 1./np.array(corfea)
    corfarm['all'] = sum(corfa*weight)/sum(weight)        
    for arm in arms:
        corf = 1
        if usearm == '':
            if corfarm.has_key('all'):
                corf = corfarm['all']
            else:
                print '\t\tCorrection factor for %s not defined' %usearm
        else:
            if corfarm.has_key(usearm):
                corf = corfarm[arm]
            else:
                print '\t\tCorrection factor for %s not defined' %usearm
        print '\t\tMultiplying %s luminosity spectrum with %.2f' %(arm, corf)
        s2d.lumspec[arm] *= corf
        s2d.lumerr[arm] *= corf
        if mdata != 0:
            print '\t\tMultiplying %s data with %.2f' %(arm, corf)
            s2d.oneddata[arm] *= corf
            s2d.onedback[arm] *= corf
            s2d.onederro[arm] *= corf
                

#==============================================================================


def checkWL(s2d, arms, intensity = 2, chop = 10, mod = 1):
    print '\tChecking accuracy of wavelength solution against skylines'
    for arm in arms:
        if arm == 'uvb':
            chopap, order = min(chop, 1), 0
            xmin = s2d.wltopix(arm, 5450)
            xmax, err = -200, 0.03
        elif arm == 'vis':
            xmin = s2d.wltopix(arm, 6000)
            xmax = s2d.wltopix(arm, 9800)
            chopap, order, err = chop, 1, 0.05
        elif arm == 'nir':
            xmin = s2d.wltopix(arm, 10200)
            xmax = s2d.wltopix(arm, 21000)
            chopap, order, err = chop, 1, 0.02
        waveccs = s2d.wave[arm][xmin:xmax]
        skyrmss = s2d.skyrms[arm][xmin:xmax]
        skyrads = s2d.skyrad[arm][xmin:xmax]

        dl = (waveccs[-1]-waveccs[0])/len(waveccs)   
        skyrad = np.array_split(skyrads, chopap)
        skyrms = np.array_split(skyrmss, chopap)
        wavecc = np.array_split(waveccs, chopap)
        wloff, wls, wlerrs, fwhms, sigma = [], [], [], [], 2
        pp = PdfPages('%s_wlacc_%s.pdf' %(s2d.object, arm))
        for i in range(len(skyrad)):
            corr = np.correlate(skyrms[i], skyrad[i], "full")
            if len(corr) % 2 == 0:
                print '\t\tSomething went wrong here'
                corrc = corr[len(corr)/2-14 : len(corr)/2+14]
            else:
                corrc = corr[len(corr)/2-14 : len(corr)/2+15]
            xcor = np.arange(len(corrc)) - np.median(np.arange(len(corrc)))
            # Detrend baseline:
            minc, maxc = min(corrc), max(corrc)
            sel = [corrc < minc+0.3*(maxc-minc)]
            b = np.polyfit(xcor[sel], corrc[sel], deg = 1)
            corrc -= np.polyval(b, xcor)
            corrc *= 1/max(corrc)
            # Fit Gauss
            offset = xcor[np.where(corrc==corrc.max())]
            peak = corrc.max()

            params = onedgaussfit(xcor, corrc, err = corrc*0+err,
                                  params=[0, peak, offset, sigma],
                                  fixed=[0, 0, 0, 0],
                                  minpars=[0, 0, 0, 0],
                                  limitedmin=[0, 0, 0, 0])  
            
            offset, sigma = params[0][2], params[0][3]
            fwhms.append(sigma*dl*2.3538)
            wloff.append(params[0][2]*dl)
            wls.append(np.average(wavecc[i]))
            wlerrs.append(params[2][2]*dl)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
            
            ax.errorbar(xcor, corrc, yerr = corrc*0+err, capsize = 0,
                        color ='black',fmt = 'o')
            ax.plot(params[-1], params[1], color ='black')
            ax.set_xlabel(r'$\rm{Pixel\,value}$')
            ax.set_ylabel(r'$\rm{CCF}$')
            pp.savefig(fig)
            plt.close(fig)

        a = np.polyfit(wls, wloff, w = 1./np.array(wlerrs)**2, deg = order)

        print '\t\tMedian Offset (%s arm): %.2f AA' \
                    %(arm, np.median(wloff))
        print '\t\tRMS scatter in offset (%s arm): %.2f AA' \
                    %(arm, np.std(wloff))
        if abs(np.median(wloff)) > 1.5 or np.std(wloff) > 0.5:
            print '\t\tWARNING: WAVELENGTH SCALE WILL BE MODIFIED > 1 AA'
        if mod == 1:
            s2d.wave[arm] -= np.polyval(a, s2d.wave[arm])
            print '\t\tWavelength scale modified'
            s2d.restwave[arm] = s2d.wave[arm]/(1+s2d.redshift)
        else:
            print '\t\tWavelength scale not modified'

        fig1 = plt.figure(figsize = (11,6))
        ax1 = fig1.add_subplot(1, 2, 1)
        ax1.errorbar(np.array(wls)/1E4, wloff, yerr = wlerrs, capsize = 0,
                    fmt='o', color ='black')
        ax1.plot(s2d.wave[arm]/1E4, np.polyval(a, s2d.wave[arm]), color ='black')
        ax1.set_xlabel(r'$\rm{Observed\,wavelength\,(\mu m)}$')
        ax1.set_ylabel(r'$\Delta\lambda\,(\AA)$')
        
        ax2 = fig1.add_subplot(1, 2, 2)
        ax2.plot(np.array(wls)/1E4, fwhms, #yerr = wlerrs, capsize = 0,
                    'o', color ='black')
        ax2.set_xlabel(r'$\rm{Observed\,wavelength\,(\mu m)}$')
        ax2.set_ylabel(r'$\rm{FWHM\,(\AA)}$')
        pp.savefig(fig1)
        plt.close(fig1)
        pp.close()   