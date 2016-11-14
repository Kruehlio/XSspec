#!/usr/bin/env python

import matplotlib.mlab as mlab
import os
from matplotlib import rc
from matplotlib.patheffects import withStroke
import matplotlib.pyplot as plt
import numpy as np

from .fitter import (onedgaussian, onedgaussfit, onedtwogaussfit, 
                 onedthreegaussfit)
from .functions import (transNIR, tell)
from .astro import (absll, emll, errs, fluxab, fluxaberr, ergJy,
                 airtovac, binspec)
                 
from scipy import integrate

c, c1 = 2.99792458E5, 2.99792458E8
abslist, emllist = absll(), emll()
linelist = dict(emllist.items() + abslist.items())

myeffect = withStroke(foreground="w", linewidth=3)
kwargs = dict(path_effects=[myeffect])

class specPlotter():
    ''' Plotter class associated to the spectrum class. Requires a spectrum2d
    object when calling. Methods: plot2d, plotLine, plotTel, plot1d, plot1dall '''
    
    def __init__(self, s2d, tex=''):
        '''Read in the required data from the spectrum2d object'''
        self.s = s2d
        self.linepars = {}
        self.oneddata = s2d.oneddata
        self.onederro = s2d.onederro
        self.wave = s2d.wave
        self.linepars = s2d.linepars
        self.binwava = []
        self.binerra = []
        self.bindata = []
        self.binwave = []

        print '\n\t######################'
        print '\tPlotter'
        print '\t######################'
        if tex in ['yes', True, 1, 'Y', 'y']:
            rc('text', usetex=True)
        
    def plot1d(self, arm, wl1, wl2, lim=[], lines = [], chop = 7, ewlim = 0.15,
               unit = 'erg', norm = 0, median = 1, fs = 13,
               raster = False, ploto = 0, rest=False):
        
        if len(arm) == 6:
          if arm == 'uvbvis':
            uvbsel = self.wave['uvb'] < 5550
            vissel = self.wave['vis'] >= 5550
            self.wave[arm] = np.append(self.wave['uvb'][uvbsel], self.wave['vis'][vissel])
            self.onederro[arm] = np.append(self.onederro['uvb'][uvbsel], self.onederro['vis'][vissel])
            self.oneddata[arm] = np.append(self.oneddata['uvb'][uvbsel], self.oneddata['vis'][vissel])

          if arm == 'visnir':
            vissel = self.wave['vis'] < 10000
            nirsel = self.wave['nir'] >= 10000
            self.wave[arm] = np.append(self.wave['vis'][vissel], self.wave['nir'][nirsel])
            self.onederro[arm] = np.append(self.onederro['vis'][vissel], self.onederro['nir'][nirsel])
            self.oneddata[arm] = np.append(self.oneddata['vis'][vissel], self.oneddata['nir'][nirsel])
          
          self.s.soneddata[arm] = ''
          self.s.oneddla[arm]  = ''
          self.s.cont[arm]  = ''
          self.s.model[arm] = ''
            
            
        if self.wave[arm] == []:
            print '\tNeed a spectrum for arm %s'%arm
            return            
        
        c = 2.99792458E5
        # Telluric lines from UVES in air
        telwl = airtovac(np.array(tell(intensity = -0.5))) *\
             ((1 + self.s.vhel/c) / (1 - self.s.vhel/c) )**0.5
        # Telluric lines form Gemini in vacuum
        transwl = (np.array(transNIR(intensity = -.9999))) *\
             ((1 + self.s.vhel/c) / (1 - self.s.vhel/c) )**0.5
#        print len(transwl)
        colors = ['b', 'g', 'r', 'black', 'purple', 'c', 'm', 'coral', 
                  'crimson', 'darkred', 'darksalmon']
        x1 = self.s.wltopix(arm, wl1)
        x2 = self.s.wltopix(arm, wl2)
        chopsize = {7: [16*2/3., 0.06], 5: [14*2/3., 0.10], 3: [11*2/3., 0.14],
                     2 : [9*2/3., 0.17], 1 : [7*2/3., 0.17]}

        if rest == True:
            z = self.s.redshift
        else:
            z = 0


        fig = plt.figure(figsize = (8, chopsize[chop][0]))
        fig.subplots_adjust(bottom=chopsize[chop][1], top=0.95, left=0.14, right=0.97)
        fig.subplots_adjust(hspace=0.22, wspace=0.0)

        splitData = np.array_split(self.oneddata[arm][x1:x2], chop)  
        splitErro = np.array_split(self.onederro[arm][x1:x2], chop)        
        splitWave = np.array_split(self.wave[arm][x1:x2], chop)     

        if len(self.s.oneddla[arm]) > 0:
            splitDLA = \
                np.array_split(self.s.oneddla[arm][0][x1:x2]*self.s.cont[arm][x1:x2], chop)
        if len(self.s.cont[arm]) > 0:
            splitCont = np.array_split(self.s.cont[arm][x1:x2], chop) 
        if len(self.s.model[arm]) > 0:
            splitModel = np.array_split(self.s.model[arm][x1:x2], chop) 
        mult = self.s.mult
        if lim == []:
            lim = chop*[1]

        for i in np.arange(chop)+1:
            if lim[-1] == 1:
                vals = np.sort(splitData[i-1])
                minx = vals[int(0.1 * len(vals))]
                maxx = 2*vals[int(0.9 * len(vals))]
                lim[i-1] = [min(-0.2, minx*mult), maxx*mult]
           
            wav, dat, ero = binspec(splitWave[i-1], splitData[i-1], 
                                    splitErro[i-1], wl = median)
            binsize = wav[1]-wav[0]

            if len(self.s.cont[arm]) > 0:
                wav2, cont = binspec(splitWave[i-1], splitCont[i-1], wl = median)
            if len(self.s.model[arm]) > 0:
                wav2, model = binspec(splitWave[i-1], splitModel[i-1], wl = median)
            if len(self.s.oneddla[arm]) > 0:
                wav2, dla = binspec(splitWave[i-1], splitDLA[i-1], wl = median)

            ax = fig.add_subplot(chop, 1, i)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
            ax.tick_params(axis='both', which='major', labelsize=fs)    
            
            for redi, c in zip(self.s.intsys.keys(), colors):
                prvline, j = 0, 0
                for line in self.s.intsys[redi]: 
                    if linelist.has_key(line):
                        redwl = linelist[line][0] * (1 + redi)
                        if redwl < min(max(wav), wl2) and redwl > max(min(wav), wl1):
                            j += 1
                            horzal, off = 'left', -1.2
                            if (redwl - prvline) > 22 and j != 1:  horzal, off = 'right', 1.2
                            liney = lim[i-1][1]*0.72
                            if j in range (0, 200, 2): liney = lim[i-1][1]*0.87
                            ax.axvline(x=redwl, ymin = 0.6, lw = 1, color= c, alpha = 0.3)
                            if line == 'Lyg_972': linedesc = r'$\rm{Ly}\gamma$'
                            elif line == 'Lyb_1025': linedesc = r'$\rm{Ly}\beta$'
                            elif line == 'Lya_1215': linedesc = r'$\rm{Ly}\alpha$'
                            else: linedesc = r'$\rm{%s}$' %(line.split('_')[0])
                            ax.text((redwl+off)/(1+z), liney, linedesc, 
                                    ha = horzal, color = c)
                            prvline = redwl
                    else:
                        print('%s not found in line list' %line)
            
            prvline, j, = 0, 0
            if isinstance(lines, str) and os.path.isfile(lines):
                f = open(lines, 'r')
                linefile = [g for g in f.readlines() if not g.startswith('#')]
                f.close()
                prvline, j = 0, 0
                for line in linefile:
                    try: linedesc = line.split()[1]
                    except IndexError: linedesc = '?'
                    try: redwl = float(line.split()[0])
                    except IndexError: redwl = 0.
                    j += 1
                    horzal, off = 'left', -1.2
                    if (redwl - prvline) > 22 and j != 1:  horzal, off = 'right', 1.2
                    liney = lim[i-1][1]*0.64
                    if j in range (0, 200, 2): liney = lim[i-1][1]*0.82                        
                    if redwl < min(max(wav), wl2) and redwl > max(min(wav), wl1):
                      ax.axvline(x=redwl, ymin = 0.6, lw = 1.5, 
                               color='#663300', alpha = 0.4)
                      ax.text((redwl+off)/(1+z), liney, r'$\rm{%s}$'%linedesc, 
                              ha = horzal, color ='#663300')
                      prvline = redwl

            else:
              for line in lines:
                j += 1    
                redwl = line[1][0] * (1 + self.s.redshift)
                if line[1][1] > ewlim:
                    if redwl < min(max(wav), wl2) and redwl > max(min(wav), wl1):
                        horzal, off = 'left', 1.2
                        if (redwl - prvline) > 18: horzal, off = 'right', -1.2
                        liney = lim[i-1][1]*0.64
                        if j in range (0, 200, 2): liney = lim[i-1][1]*0.82
                        ax.axvline(x=redwl, ymin = 0.6, lw = 1.5, 
                                   color='#663300', alpha = 0.4)
                        if line[0] == 'Lyg_972': linedesc = r'$\rm{Ly}\gamma$'
                        elif line[0] == 'Lyb_1025': linedesc = r'$\rm{Ly}\beta$'
                        elif line[0] == 'Lya_1215': linedesc = r'$\rm{Ly}\alpha$'
                        elif line[0] == 'Hdelta': linedesc = r'$\rm{H}\delta$'
                        elif line[0] == 'Hgamma': linedesc = r'$\rm{H}\gamma$'
                        elif line[0] == 'Hbeta': linedesc = r'$\rm{H}\beta$'
                        elif line[0] == 'Halpha': linedesc = r'$\rm{H}\alpha$'
                        else: linedesc = r'$\rm{%s}$' %(line[0].split('_')[0])
                        ax.text((redwl+off)/(1+z), liney, linedesc, ha = horzal,
                                color ='#663300')
                        prvline = redwl
            
            if arm == 'uvb':
                ax.axvspan(5650/(1+z), 5720/(1+z), ymin=0.0, ymax=0.05, alpha=0.3,
                        color = '#6688BB')
            
            if norm == 0:
                if unit == 'erg':
                    ax.plot(wav/(1+z), ero*mult, color = 'grey', alpha = 1.0, # rasterized = raster,
                            drawstyle = 'steps-mid',  lw = 0.6, zorder = 1) 
                    ax.plot(wav/(1+z), dat*mult, #ero*mult, binsize/2,
                            color = 'firebrick',# rasterized = raster,
                            drawstyle = 'steps-mid',  lw = 0.8, zorder = 10)#, capsize = 0)
                elif unit == 'Jy':
                    ax.plot(wav/(1+z), ergJy(ero, wav), color = 'grey', alpha = 1.0, # rasterized = raster,
                            drawstyle = 'steps-mid',  lw = 0.6, zorder = 1) 
                    ax.plot(wav/(1+z), ergJy(dat, wav), #ero*mult, binsize/2,
                            color = 'firebrick',# rasterized = raster,
                            drawstyle = 'steps-mid',  lw = 0.8, zorder = 10)#, capsize = 0)
                     
                if self.s.model[arm] != '':
                    ax.plot(wav/(1+z), model*mult, color = 'green', alpha = 0.8,
                                drawstyle = '-',  lw = 1.5)
                if self.s.oneddla[arm] != '':
                    ax.plot(wav/(1+z), dla*mult, color = 'blue', alpha = 0.8,
                        drawstyle = '-',  lw = 0.5, zorder = 11)

            elif norm == 1:
                ax.plot(wav/(1+z), dat/cont, color = 'black', rasterized = raster,
                            drawstyle = 'steps-mid',  lw = 2)
                ax.plot(wav/(1+z), ero/cont, color = 'grey', alpha = 0.8, rasterized = raster,
                            drawstyle = 'steps-mid',  lw = 1) 
                if self.s.model[arm] != '':
                    ax.plot(wav/(1+z), model*mult, color = 'green', alpha = 0.8,
                                drawstyle = '-',  lw = 1.5)
                if self.s.oneddla[arm] != '':
                    ax.plot(wav2/(1+z), dla/cont, color = 'red', alpha = 0.8,
                        drawstyle = '-',  lw = 0.5)

            ax.plot(self.wave[arm]/(1+z), 0*self.wave[arm], color = 'black', lw = 0.1, ls = '--')
            if False:  
                ax.plot(telwl/(1+z), np.array(len(telwl)* [lim[i-1][1]*0.93]), 'o',
                        color = 'yellow', ms = 3.5)
                ax.plot(transwl/(1+z), np.array(len(transwl)* [lim[i-1][1]*0.93]), 'o',
                        color = 'yellow', ms = 6.5)
            if i == (chop/2)+1:
                if unit == 'Jy':
                    ax.set_ylabel(r'$F_{\nu}\,\rm{(\mu Jy)}$')
                elif mult < 1E10:
                    ax.set_ylabel(r'$\rm{Counts}$')
                elif norm == 0:
                    ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$' \
                                %(str(mult))[-2:])
                elif norm == 1:
                    ax.set_ylabel(r'$\rm{Normalized\,flux}$')
            if lim != []:
                ax.set_ylim(lim[i-1][0], lim[i-1][1])
            ax.set_xlim(min(wav)/(1+z)-binsize/2, 
                        max(wav)/(1+z)+binsize/2)#, yerr=self.onederro[arm])
            
            armloc = {'vis':100./chop*7, 'uvb': 50./chop*7, 
            'uvbvis': 100./chop*7, 'nir': 600./chop*3, 'visnir': 200./chop*7, }

            majorLocatorX = plt.MultipleLocator(armloc[arm]/(1+z))
            minorLocatorX = plt.MultipleLocator(armloc[arm]/10/(1+z))

            ax.xaxis.set_major_locator(majorLocatorX)
            ax.xaxis.set_minor_locator(minorLocatorX)
        if rest == True:
            ax.set_xlabel(r'$\rm{Restframe\,wavelength\, (\AA)}$')
        else:
            ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')

        fig.savefig('%s_1d_%s.pdf' %(self.s.object, arm))
#        fig.savefig('%s_1d_%s.eps' %(self.s.object, arm))#, rasterized = raster, dpi=50)
        plt.close(fig)
################################################################################
        
    def plot2d(self, arm = '', y1 = 0, y2 = -1, cmap = 'afmhot_r', xsc = 'AA',
               vlim1 = -0.1, vlim2 = 0.8, linename = 'Halpha',
               z1 = '', z2 = '', z3 = '',
               zfix1 = False, zfix2 = False, zfix3 = False,
               basefix = '', gh1 = '',
               bfix = True, 
               tieg = '', 
               fwhmfix1 = '', fwhmfix2 = '', fwhmfix3 = '',
               ng = 1, nx1 = 28, nx2 = 28, excl = [], out = 'pdf', 
               fs = 25, radiamax = 8E2, tellmax = 0.85):
        """ Plot and fit an emision line to the data, parameters are:
           nx1, nx2 (wl range +/- to plot, default nx1 = nx2 = 15)
           y1, y2 (datarange in pixel in spatial direction, default all)
           xsc (Scale of plt, default AA, option kms)
           z1, z2, z3 (redshifts of Gaussian centroids to fix), default fix)
           ng (number of Gaussians to fit (1, 2, 3), default: 1)
           excl (List of data pairs to exclude from the fitting)
           fwhmfix1, fwhmfix2, fwhmfix3 (fixed fwhm of line in kms - corr. for resolution)
           basefix (fixed basline of Gaussian fit in erg/cm^2/s/AA)
           vlim1, vlim2 (color scaling of 2d-plot)"""
        
        limitmin = [True] + 3*[False, True, True]
        self.linepars[linename] = {}      
        if z1 == '': z1 = self.s.redshift
        if z2 == '': zmean = z1
        if z3 == '': zmean = z1

        if arm == '':
            try:
                obswl = emllist[linename][0]*(1+float(z1))
            except KeyError:
                print '\tERROR: Do not know line %s' %linename
            if 3200 < obswl < 5630: arm = 'uvb'
            elif obswl < 10150: arm = 'vis'
            elif obswl < 25000: arm = 'nir'
            else:
                print '\tERROR: %s not in X-shooter spectral range' %linename
                return

        if fwhmfix1 != '' and fwhmfix1 > 10:
            fwhmfix1 = (fwhmfix1**2 + self.s.reso[arm]**2)**0.5
        if fwhmfix2 != '' and fwhmfix2 > 10:
            fwhmfix2 = (fwhmfix2**2 + self.s.reso[arm]**2)**0.5
        if fwhmfix3 != ''and fwhmfix3 > 10:
            fwhmfix3 = (fwhmfix3**2 + self.s.reso[arm]**2)**0.5
#            
        if len(self.oneddata[arm]) == 0:
            print '\tERROR: Need to have 1d spectrum for %s arm' %(arm.upper()) 
            return
        #print max(self.wave[arm])
        print '\n\t############################'
        print '\tEmission line analysis: %s in %s arm' %(linename, arm.upper())
        print '\t############################'

        if z1 != '' and z2 != '': zmean = z1/2. + z2/2.
        if z1 != '' and z2 != '' and z3 != '': zmean = z1/3. + z2/3. + z3/3.

        un = str(self.s.mult)[-2:]
        majorLocator = plt.MaxNLocator(4)
        minorLocator = plt.MaxNLocator(20)
        xlab = r'$\rm{Observed\,wavelength\, (\AA)}$'
        ylab = r'$F_{\lambda}\,\rm{(10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$' %un

        if y1 == 0: y1 = 0
        if y2 == -1: y2 = len(self.s.data[arm][0]) - 1

        x1 = self.s.wltopix(arm, emllist[linename][0]*(1 + zmean) - nx1)
        x2 = self.s.wltopix(arm, emllist[linename][0]*(1 + zmean) + nx2)
   
        x110 = self.s.wltopix(arm, emllist[linename][0]*(1 + zmean) - 10)
        x210 = self.s.wltopix(arm, emllist[linename][0]*(1 + zmean) + 10)     

        dl = (self.wave[arm][-1]-self.wave[arm][0]) / (len(self.wave[arm]) - 1)

        if self.s.skywl != []:
            tellsel =  (emllist[linename][0]*(1+zmean) - nx1 < self.s.skywl) * \
                   (emllist[linename][0]*(1+zmean) + nx2 > self.s.skywl)
            skywlsel = np.array(self.s.skywl[tellsel])
            skytranssel = np.array(self.s.skytrans[tellsel])
            skyradiasel = np.array(self.s.skyradia[tellsel])
            tell, skyline = 0, 0
            for skywl, skyt, skyr in zip(skywlsel, skytranssel, skyradiasel):
                if arm in ['nir', 'vis']:
                    if skyt < tellmax and tell == 0:
                        tell, telexl = 1, [skywl - 0.25]
                    if skyt > tellmax and tell == 1:
                        telexl.append(skywl + 0.25)
                        tell = 0
                        excl.append(telexl)
                if skyr > radiamax and skyline == 0 and skywl > 5000:
                    skyline, skyexl = 1, [skywl - 0.25]
                if skyr < radiamax and skyline == 1 and skywl > 5000:
                    skyexl.append(skywl + 0.25)
                    skyline = 0
                    excl.append(skyexl)
                if skywl == skywlsel[-1]:
                    if skyline == 1:
                        skyexl.append(skywl + 0.25)
                        excl.append(skyexl)
                    if tell == 1:
                        telexl.append(skywl + 0.25)
                        excl.append(telexl)

        xful = self.wave[arm][x1:x2]

        wlsel  = np.array(len(self.wave[arm][x1:x2])*[True])
        if excl != [] and True:
            for exc in excl:
                exclsel1 = self.wave[arm][x1:x2] < exc[0] 
                exclsel2 = self.wave[arm][x1:x2] > exc[1]
                exclsel = exclsel1 ^ exclsel2
                wlsel *= exclsel

        radiareg = self.s.skyrad[arm][x110:x210]
        tellreg = self.s.skytel[arm][x110:x210]
        sellreg = (radiareg < radiamax) * (tellreg > tellmax)
        telltransavg = np.average(tellreg[sellreg]) 
        tellmult = 1 / telltransavg
        tellmerr = (1 / telltransavg - 1)/2.
        telsrele = (tellmerr)**2
        self.linepars[linename]['tellmult'] = [tellmult, tellmerr]

        x = self.wave[arm][x1:x2][wlsel]
        xavg = np.average(xful)
        
        # Region for background
        backsel = (self.wave[arm] > (xavg - 75/dl)) * (self.wave[arm] < (xavg + 75/dl))  
        backsel *= (self.wave[arm] > (xavg + 10)) ^ (self.wave[arm] < (xavg - 10))  

        mediandat = np.array(self.oneddata[arm][backsel])
        tellmed = np.array(self.s.skytel[arm][backsel])
        radiamed = np.array(self.s.skyrad[arm][backsel])
        if arm in ['vis', 'nir']:
            mediandat = mediandat[(tellmed > 0.85) * (radiamed < 1E3)]
            
        while True:
            bguess = np.median(mediandat)
            brms = np.std(mediandat)
            clip = (mediandat > (bguess - 3*brms)) * (mediandat < (bguess + 3*brms))
            if clip.all() == True:
                break
            mediandat = mediandat[clip]
#        print bguess    
        bguess *= self.s.mult
        berr = np.std(bguess)*self.s.mult/len(mediandat)**0.5
        y = self.oneddata[arm][x1:x2][wlsel]*self.s.mult
        yerr = self.onederro[arm][x1:x2][wlsel]*self.s.mult

        if basefix != '':
            bguess = basefix
            berr = 0
        bguess = max(1E-4, bguess)

        xmin, xmax = xful[0] - 0.5*dl, xful[-1] + 0.5*dl
        if y == []:
            print '\tAll wavelengths excluded'
            return
            
        sigmafix1, sigmafix2, sigmafix3, bfix = 0, 0, 0, bfix
        sigmain1, sigmain2, sigmain3 = 3, 3, 3
        wlmean1, wlmean2, wlmean3 = x[len(x)/2], x[len(x)/2]+4, x[len(x)/2]+8
        
        if fwhmfix1 != '': 
            sigmain1, sigmafix1 = fwhmfix1/c * np.median(x)/2.3548, 1
            if fwhmfix1 < 10:
                sigmain1 = fwhmfix1/2.3548
        if fwhmfix2 != '': 
            sigmain2, sigmafix2 = fwhmfix2/c * np.median(x)/2.3548, 1
            if fwhmfix2 < 10:
                sigmain2 = fwhmfix2/2.3548
        if fwhmfix3 != '': 
            sigmain3, sigmafix3 = fwhmfix3/c * np.median(x)/2.3548, 1
            if fwhmfix3 < 10:
                sigmain3 = fwhmfix3/2.3548

        if z1 != '': wlmean1 = emllist[linename][0]*(1+z1)
        if z2 != '': wlmean2 = emllist[linename][0]*(1+z2)
        if z3 != '': wlmean3 = emllist[linename][0]*(1+z3)

        if ng == 1: 
            params = onedgaussfit(x, y, err = yerr, 
                              params = [bguess, 0, wlmean1, sigmain1],
                              fixed =  [bfix, 0, zfix1, sigmafix1],
                              limitedmin = limitmin) 
        elif ng == 2:
            params = onedtwogaussfit(x, y, err = yerr, 
              params=[bguess, 0, wlmean1, sigmain1, 0, wlmean2, sigmain2],
              fixed = [bfix, 0, zfix1, 0, 0, zfix2, 0],
              limitedmin = limitmin)       
           # print '\tChi2 = %.2f for %i d.o.f => red. chi^2: %.2f'\
           #     %(params[3][0], params[3][1], params[3][0]/params[3][1])
  
        elif ng == 3:
            params = onedthreegaussfit(x, y, err = yerr, 
              params=[bguess, 0, wlmean1, sigmain1, 
                              0, wlmean2, sigmain2,
                              0, wlmean3, sigmain3],
              fixed = [bfix,  0, zfix1, 0, 
                              0, zfix2, 0,
                              0, zfix3, 0],
              limitedmin = limitmin)       
            #print '\tChi2 = %.2f for %i d.o.f => red. chi^2: %.2f'\
            #    %(params[3][0], params[3][1], params[3][0]/params[3][1])

        if z1 == '': wlmean1 = params[0][2]
        if fwhmfix1 == '': sigmain1 = params[0][3]
        if gh1 == '': ghin1, hfix1 = params[0][1], 0
        else: ghin1, hfix1, bfix = float(gh1), 1, 0
            
        if ng > 1  :
            if z2 == '': wlmean2 = params[0][5]
            if fwhmfix2 == '': sigmain2 = params[0][6]
        if ng > 2  :
            if z3 == '': wlmean3 = params[0][8]
            if fwhmfix3 == '': sigmain2 = params[0][9]

        if ng == 1 and linename not in ['[OII](3726)', '[OII](3729)']: 
            params = onedgaussfit(x, y-bguess, err = yerr, 
                      params = [0, ghin1, wlmean1, sigmain1],
                      fixed = [bfix, hfix1, zfix1, sigmafix1], quiet = True,
                      limitedmin = limitmin)
        ms, As = [], []
        if ng == 2 or linename in ['[OII](3726)', '[OII](3729)']:
            if tieg != '':
                tied = ['','','','','p[1]/%.3f' %float(tieg),'','']
            else:
                if tieg == 1:
                    tieg = 0.9999
                tied = ['','','','','','','']

            if linename == '[OII](3726)':
                tied[-2:] = ['p[2]/3727.092*3729.875','p[3]']
                self.linepars['[OII](3729)'] = {}        

            elif linename == '[OII](3729)':
                tied[-2:] = ['p[2]/3727.092*3729.875','p[3]']
                self.linepars['[OII](3726)'] = {}        

            params = onedtwogaussfit(x, y-bguess, err = yerr, tied = tied,
                      params=[0, 0, wlmean1, sigmain1, 1, wlmean2, sigmain2], 
                      fixed = [bfix, 0, zfix1, sigmafix1, 0, zfix2, sigmafix2],
                      limitedmin = limitmin) 
                      
            A2, m2, s2 = params[0][4], params[0][5], params[0][6]
            Ae2, me2, se2 = params[2][4], params[2][5], params[2][6]
            ms.append(m2), As.append(A2)
            
        if ng == 3:
            tied = ['','','','','','','','','','']
            params = onedthreegaussfit(x, y-bguess, err = yerr, tied = tied,
                      params=[0, 0, wlmean1, sigmain1, 
                                  0, wlmean2, sigmain2,
                                  0, wlmean3, sigmain3], 
                      fixed = [bfix, 0, zfix1, sigmafix1, 
                               0, zfix2, sigmafix2,
                               0, zfix3, sigmafix3],
                      limitedmin = limitmin) 
            
            A2, m2, s2 = params[0][4], params[0][5], params[0][6]
            Ae2, me2, se2 = params[2][4], params[2][5], params[2][6]
            A3, m3, s3 = params[0][7], params[0][8], params[0][9]
            Ae3, me3, se3 = params[2][7], params[2][8], params[2][9]
            ms.append(m2), As.append(A2)
            ms.append(m3), As.append(A3)

        print '\tChi2 of Gauss fit: %.2f for %i d.o.f' %(params[3][0], params[3][1])
        print '\tBaseline = %.3f +/- %.3f 10^-%s erg/cm^2/s/AA' \
                %(bguess, berr, un)
        print '\tBaseline estimation uses %i wavelengths' %len(mediandat)
        baselJy = bguess / c * params[0][2]**2 * self.s.mult / 1E18
#        print '\tBaseline = %.2f +/- %.2f muJy' \
#                %(baselJy, berr/bguess*baselJy)
        print '\tBaseline = %.2f +/- %.2f mag AB' \
                %(fluxab(baselJy), fluxaberr(baselJy, berr/bguess*baselJy))
        print '\tTelluric multiplier: %.3f\n' %(tellmult)
        self.linepars[linename]['Chi2/d.o.f'] = ['%.3f/%i' %(params[3][0], params[3][1])]
        b, A, m, s = bguess, params[0][1], params[0][2], params[0][3]
#        basel -= b
        ms.append(m), As.append(A)
        be, Ae, me, se = params[2][0], params[2][1], params[2][2], params[2][3]
        if ng in [1, 2, 3]:
            fl, fle, prof1   = self.gausspar(arm, A, m, s, Ae, me, se, un, 
                                       linename, params[-1])
            relse = ((fle/fl)**2 + telsrele)**0.5
            self.linepars[linename]['lf_gauss'] = [fl*tellmult, relse*fl*tellmult]
            sse = (se/s)**2
                       
        if ng in [2, 3] or linename in ['[OII](3726)', '[OII](3729)']:
            if linename == '[OII](3726)':
                print '\t############################'
                print '\t\tLine: [OII](3729)'
                print '\t############################'                
                fl2, fle2, prof2 = self.gausspar(arm, A2, m2, s2, Ae2, me2, se2, un,
                                         '[OII](3729)', params[-1])
                relse2 = ((fle2/fl2)**2 + telsrele)**0.5
                self.linepars['[OII](3729)']['lf_gauss'] = [fl2*tellmult, relse2*fl2*tellmult]
                self.linepars['[OII](3729)']['Chi2/d.o.f'] = \
                    ['%.3f/%i' %(params[3][0], params[3][1])]

            elif linename == '[OII](3729)':
                print '\t############################'
                print '\t\tLine: [OII](3726)'
                print '\t############################' 
                fl2, fle2, prof2 = self.gausspar(arm, A2, m2, s2, Ae2, me2, se2, un,
                                         '[OII](3726)', params[-1])
                relse2 = ((fle2/fl2)**2 + telsrele)**0.5
                self.linepars['[OII](3726)']['lf_gauss'] = [fl2*tellmult, relse2*fl2*tellmult]
                self.linepars['[OII](3726)']['Chi2/d.o.f'] = \
                    ['%.3f/%i' %(params[3][0], params[3][1])]
            else:
                fl2, fle2, prof2 = self.gausspar(arm, A2, m2, s2, Ae2, me2, se2, un,
                                         linename, params[-1])
                totfl = fl+fl2
                relse2 = ((fle2/totfl)**2 + (fle/totfl)**2 + telsrele)**0.5
                self.linepars[linename]['lf_gauss'] = \
                    [(fl2+fl)*tellmult, (fl2+fl)*tellmult*relse2]
                sse += (se2/s2)**2
            print '\t############################' 
            print '\tLineflux (summed)= %.2f +/- %.2f 10^-%s erg/cm^2/s' \
                    %(fl2+fl, (fle**2 + fle2**2)**0.5, un)
        if ng == 3:  
            fl3, fle3, prof3 = self.gausspar(arm, A3, m3, s3, Ae3, me3, se3, un,
                                      linename, params[-1])
            print '\t############################' 
            print '\tLineflux (summed)= %.2f +/- %.2f 10^-%s erg/cm^2/s' \
                    %(fl2+fl+fl3, (fle**2 + fle2**2 + fle3**2)**0.5, un)
            totfl = fl2+fl+fl3
            relse3 = ((fle3/totfl)**2 + (fle2/totfl)**2 + (fle/totfl)**2 + telsrele)**0.5
            self.linepars[linename]['lf_gauss'] = \
                [(fl2+fl+fl3)*tellmult, (fl2+fl+fl3)*tellmult*relse3]
            sse += (se3/s3)**2
        ewobs = self.linepars[linename]['lf_gauss'][0] / bguess
        ewerr = self.linepars[linename]['lf_gauss'][1] / bguess

        print '\tEW (obs) = %.1f +/- %.1f AA' %(ewobs, ewerr)
        print '\tEW (rest) = %.1f +/- %.1f AA' \
            %(ewobs/(1+self.s.redshift), ewerr/(1+self.s.redshift))
        retax, retfit = params[-1], params[1]
        # Find good limits for integration
        #wlmaxflux = retax[list(retfit).index(max(retfit))]
        totflux = integrate.trapz(abs(retfit), retax) 
        fluxfrac, wlintstart, wlintend, fluxcor, j = 0, x[0], x[-1], 1, 0
        for i in range(len(retax)):
            if i > 1:
                fluxfrac = integrate.trapz(abs(retfit[j:i]), retax[j:i])/totflux
            if fluxfrac < 0.16: wlsig1 = retax[i]
            if fluxfrac < 0.84: wlsig2 = retax[i]
                
            if abs(retfit[i]) > 0.25*max(abs(retfit)) and wlintstart == x[0]:
                wlintstart = retax[i]
                j = i
            if abs(retfit[i]) < 0.25*max(abs(retfit)) and wlintend == x[-1] and fluxfrac > 0.5:
                wlintend = retax[i]
                fluxcor = fluxfrac
        dvfwhm = (wlsig2-wlsig1)/(wlsig2/2.+wlsig1/2.)*c/2.*2.3548
        dvsig = dvfwhm/2.3548
        self.linepars[linename]['vel_disp'] = \
            [((dvfwhm**2 - self.s.reso[arm]**2)**0.5)/2.3548, dvsig * sse**0.5]
        ilmin = max(0,abs(x-wlintstart).argmin() - 4)
        ilmax = min(abs(x-wlintend).argmin() + 4, len(x)-1)

        lineflints, n = [], 1000
        for i in range(n):
            dataerr = np.random.normal(y[ilmin:ilmax] - bguess, yerr[ilmin:ilmax])/fluxcor
            lineflint = integrate.trapz(dataerr, x[ilmin:ilmax]) 
            lineflints.append(lineflint)
        lfib, lfin, lfip, lf3s = errs(lineflints, ul=3)
        print '\tLineflux (numint)= %.2f +%.2f %.2f 10^-%s erg/cm^2/s' \
        %(lfib, lfip, lfin, un)
        print '\tLineflux (3s upper limit)= %.2f 10^-%s erg/cm^2/s' %(lf3s, un)
        try:
            print '\tUsed wavelength %.1f - %.1f \AA' %(x[ilmin], x[ilmax])
        except IndexError:
            print '\tNo wavelengths left after exclusion'
        print '\t############################' 

        rellfip = ((lfip/lfib)**2 + telsrele)**0.5
        rellfin = ((lfin/lfib)**2 + telsrele)**0.5
        if linename in ['[OII](3729)','[OII](3726)'] : 
            self.linepars['[OII](3729)']['lf_numint'] = \
            [lfib*tellmult, rellfip*lfib*tellmult, rellfin*lfib*tellmult]
            self.linepars['[OII](3726)']['lf_numint'] = \
            [lfib*tellmult, rellfip*lfib*tellmult, rellfin*lfib*tellmult]
        else:
            self.linepars[linename]['lf_numint'] = \
            [lfib*tellmult, rellfip*lfib*tellmult, rellfin*lfib*tellmult]

        m = np.average(ms)
        xmin2, xmax2, xran = (xmin-m) / m*c, (xmax-m) / m*c, (x-m) / m*c
        skywlplot = self.s.skywl    
        
        ymin, ymax = min(y - 1.2*yerr), 1.05*max(max(y + yerr), max(retfit + bguess))
        extent = (xmin, xmax, ymin, ymax)

        if xsc == 'kms':
            majorLocator = plt.MaxNLocator(8)
            minorLocator = plt.MaxNLocator(16)

            xful = (xful - m)/m*c
            extent = (xmin2, xmax2, ymin, ymax)
            xmin, xmax, x = xmin2, xmax2, xran
            retax = (retax-m) / m*c            
            xlab = r'$\rm{Velocity\,(km\,s^{-1})}$'
            instsig = self.s.reso[arm]/(2*(2*np.log(2)**0.5))
            instsig2 = c/5400./(2*(2*np.log(2)**0.5))
            if excl != []:
                for i in range(len(excl)):
                    excl[i][0] = (excl[i][0]-m)/m*c
                    excl[i][1] = (excl[i][1]-m)/m*c
            if self.s.skywl != []:
                skywlplot = (self.s.skywl-m) / m*c   
        
#==============================================================================
# Figure        
#==============================================================================
        
        fig = plt.figure(figsize = (9.5, 5.3))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        fig.subplots_adjust(bottom=0.15, top=0.98, left=0.14, right=0.99)
        
        if len(self.s.smooth[arm]) > 0:
            pxscale =  self.s.head[arm]['CDELT2']
            cax = ax.imshow(self.s.smooth[arm][x1:x2].transpose()[y1:y2]*self.s.mult/pxscale, 
                  cmap = cmap, interpolation='nearest',
                  vmax = vlim2 + 0.03, vmin = vlim1 - 0.03,
                  origin = 'lower', extent = extent, aspect="auto")
        
        elif len(self.s.data[arm]) > 0:
            pxscale =  self.s.head[arm]['CDELT2']
            cax = ax.imshow(self.s.data[arm][x1:x2].transpose()[y1:y2]*self.s.mult/pxscale, 
                  cmap = cmap, interpolation='nearest',
                  vmax = vlim2 + 0.03, vmin = vlim1 - 0.03,
                  origin = 'lower', extent = extent, aspect="auto")  
        
        
        if len(self.s.smooth[arm]) > 0 or len(self.s.data[arm]) > 0:
            #ticks = arange(vlim1, vlim2+round(vlim2/5.,2), round(vlim2/5.,2) )
            #cbar = fig.colorbar(cax, ticks=ticks, orientation='vertical', pad=0.10)
            cbar = fig.colorbar(cax, orientation='vertical', pad=0.12, shrink=0.95)
            cbar.set_label(r'$\rm{10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1}\, arcsec^{-1}}$' %un,
                           fontsize = fs-2)
            cbar.locator = plt.MaxNLocator( nbins = 5)        
            cbar. update_ticks()
            ax2 = ax.figure.add_axes(ax.get_position(), frameon = False, sharex = ax)
            ax2.yaxis.tick_right()
            ax.yaxis.tick_left()
            yzero = np.polyval(self.s.trace[arm][0], np.average(x)) 
            ax2.set_ylim([(y1-yzero) * pxscale, (y2 - yzero) * pxscale])
            ax2.yaxis.set_label_position("right")
            ax2.set_ylabel(r'$\rm{Spatial\,position\,(^{\prime\prime})}$', fontsize = fs)
        
        if excl != []:
            for exc in excl:
                ax.axvspan(xmin = exc[0], xmax = exc[1],
                           color = 'white', alpha = 0.7)
        
        if self.s.skywl != []:
            ax.plot(skywlplot, self.s.skytrans*(ymax-ymin)+ymin, '-', color = 'black', 
                    lw = 1.0)
            #ax.fill_between(self.s.skywl, ymax, self.s.skytrans*ymax, 
            #                color = 'black', alpha = 0.5)
            
        ax.plot(retax, retfit + bguess, color = 'white', lw = 3.5)
        ax.plot(retax, retfit + bguess, color = 'black', lw = 1.5)
        
        ax.errorbar(x, y, yerr, capsize = 0, color = 'black', fmt = 'o',  
                    mec = 'lightgrey', lw = 1.5, mew = 0.3)

        ax.errorbar(x[ilmin:ilmax], y[ilmin:ilmax], yerr[ilmin:ilmax], 
                    capsize = 0, color = 'black', fmt = 'o',  
                    mec = 'darkgrey', lw = 1.5, mew = 0.9)
        
        ax.errorbar(xful[-wlsel], 
                    self.oneddata[arm][x1:x2][-wlsel]*self.s.mult, 
                    self.onederro[arm][x1:x2][-wlsel]*self.s.mult,
                    capsize = 0, color = 'grey', fmt = 'o',  
                    mec = 'black', lw = 1.5)
        
        if ng == 2 or linename in ['[OII](3726)', '[OII](3729)']:
            ax.plot(retax, prof2 + bguess, '--', color = 'grey', lw = 1.5)
            ax.plot(retax, prof1 + bguess, '--', color = 'grey', lw = 1.5)
        if ng == 3:
            ax.plot(retax, prof2 + bguess, '--', color = 'grey', lw = 1.5)
            ax.plot(retax, prof1 + bguess, '--', color = 'grey', lw = 1.5) 
            ax.plot(retax, prof3 + bguess, '--', color = 'grey', lw = 1.5) 

        if linename.startswith('H') and linename[3] is not 'I':
            plt.figtext(0.69, 0.9, r'$\rm{H}\%s$'%linename[1:], fontsize = 26, ha = 'center', 
                va = 'center', color = 'black', weight='bold', **kwargs)
        elif linename in ['[OII](3726)', '[OII](3729)']:
            plt.figtext(0.62, 0.9, r'$\rm{[OII]\,(\lambda 3727)}$', fontsize = 26, ha = 'center', 
                va = 'center', color = 'black', weight='bold', **kwargs)
        elif linename.startswith('He'):
            plt.figtext(0.69, 0.9, r'$\rm{HeII}$', fontsize = 26, ha = 'center', 
                va = 'center', color = 'black', weight='bold', **kwargs)
        elif linename == 'Lyalpha':
            plt.figtext(0.69, 0.9, r'$\rm{Ly}\alpha$', fontsize = 26, ha = 'center', 
                va = 'center', color = 'black', weight='bold', **kwargs)
        else:
            plotname = linename.split('](')[0] + ']'
            plotname +=  '\,(\lambda' + linename.split('](')[1]
            plt.figtext(0.62, 0.9, r'$\rm{%s}$'%plotname, fontsize = 26, ha = 'center', 
                va = 'center', color = 'black', weight='bold', **kwargs)
            #ax.vlines(x = mo, ymin = ymin + 0.15, ymax = Ao + basel - 0.02, lw = 1.5, 
            #          color='grey', linestyles='dashed') 
                  
        if linename == '[OII](3726)':
            ax.text(m2, ymin + 0.2, r'$[OII](\lambda\,3729)$', fontsize = 18, ha = 'center', 
                va = 'center', color = 'grey', weight='bold', **kwargs)
            ax.vlines(x=m2, ymin = ymin + 0.25, ymax = A2 + bguess - 0.02, lw = 1.5,
                      color='grey', linestyles='dashed') 
        if xsc == 'kms':
            instprof1 = onedgaussian(retax, bguess, As[0], (ms[0]-m) / m*c, instsig)
            instprof2 = onedgaussian(retax, bguess, As[0], (ms[0]-m) / m*c, instsig2)
            ax.fill_between(retax, instprof1, instprof2, 
                            color = 'grey', alpha = 0.8)
        if False:        
            ax.plot(x, self.s.soneddata[arm][x1:x2]*self.s.mult, 
                color = 'black', lw = 1.5)
        ax.plot(x, x*0+bguess, '-', color = 'black')
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator) 
        ax.set_xlim(xmin, xmax) 
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(ylab, fontsize = fs), ax.set_xlabel(xlab, fontsize = fs)
#        ax.ticklabel_format(useOffset = False)
        linesaven = linename
        for a in '()[]':
            linesaven = linesaven.replace(a,'')
        fig.savefig('%s_%s_%s_2d.%s' %(self.s.object, linesaven, arm, out), 
                    format = out)
        plt.close(fig)
        
################################################################################
    def writelines(self):
         f = open('%s_lines.txt' %self.s.object, 'w')
         f.write('#Linename')#\tVel.disp(km/s)\tError+(km/s)\tError-(km/s)\tRedshift')
                  
         f.write('\tFluxG\tFluxE\n')
         for key in self.linepars.keys():
             if key.startswith('sl'):
                 f.write('Slitloss correction: %s ARM, %s band: %.2f+/-%.2f\n'\
                     %(self.linepars[key]))
         for key in self.linepars.keys():
             if not key.startswith('sl'):
                 f.write('%s\t' %key)
                 if key.startswith('H'):
                     f.write('\t')
                 for key2 in ['z1', 'z2', 'z3' , 's1', 's2', 's3',
                      'lf_gauss', 'lf_numint', 'Chi2/d.o.f', 'vel_disp',
                      'tellmult']:
                      if self.linepars[key].has_key(key2):
                         fmt = '%.3f\t'
                         if key2.startswith('z'):
                             fmt = '%.5f\t'
                         if key2.startswith('Chi2'):
                             fmt = '%s\t'
                         fileent = [fmt%a for a in self.linepars[key][key2]]
                         f.write('%s' % ('\t'.join(fileent)))
                 f.write('\n')
         f.close()            
         
################################################################################            
    def gausspar(self, arm, A, m, s, Ae, me, se, un, linename, bins):
        n = 1        
        if self.linepars.has_key(linename):
            if self.linepars[linename].has_key('Redshift1'):
                n = 2
            if self.linepars[linename].has_key('Redshift2'):
                n = 3
        print '\tGauss amplitude = %.2f +/- %.2f 10^-%s erg/cm^2/s/AA' %(A, Ae, un)
        print '\tGauss mean = %.2f +/- %.2f AA' %(m, me)
        print '\tGauss sigma = %.2f +/- %.2f AA' %(s, se)
        print '\tGauss FWHM = %.2f +/- %.2f AA' %(2.3548*s, 2.3548*se)
        measg = 2.3548*s/m*c
        measge = 2.3548*se/m*c
        print '\tGauss FWHM = %.2f +/- %.2f km/s' %(measg, measge)
        if self.s.reso[arm] != '' and (measg-measge) > self.s.reso[arm]:
            intrg = (measg**2-self.s.reso[arm]**2)**0.5    
            intrgm = intrg - ((measg-measge)**2-self.s.reso[arm]**2)**0.5
            intrgp = ((measg+measge)**2-self.s.reso[arm]**2)**0.5 - intrg
            print '\tGauss FWHM (corr. for R) = %.1f +%.1f -%.1f km/s' \
                    %(intrg, intrgp, intrgm)
            print '\tVel. dispersion (corr. for R) = %.1f +%.1f -%.1f km/s' \
                    %(intrg/2.3548, intrgp/2.3548, intrgm/2.3548)
            self.linepars[linename]['s%i'%n] = [intrg/2.3548, intrgp/2.3548, intrgm/2.3548] 
        else:
            self.linepars[linename]['s%i'%n] = [-99, -99, -99] 
        linefl = A*s*(3.1416*2)**0.5
        linefle = ((Ae/A)**2 + (se/s)**2)**0.5*linefl
        print '\tLineflux = %.2f +/- %.2f 10^-%s erg/cm^2/s' %(linefl, linefle, un)
        reds = m / emllist[linename][0] - 1
        redse = me / m * reds
        print '\tRedshift = %.6f +/- %.6f' %(reds, redse)
        self.linepars[linename]['z%i'%n] = [reds, redse]
        prof = (2*3.1416)**0.5*s * A*mlab.normpdf( bins, m, s )
        return linefl, linefle, prof 
################################################################################        
    def plotLine(self, linename, ign = [], u = 'km/s', 
                 Aw = 20, Awm = '', z = '', FWHM = 1, fixz = False,
                 cpm = 10, cpp = 10, verbose = 0):
        if z == '':  z = float(self.s.redshift)
        
        try:
            redwl = linelist[linename][0] * (1 + z)
        except KeyError:
            if verbose > 0:
                print '\t\tLine %s not known' %linename
            return            
            
        if 3100 < redwl < 5500: arm = 'uvb'
        elif 5500 < redwl < 10000: arm = 'vis'
        elif 10000 <  redwl < 25000: arm = 'nir'
        else:
            if verbose > 0:
                print 'Lines %s not in wavelength response' %linename
            return             
        
        if self.oneddata[arm] == '':
            if verbose > 0:
                print 'Arm %s not available' %arm
            return                
        
        if Awm == '':  Awm = Aw
        fixfwhm = False
        if FWHM != 1: fixfwhm = True
        sigma = FWHM / 2.3548
        
        dl = (self.wave[arm][-1]-self.wave[arm][0]) / (len(self.wave[arm]) - 1)
        if linename =='MgII_2796' and ign==[]:
            ign = [linelist['MgII_2803'][0] * (1 + z) - 5,
                   linelist['MgII_2803'][0] * (1 + z) + 5]    
        
        
        x0 = int((redwl-self.wave[arm][0])/dl)
        x1, x2 = int(x0 - Awm/dl), int(x0 + Aw/dl)
        
        x = self.wave[arm][x1:x2]
        y = self.s.mult * self.oneddata[arm][x1:x2]
        yerr = self.s.mult * self.onederro[arm][x1:x2]

        if ign != []:
            xig1 = int((ign[0]-self.wave[arm][0])/dl)
            xig2 = int((ign[1]-self.wave[arm][0])/dl)
            fitx_a = np.array(self.wave[arm])
            fity_a = np.array(self.s.mult * self.oneddata[arm])
            fite_a = np.array(self.s.mult * self.onederro[arm])
            fitx = np.array(list(fitx_a[x1:xig1])+list(fitx_a[xig2:x2]))
            fity = np.array(list(fity_a[x1:xig1])+list(fity_a[xig2:x2]))
            fite = np.array(list(fite_a[x1:xig1])+list(fite_a[xig2:x2]))
        else:
            fitx_a = np.array(self.wave[arm])
            fity_a = np.array(self.s.mult * self.oneddata[arm])
            fite_a = np.array(self.s.mult * self.onederro[arm])            
            fitx, fity, fite = np.array(x), np.array(y), np.array(yerr)
    
        xc1 = 5/dl
        cont = np.append(fity[cpm : int(len(x)/2-xc1) ],  fity[int(len(x)/2+xc1) : -cpp])
        cont = np.median(cont)

        if u == 'km/s':
            fitx = (fitx - redwl) / redwl * c
            fitx_a = (fitx_a - redwl) / redwl * c
            x = (x - redwl) / redwl * c
            xl = r'$\rm{Velocity\, (km\,s^{-1})}$'
            xmin, xmax = -1* Awm * 30,  Aw * 30
            xmid, xsig = 0, sigma/redwl * c
            dl = (fitx[-1] - fitx[0])/len(fitx)
            skywlplot = self.s.skywl / redwl * c

        elif u == 'AA':
            xl = r'$\rm{Observed\, wavelength\, (\AA)}$'
            xmin = redwl - Awm
            xmax = redwl + Aw
            xmid, xsig = redwl, sigma
            skywlplot = self.s.skywl   

        
        ewint = []
        for i in range(3000):
            dint = np.random.normal(fity, fite)
            inte = sum(cont - dint) * dl
            ewint.append(inte/cont)


        print '\t###########################'
        print '\tLine %s' %linename
        ewintb, ewintm, ewintp, ew2s = errs(ewint)

        if verbose > 0:
            print '\tContinuum [erg/cm^2/s]: %.2f' %(cont)
            print '\tEW (obs) / Integrated [AA or kms]: %.2f + %.2f %.2f' %(ewintb, ewintp, ewintm)
            print '\tEW (red) / Integrated [AA or kms]: %.2f + %.2f %.2f' \
            %(ewintb/(1+z), ewintp/(1+z), ewintm/(1+z))
        
        params = onedgaussfit(fitx, fity, err = fite, 
                params=[cont, -cont, xmid, xsig],
                fixed=[False, True, fixz, fixfwhm])   
    
        mean, meane = params[0][2], params[2][2]
        con, cone = params[0][0], params[2][0]
        a, ae = params[0][1], params[2][1]
        s, se = params[0][3], params[2][3]
#            print '\tContinuum %.2f +/- %.2f' %(con, cone)
        if verbose > 0:
            print '\tAmplitude %.2f +/- %.2f' %(a, ae)
            print '\tMean: %.2f +/- %.2f %s' %(mean, meane, u)
    
            print '\tSigma: %.2f +/- %.2f %s' %(s, se, u)
#            print '\tFWHM: %.2f +/- %.2f %s' %(s*2.3548, se*2.3548, u)
        if u == 'AA':
            reds = mean/linelist[linename][0]-1
            print '\tRedshift: %.6f +/- %.6f' %(reds, (meane/mean)*reds)
            if verbose > 0:
                print '\tIntegral: %.2f +/- %.2f' %(a*s*(3.1416*2)**0.5, \
                            ((ae/a)**2+(se/s)**2)**0.5*a*s*(3.1416*2)**0.5)
                print '\tEW (obs) / Gauss fit: %.2f +/- %.2f' %(a*s*(3.1416*2)**0.5/con, \
                            ((ae/a)**2+(se/s)**2+(cone/con))**0.5*a*s*(3.1416*2)**0.5/con)
                print '\tEW (red) / Gauss fit: %.2f +/- %.2f\n' %(a*s*(3.1416*2)**0.5/con/(1+z), \
                            ((ae/a)**2+(se/s)**2+(cone/con))**0.5*a*s*(3.1416*2)**0.5/con/(1+z))
                                            
        absgrid = np.arange(xmin, xmax, (xmax-xmin)/600.)
        absso = params[0][0] + params[0][1]*mlab.normpdf(absgrid, mean, s)* \
                (2*3.1416)**0.5*s
        abssox = params[0][0] + params[0][1]*mlab.normpdf(x, mean, s)* \
                (2*3.1416)**0.5*s
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))

        # Gauss fit
#        ax.plot(absgrid, absso, c = 'b', lw=2)
        # Data
#            ax.errorbar(x[cpm : int(len(x)/2-xc1) ], 
#                        y[cpm : int(len(x)/2-xc1) ], 
#                        yerr = yerr[cpm : int(len(x)/2-xc1) ], 
#                        lw = 3, drawstyle = 'steps-mid', c = 'grey', capsize = 0)  
#            ax.errorbar(x[int(len(x)/2+xc1) : -cpp], 
#                        y[int(len(x)/2+xc1) : -cpp], 
#                        yerr = yerr[int(len(x)/2+xc1) : -cpp], 
#                        lw = 3, drawstyle = 'steps-mid', c = 'grey', capsize = 0)  
        ax.errorbar(x, y, yerr = yerr, drawstyle = 'steps-mid', lw=0.8,
                    c = 'black', capsize = 0)  

        if ign != []:
            #pass
            # Ignored Regions
            ax.errorbar(fitx_a[xig1:xig2], fity_a[xig1:xig2], yerr = fite_a[xig1:xig2], 
                        drawstyle = 'steps-mid', c = 'r', capsize = 0)
        # Residuals
        ax.errorbar(x, y - abssox, yerr = yerr, c = 'grey', lw = 0.8,
                     drawstyle = 'steps-mid', alpha = 0.5, capsize = 0)
        # Zero 
        ax.plot(x, 0*x, color = 'black', linestyle = '--')
        ymin, ymax = -3 * np.median(yerr), max(y) * 1.1
        if self.s.telcorr[arm] != '':
            ax.plot(x, self.s.telcorr[arm][x1:x2]*ymax*0.99,
                    '-', color = 'firebrick',  lw = 1.5)            
        elif self.s.skywl != []:
            ax.plot(skywlplot, self.s.skytrans*ymax*0.99, '-', color = 'firebrick', 
                    lw = 1.5)
        # Continuum
#            ax.plot(x, 0*x + cont, color = 'red', linestyle = '-')
        ax.set_xlim(xmin, xmax) #, yerr=self.onederro[arm])
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xl)
        if self.s.mult > 1E10:
            ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$' \
                        %(str(self.s.mult))[-2:])
        else:
            ax.set_ylabel(r'$\rm{Counts}$')
        fig.savefig('%s_%s.pdf' %(self.s.object, linename))        
        plt.close(fig)

################################################################################
    def plotTel(self, arm, fit = [], Aw = 5, cont = ''):
        
        wl = fit[0]/2. + fit[1]/2.
        dl = (self.wave[arm][-1]-self.wave[arm][0]) / (len(self.wave[arm]) - 1)
        x0 = self.s.wltopix(arm, wl)
        x1, x2 = int(x0 - Aw/dl), int(x0 + Aw/dl)
        fit1, fit2 = self.s.wltopix(arm, fit[0]), self.s.wltopix(arm, fit[1])
        moneddata = self.s.mult * self.oneddata[arm]
        x = self.wave[arm][x1:x2]
        y = moneddata[x1 : x2]
        yerr = moneddata[x1 : x2]
        
        intrwtel = 0.0/2.3548
        fitx = self.wave[arm][fit1:fit2]
        fity = moneddata[fit1 : fit2]
        fite = moneddata[fit1 : fit2]

        xl = r'$\rm{Observed\, wavelength\, (\AA)}$'
        xmin, xmax = wl - Aw, wl + Aw
        xmid, xsig = wl, (self.s.reso[arm]/c*wl/2.3548)
        fixed = 1
        if not cont:
            fixed, cont = 0, 2.6
        params = onedgaussfit(fitx, fity, err = fite, 
                params=[cont, -0.5, xmid, xsig],fixed=[fixed,0,0,0])   

        mean, meane = params[0][2], params[2][2]
        con, cone = params[0][0], params[2][0]
        a, ae = params[0][1], params[2][1]
        s, se = params[0][3], params[2][3]
        
        print '\t###########################'
        print '\tFitting line at %s' %wl
        print '\tContinuum %.2f +/- %.2f' %(con, cone)
        print '\tAmplitude %.2f +/- %.2f' %(a, ae)
        print '\tMean: %.2f +/- %.2f' %(mean, meane)

        print '\tSigma: %.2f +/- %.2f' %(s, se)
        print '\tFWHM: %.2f +/- %.2f' %(s*2.3548, se*2.3548)
        w = (s**2-intrwtel**2)**0.5*2.3548
        print '\tFWHM (Tel cor): %.2f +/- %.2f' %(w, se/s*w)
        print '\tResolution: %.2f +/- %.2f' %(mean/w, 
                          ((meane/mean)**2+(se/s)**2)**0.5 * mean/w)                        
                            
        absgrid = np.arange((xmin), (xmax), (xmax-xmin)/600.)
        absso = params[0][0] + params[0][1]*mlab.normpdf(absgrid, mean, s)* \
                (2*3.1416)**0.5*s
        abssox = params[0][0] + params[0][1]*mlab.normpdf(x, mean, s)* \
                (2*3.1416)**0.5*s
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1, 1, 1)        
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))

        ax.plot(absgrid, absso)
        ax.errorbar(x, y, yerr = yerr, drawstyle = 'steps-mid', c = 'black', capsize = 0)       
        ax.errorbar(fitx, fity, fite, drawstyle = 'steps-mid', c = 'r', capsize = 0)
        ax.errorbar(x, y - abssox, yerr = yerr, c = 'grey',
                     drawstyle = 'steps-mid', alpha = 0.5, capsize = 0)
        ax.plot(x, 0*x, c = 'black', ls = '--')
        
        ax.set_xlim((xmin), (xmax)) #, yerr=self.onederro[arm])
        ax.set_ylim(-3 * np.median(yerr), cont * 1.1)
        ax.set_xlabel(xl)
        ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
        fig.savefig('%s_tell_%s.pdf' %(self.s.object, wl)) 
        plt.close(fig)
        return mean/w  
        
################################################################################
    def plot1dall(self, arms= ['uvb','vis', 'nir'], 
                  ux1 = 3200, ux2 = 5600, um = 30,
                  uv1 = 5550, uv2 = 10060, vm = 20,
                  un1 = 10060, un2 = 23000, nm = 20,
                  lim = [-1.2, 5], logx = 0, logy = 0, unit = 'erg'):
        fig = plt.figure(figsize = (10,6))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        fig.subplots_adjust(bottom=0.15)
        if arms == '':
            arms ==  self.s.data.keys()

        wlmin, wlmax = 1E4, 1E3
        for arm in arms:#, 'nir']:
            if len(self.wave[arm]) > 0:
                if arm == 'uvb':
                    x1, x2, median, color = ux1, ux2, um, '0.00'
                if arm == 'vis':
                    x1, x2, median, color = uv1, uv2, vm, '0.35'
                if arm == 'nir':
                    x1, x2, median, color = un1, un2, nm, '0.7'
                x1 = self.s.wltopix(arm, x1)
                x2 = self.s.wltopix(arm, x2)
                splitData = self.oneddata[arm][x1:x2]
                splitErro = self.onederro[arm][x1:x2]      
                splitWave = self.wave[arm][x1:x2]
#                seltel = self.s.skytel[arm][x1:x2] > 0.95
#                selrad = self.s.skyrad[arm][x1:x2] < 3E3
#                sel = seltel * selrad

                wav, dat, ero = binspec(splitWave, splitData, 
                                        splitErro, wl = median)
                binsize = wav[1]-wav[0]
                if len(wav) < 50: lw = 2
                elif len(wav) < 250: lw = 1
                else: lw = 0.5
                if unit == 'Jy':
                    plotdat = ergJy(dat, wav)
                    if self.s.model[arm] != '':
                        ax.plot(self.wave[arm], ergJy(self.s.model[arm], self.wave[arm]))
                else:
                    plotdat = dat*self.s.mult
                    if self.s.model[arm] != '':
                        ax.plot(self.wave[arm], self.s.model[arm])

                ax.errorbar(wav, plotdat, ero*self.s.mult, binsize/2,
                            color = color, drawstyle = 'steps-mid',  lw = lw, capsize = 0)
                if min(wav)-binsize/2 < wlmin:
                    wlmin = min(wav)-binsize/2
                if max(wav)+binsize/2 > wlmax:
                    wlmax = max(wav)+binsize/2
#        GROND phot
#        photo = array([23.480, 22.430, 21.98, 21.61, 21.00, 20.80])
#        photoe = array([0.10, 0.08, 0.08, 0.07, 0.15, 0.17])
#        wl = array([4587, 6219, 7640, 8989, 12399, 16468])
#        ax.errorbar(wl, abflux(photo)/0.4, abflux(photo)*photoe/0.4, 
#                    fmt='o', ms = 8, capsize = 0)
        if unit == 'Jy':
            ax.set_ylabel(r'$F_{\lambda}\,\rm{(\mu Jy)}$')
        else:
            ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$' \
            %(str(self.s.mult))[-2:])
        ax.axhline(y=0, ls = '--', color = 'grey')
        ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
        if logx == 1:
            ax.set_xscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if logy == 1:
            ax.set_yscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_ylim(lim[0], lim[1])
        ax.set_xlim(wlmin, wlmax)#, yerr=self.onederro[arm])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
        fig.savefig('%s_1dall.pdf' %(self.s.object))        
        plt.close(fig)
       