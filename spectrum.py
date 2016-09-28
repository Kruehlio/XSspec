#!/usr/bin/env python
import sys
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pyfits
import numpy as np

from matplotlib import rc
from matplotlib.patheffects import withStroke
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate, constants

from .functions import (blur_image, smooth, dlafunc, ccmred,
                            redlaw)
from .fitter import (onedgaussfit, onedmoffatfit, onedmoffat,
                         plfit, pl, plfm, plfmfit)
from .astro import (airtovac, vactoair, LDMP,  absll, emll, abflux, 
                isnumber, getebv, binspec, errs, ergJy, Jyerg, highabswin)

c = constants.c/1E3
abslist, emllist = absll(), emll()
linelist = dict(emllist.items() + abslist.items())

myeffect = withStroke(foreground="w", linewidth=3)
kwargs = dict(path_effects=[myeffect])

# Theoretical sky models, RADIA is Skylines, TRANS is transmission, both at airmass
# 1.3 and for Paranal. See
# https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC

PAR_RADIA = os.path.join(os.path.dirname(__file__), "etc/paranal_radia_15_13.txt")
PAR_TRANS = os.path.join(os.path.dirname(__file__), "etc/paranal_trans_10_13_mod.txt")

class spectrum2d:
    """  Spectrum class for data manipulation and analysis
    
    Arguments:
        inst: Instrument that produced the spectrum (optional, default=xs)
    
    Methods:
        set1dFiles (Read in data file from 1d ascii spsectrum)
        setHead (Read in header from fits file)
        setReso (Set spectral resolving power R)
        binOneDSpec (Bin one-dimensional spectrum)
        binTwoDSpec (Bin two-dimensional spectrum)
        setFiles (Set input files - 2d fits files)
        fluxCor (Do a flux calibration)
        checkWL (Check wavelength solution via cross-correlation with skylines)
        smooth2d (Smooth the 2d spectrum)
        vacCor (Convert from air to vacuum wavelengths)
        helioCor (Convert from observed to heliocentric)
        ebvCal (Correct for Galactic E_B-V)
        scaleSpec (Derive scale factor from spectrum to photometry)
        applyScale (Apply scale factor to spectrum)
        setMod (Define physical afterglow model)
        setAscMod (Defines model from ascii file)
        scaleMod (Scale spectrum to afterglow model)
        makeProf (Create spatial profile in 2d-spectrum)
        extr1d (Extract 1d spectrum from 2d using the trace profile)
        write1d (Write out 1d spectrum into ascii file)
        smooth1d (Smooth the 1d spectrum)
        setCont (Set a simple continuum model)
        sn (Calculate signal to noise ratio depending on wavelength)
        wltopix (Convert wavelength to pixel)
        fitCont (Fit continuum with afterglow model)
        dlaabs (Add DLA absorber to continuum)
        writevp (writeout VPFit files)
        telcor (Create telluric correction using telluric star observations)
        appltel (Apply telluric correction from telluric star observations)
        stacklines (Under progress)
    """
    
    def __init__(self, inst = 'xs', tex =''):
        self.inst = inst
        self.datfiles = {'uvb': '', 'vis': '', 'nir': '', 'all': ''}
        self.redshift = 0
        self.output = {'all': 'ALL','uvb': 'UVB', 'vis': 'VIS', 'nir': 'NIR'}
        self.nh = ''
        self.object = ''
        self.mult = 1E17
        self.wlmult = 10
        self.dAxis = {'uvb': 1, 'vis': 1, 'nir': 1}
        self.tAxis = {'uvb': 1, 'vis': 1, 'nir': 1}
        self.ebv, self.rv, self.vhel = '', '',  0
        self.profile = {'uvb': 'moffat', 'vis':  'moffat', 'nir':  'moffat'}
        self.reso = {'uvb': c/5100., 'vis': c/8800., 'nir': c/5100.}
        # Intervening systems
        self.intsys = {}
        # Wavelength array
        self.wave = {'uvb': '', 'vis': '', 'nir': ''}
        # 2d data array
        self.data = {'uvb': '', 'vis': '', 'nir': ''}
        # 2d error array
        self.erro = {'uvb': '', 'vis': '', 'nir': ''}
        # 2d flag array
        self.flag = {'uvb': '', 'vis': '', 'nir': ''}
        # Fits header
        self.head = {'uvb': '', 'vis': '', 'nir': ''}
        # Optimal extraction profile
        self.prof = {'uvb': '', 'vis': '', 'nir': ''}
        # Trace parameters
        self.trace = {'uvb': '', 'vis': '', 'nir': ''}
        # Smoothed 2d data
        self.smooth = {'uvb': '', 'vis': '', 'nir': ''}
        # Data range in 2d fits frame, this is pixel
        self.datarange = {'uvb': [], 'vis': [], 'nir': []}
        # Background pixel in datarange
        self.backrange = {'uvb': [5, 5], 'vis': [5, 5], 'nir': [4, 4]}
        # WL range for different arms
        self.wlrange = {'uvb': [3000., 5850.], 'vis': [5500, 10200], 
                        'nir': [10000, 24500], 'all': [3000, 24500]}
        # 1d data after optimal extraction
        self.oneddata = {'uvb': '', 'vis': '', 'nir': ''}
        self.onedback = {'uvb': '', 'vis': '', 'nir': ''}
        # Smoothed 1d data after optimal extraction
        self.soneddata = {'uvb': '', 'vis': '', 'nir': ''}
        # 1d error after optimal extraction
        self.onederro = {'uvb': '', 'vis': '', 'nir': ''}
        # 1d SKY rms
        self.skyrms = {'uvb': '', 'vis': '', 'nir': ''}
        # 1d afterglow model based on supplied beta, AV and z
        self.model = {'uvb': '', 'vis': '', 'nir': ''}
        # Matching factor to Afterglow molde
        self.match = {'uvb': '', 'vis': '', 'nir': ''}
        # Correction factor for Gal. E_B-V
        self.ebvcorr = {'uvb': '', 'vis': '', 'nir': ''}
        # 1d continuum (without absorption lines)
        self.cont = {'uvb': '', 'vis': '', 'nir': ''}
        self.woabs = {'uvb': '', 'vis': '', 'nir': ''}
        # 1d data including DLA absorption
        self.oneddla = {'uvb': '', 'vis': '', 'nir': ''}
        # Correction factor to photometry
        self.slitcorr = {'all': '', 'uvb': '', 'vis': '', 'nir': ''}
        # Telluric correction
        self.telcorr = {'uvb': '', 'vis': '', 'nir': ''}
        self.telwave = {'uvb': '', 'vis': '', 'nir': ''}
        # Cleaned means no tellurics, no absorption lines
        self.cleanwav = {'uvb': '', 'vis': '', 'nir': ''}
        self.cleandat = {'uvb': '', 'vis': '', 'nir': ''}
        self.cleanerr = {'uvb': '', 'vis': '', 'nir': ''}
        self.lineflux = {}
        self.skytel = {}
        self.skyrad = {}
        self.linepars = {}
        # Luminosity spectrum
        self.lumspec = {'uvb': '', 'vis': '', 'nir': ''}
        self.lumerr = {'uvb': '', 'vis': '', 'nir': ''}
        self.restwave = {'uvb': '', 'vis': '', 'nir': ''}
        print '\n\t######################'
        print '\tSpectrum class'
        self.setSkySpec()
        #self.setSkyEml()
        print '\t######################'
        if tex in ['yes', True, 1, 'Y', 'y']:
            rc('text', usetex=True)

################################################################################

    def setSkySpec(self):
        skywl, skytrans, skyradia = [], [], []
        filen = os.path.expanduser(PAR_TRANS)
        fin = open(filen, 'r')
        lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
        fin.close() 
        for line in lines:
            if line != [] and isnumber(line[0]):
                if float(line[0]) > 0.2999 and float(line[0]) < 2.5:
                    skywl.append(float(line[0])*1E4)
                    skytrans.append(float(line[1]))
        self.skywl, self.skytrans = (np.array(skywl)), np.array(skytrans)
        self.skywlair = vactoair(self.skywl)
        
        filen = os.path.expanduser(PAR_RADIA)
        fin = open(filen, 'r')
        lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
        fin.close() 
        for line in lines:
            if line != [] and isnumber(line[0]):
                if float(line[0]) > 0.2999 and float(line[0]) < 2.5:
                    skyradia.append(float(line[1]))
        self.skyradia = np.array(skyradia)
        print '\tTheoretical Sky Spectrum set'

################################################################################

    def set1dFiles(self, arm, filen, mult = 10, errsc = 1., mode = 'txt'):
        if self.datfiles.has_key(arm) and mode == 'txt':
            print '\t1d-data file %s as arm %s added' %(filen, arm)
            lines = [line.strip() for line in open(filen)]
            wave, data, erro = np.array([]), np.array([]), np.array([])
            for line in lines:
                if line != [] and line.split() != [] and isnumber(line.split()[0]):
                    wave = np.append(wave, float(line.split()[0]))
                    data = np.append(data, float(line.split()[1]))
                    erro = np.append(erro, float(line.split()[2]))
            self.wave[arm] = wave
            self.oneddata[arm] = data
            self.onederro[arm] = erro*errsc
            self.skyrms[arm] = erro*errsc
       
        elif self.datfiles.has_key(arm) and mode == 'fits':
            print '\t1d-data fits file %s as arm %s added' %(filen, arm)
            hdulist = pyfits.open(filen)
            self.oneddata[arm] = hdulist[0].data
            erro = abs(hdulist[1].data)
            erro[erro < 1e-22] = 1e-22
            self.onederro[arm] = erro*errsc
            self.skyrms[arm] = erro*errsc
            self.head[arm] = hdulist[0].header
            pix = np.arange((self.head[arm]['NAXIS1'])) + self.head[arm]['CRPIX1']
            self.wave[arm] = (self.head[arm]['CRVAL1'] + (pix-1)*self.head[arm]['CDELT1'])*mult
            hdulist.close()
        else:
            print 'Arm %s not known' %arm
        tck1 = interpolate.InterpolatedUnivariateSpline(self.skywlair, self.skytrans)
        tck2 = interpolate.InterpolatedUnivariateSpline(self.skywlair, self.skyradia)
        self.skytel[arm] = tck1(self.wave[arm])
        self.skyrad[arm] = tck2(self.wave[arm])

################################################################################  
  
    def setHead(self, arm, filen):
        if self.datfiles.has_key(arm):
            hdulist = pyfits.open(filen)
            self.head[arm] = hdulist[0].header
            hdulist.close()
            
################################################################################    

    def setReso(self, arm, reso):
        print 'Resolving power in arm %s is R = %.0f' %(arm, reso)
        print '\tResolution in arm %s set to %.2f km/s' %(arm, c/reso)
        self.reso[arm] = c/reso
    
    def setWLRange(self, arm, wlrange):
        self.wlrange[arm] = wlrange

    def setBackRange(self, arm, back):
        self.backrange[arm] = back

    def setDatarange(self, arm, y1, y2):
        print '\tUsing datarange from y1 = %i to y2 = %i pixels' %(y1, y2)
        self.datarange[arm] = [y1, y2]

    def setObject(self, objectn):
        print '\tNew object added: %s' %objectn
        self.object = objectn

    def setVhelio(self, vhel):
        self.vhel = vhel

    def intSys(self, red, lines):
        self.intsys[red] = lines

    def showData(self, arm):
        print self.data[arm]

    def show1dData(self, arm):
        print self.oneddata[arm]

    def showWave(self, arm):
        print self.wave[arm]

    def showOneDData(self, arm, x = 14000):
        print self.oneddata[arm][x], self.onederro[arm][x]

    def showErro(self, arm):
        print self.erro[arm] 

    def showHead(self, arm):
        print self.head[arm] 

    def setRed(self, z):
        print '\tRedshift set to %s' %z
        self.redshift = z    

    def setOut(self, out):
        print '\tOutput set to %s' %out
        self.output = out   

    def setMult(self, mult):
        self.mult = mult 

    def setInt(self, z, lines):
        print '\tRedshift for intervening system set to %s' %z
        self.intsys[z] = lines    

################################################################################

    def binOneDSpec(self, arm, binr = 40, meth = 'average', clip = 3, do_weight = 1):
        print '\tBinning 1d spectrum by factor %i' %binr 
        if do_weight not in [0, 'False', False, 'N', 'no', 'n', 'No']:
            print '\tUsing error-weighted average in binned spectrum'
        if self.data[arm] == '':
            print '\tNeed an extracted 1d spectrum first'
            print '\tWill not bin / doing nothing'
        else:
            self.wave[arm+'o'], self.oneddata[arm+'o'], self.onederro[arm+'o'] = \
                self.wave[arm], self.oneddata[arm], self.onederro[arm] 
            self.wave[arm], self.oneddata[arm], self.onederro[arm] = \
                binspec(self.wave[arm], self.oneddata[arm], self.onederro[arm],
                        wl = binr, meth = meth, clip = clip, do_weight = do_weight)
                        
################################################################################
                        
    def binTwoDSpec(self, arm, binr = 40, meth = 'average', clip = 3, do_weight = 1):
        shapebin = (len(self.data[arm][0]), len(self.data[arm])/binr)
        bindata, binerro = np.zeros(shapebin), np.zeros(shapebin)
        print '\tBinning 2d spectrum by factor %i' %binr
        if self.data[arm] == '':
            print '\tNeed to provide the 2d spectrum first'
            print '\tWill not bin / doing nothing'
        else:
            for i in range(len(self.data[arm][0])):
                dataline = self.data[arm].transpose()[i]
                erroline = self.erro[arm].transpose()[i]
                binwave, binlinedata, binlineerro = \
                    binspec(self.wave[arm], dataline, erroline,
                        wl = binr, meth = meth, clip = clip, do_weight = do_weight)
                bindata[i] = binlinedata
                binerro[i] = binlineerro
            self.data[arm] = bindata.transpose()
            self.wave[arm] = binwave
            self.erro[arm] = binerro.transpose()
            
################################################################################  
            
    def setFiles(self, arm, filen, filesig = '', dAxis = 1, 
                 mult = 10, const = 0, fluxmult=1):
        '''Input Files uses by default columns dispersion axis (keyword dAxis),
        and assumes nm as wavelength unit (multiplies by 10, keyword mult).
        Uses header keywords NAXIS, CRVAL, CDELT, CRPIX, and by default a MEF
        file with the first extension data, second error. Error spectrum can 
        also be given as wih keyword filesig. If both are absent, we shall 
        assume sqrt(data) as error. No bad pixel masking is implemented yet.
        Replaces errors 0 with 1E-31. Write the fits header, data, error an
        wavelenths into class attributes  head[arm], data[arm], erro[arm], 
        and wave [arm]'''
        self.dAxis[arm] = dAxis
        if self.dAxis[arm] == 1:
            self.tAxis[arm], tAxis = 2, 2
        elif self.dAxis[arm] == 2:
            self.tAxis[arm], tAxis = 1, 1

        self.wlmult = mult
        if self.datfiles.has_key(arm):
            self.head[arm] = pyfits.getheader(filen, 0)
            skl =  self.head[arm]['NAXIS%i'%tAxis]
            skinc =  self.head[arm]['CDELT%i'%tAxis]
            if self.datarange[arm] == []:
                if arm in ['uvb', 'vis']: dy = 3.5/skinc
                elif arm in ['nir']: dy = 3.2/skinc
                self.datarange[arm] = [max(7,int(skl/2-dy)), 
                                       min(skl-7, int(skl/2+dy))] 
                print '\t%s datarange %i to %i pixels' \
                    %(arm.upper(), self.datarange[arm][0], self.datarange[arm][1])
            if self.object == '':
                self.output[arm] = os.path.splitext(filen)[0]
            else:
                self.output[arm] = self.object+'_%s' %arm
            self.datfiles[arm] = filen

            yminpix, ymaxpix = self.datarange[arm][0] - 1, self.datarange[arm][1] - 1
            print '\tData file %s as arm %s added' %(filen, arm)
            wlkey, wlstart = 'NAXIS%i'%dAxis, 'CRVAL%i'%dAxis
            wlinc, wlpixst = 'CDELT%i'%dAxis, 'CRPIX%i'%dAxis
            if dAxis == 1:
                y = pyfits.getdata(filen, 0)[yminpix : ymaxpix].transpose()
            else:
                ytmp = pyfits.getdata(filen, 0).transpose()[yminpix : ymaxpix]
                y = ytmp.transpose()
            y[abs(y) < 1E-4/self.mult] = 1E-4/self.mult
            self.data[arm] = y
            if filesig != '':
                if dAxis == 1:
                    yerr = pyfits.getdata(filesig, 0)[yminpix : ymaxpix].transpose()
                else:
                    yerrtmp = pyfits.getdata(filesig, 0).transpose()[yminpix : ymaxpix]
                    yerr = yerrtmp.transpose()
                if len(np.shape(yerr)) == 1:
                    print '\t\t1d error spectrum'
                    pass
            else:
                try:
                    yerr = pyfits.getdata(filen, 1)[yminpix : ymaxpix].transpose()
                    print '\t\tError extension found'
                except IndexError:
                    ytmp = []
                    print '\t\tCan not find error spectrum -> Using std(data)' 
                    for i in range(len(y)):
                        ytmp.append(np.std(y[i][yminpix:ymaxpix]))
                    print y
                    yerr = ((0*np.abs(y)**0.5).transpose() + np.array(ytmp)).transpose()
            try:
                yflag = pyfits.getdata(filen, 2)[yminpix : ymaxpix].transpose()
                print '\t\tFlag extension found'
            except IndexError:
                yflag = yerr*0
            self.flag[arm] = np.array(yflag)
                
            yerr[abs(yerr) < 1E-5/self.mult] = 1E-5/self.mult
            self.erro[arm] = yerr
            pix = np.arange(self.head[arm][wlkey]) + self.head[arm][wlpixst]
            self.wave[arm] = (self.head[arm][wlstart] + (pix-1)*self.head[arm][wlinc])*mult
            wlsel = (self.wlrange[arm][0] < self.wave[arm]) * (self.wave[arm] <self.wlrange[arm][1]) 
            self.wave[arm] = np.array(self.wave[arm][wlsel])
            self.erro[arm] = np.array(self.erro[arm][wlsel])
            self.data[arm] = np.array(self.data[arm][wlsel]*fluxmult) + const
            self.flag[arm] = np.array(self.flag[arm][wlsel])
            #print '\t\tSky spectrum to spectrum grid'
            tck1 = interpolate.InterpolatedUnivariateSpline(self.skywlair, self.skytrans)
            tck2 = interpolate.InterpolatedUnivariateSpline(self.skywlair, self.skyradia)
            self.skytel[arm] = tck1(self.wave[arm])
            self.skyrad[arm] = tck2(self.wave[arm])
        else:
            print 'Arm %s not known' %arm
            print 'Known arms: uvb, vis, nir, all'
            
################################################################################   
            
    def fluxCor(self, arm, fluxf, countf):
        print '\tConverting counts to flux'
        fluxspec = np.array(pyfits.getdata(fluxf, 0)[0].transpose())
        countspec = np.array(pyfits.getdata(countf, 0)[0].transpose())
        wlkey, wlstart = 'NAXIS%i'%self.dAxis[arm], 'CRVAL%i'%self.dAxis[arm]
        wlinc, wlpixst = 'CDELT%i'%self.dAxis[arm], 'CRPIX%i'%self.dAxis[arm]
        head = pyfits.getheader(fluxf, 0)
        pix = np.arange(head[wlkey]) + head[wlpixst]
        wave = (head[wlstart] + (pix-1)*head[wlinc])*self.wlmult
        count = np.where(countspec == 0, 1E-5, countspec)
        resp = fluxspec/count
        tck = interpolate.InterpolatedUnivariateSpline(wave, resp)
        respm = tck(self.wave[arm])
        fig = plt.figure(figsize = (9,9))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        
        ax.set_yscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.plot(self.wave[arm], respm, 'o')
        ax.set_xlabel(r'$\rm{Observed\,wavelength\,(\AA)}$')
        ax.set_ylabel(r'$\rm{Response\,function}$')
        fig.savefig('%s_resp_%s.pdf' %(self.object, arm))
        self.data[arm] = (respm*self.data[arm].transpose()).transpose()
        self.erro[arm] = (respm*self.erro[arm].transpose()).transpose()  
        
################################################################################
        
    def checkWL(self, arms, intensity = 2, chop = 10, mod = 1):
        print '\tChecking accuracy of wavelength solution against skylines'
        for arm in arms:
            if arm == 'uvb':
                chopap, order = min(chop, 1), 0
                xmin = self.wltopix(arm, 5450)
                xmax, err = -200, 0.03
            elif arm == 'vis':
                xmin = self.wltopix(arm, 6000)
                xmax = self.wltopix(arm, 9800)
                chopap, order, err = chop, 1, 0.05
            elif arm == 'nir':
                xmin = self.wltopix(arm, 10200)
                xmax = self.wltopix(arm, 21000)
                chopap, order, err = chop, 1, 0.02
            waveccs = self.wave[arm][xmin:xmax]
            skyrmss = self.skyrms[arm][xmin:xmax]
            skyrads = self.skyrad[arm][xmin:xmax]

            dl = (waveccs[-1]-waveccs[0])/len(waveccs)   
            skyrad = np.array_split(skyrads, chopap)
            skyrms = np.array_split(skyrmss, chopap)
            wavecc = np.array_split(waveccs, chopap)
            wloff, wls, wlerrs, fwhms, offset, sigma = [], [], [], [], 0, 2
            pp = PdfPages('%s_wlacc_%s.pdf' %(self.object, arm))
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
                params = onedgaussfit(xcor, corrc, err = corrc*0+err,
                                      params=[0, 1, offset, sigma],
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
                self.wave[arm] -= np.polyval(a, self.wave[arm])
                print '\t\tWavelength scale modified'
                self.restwave[arm] = self.wave[arm]/(1+self.redshift)
            else:
                print '\t\tWavelength scale not modified'

            fig1 = plt.figure(figsize = (11,6))
            ax1 = fig1.add_subplot(1, 2, 1)
            ax1.errorbar(np.array(wls)/1E4, wloff, yerr = wlerrs, capsize = 0,
                        fmt='o', color ='black')
            ax1.plot(self.wave[arm]/1E4, np.polyval(a, self.wave[arm]), color ='black')
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
            
#################################################   
            
    def smooth2d(self, arm, smoothx, smoothy = 3):  
        if len(self.data[arm]) != 0:
            self.smooth[arm] = blur_image(self.data[arm], smoothx, smoothy)
        else:
            print '\t2d data not available'
            
################################################################################   
            
    def vacCor(self, arms):
        ''' Convert wavelength scale from air (default all ESO instrument incl. 
        X-shooter) to vacuum, using spec.astro.airtovac '''
        print '\tConverting air to vacuum wavelengths'
        for arm in arms:
            self.wave[arm] = airtovac(self.wave[arm])
            self.restwave[arm] = self.wave[arm]/(1+self.redshift)
            self.output[arm] = self.output[arm]+'_vac'

################################################################################ 
   
    def helioCor(self, arms, vhel = 0):
        ''' Corrects the wavelength scale based on a given helio-centric *correction*
        value. This is not the Earth's heliocentric velocity. Using ESO keyword 
        HIERARCH ESO QC VRAD HELICOR. Alternatively, get the correction value via 
        IRAF rvcorrect'''
        if self.vhel != 0:
            vhel = self.vhel
        if vhel == 0:
            # ESO Header gives the heliocentric radial velocity correction
            vhel = self.head[arms[0]]['HIERARCH ESO QC VRAD HELICOR']
        self.vhel = vhel
        print '\tHeliocentric velocity correction: %.2f km/s:' %vhel
        # Relativistic version of 1 + vhelcor/c     
        lamscale =  ((1 + vhel/c) / (1 - vhel/c) )**0.5
        print '\tScaling wavelength by: %.6f' %(lamscale)
        for arm in arms:
            self.wave[arm] *= lamscale     
            self.restwave[arm] = self.wave[arm]/(1+self.redshift)
            self.output[arm] = self.output[arm]+'_helio'
        self.skywlair *= lamscale
        self.skywl *= lamscale

################################################################################   
     
    def ebvCal(self, arms, ebv = '', rv = 3.08):
        if ebv == '':
            ra, dec = self.head[arms[0]]['RA'], self.head[arms[0]]['DEC']
            ebv, std, ref, av = getebv(ra, dec, rv)
            if ebv != '':
                print '\t\tQueried E_B-V %.3f' %ebv
            else:
                ebv = 0.
        self.ebv = ebv
        self.rv = rv
        for arm in arms:
            self.ebvcorr[arm] = ccmred(self.wave[arm], ebv, rv)
            
################################################################################
            
    def scaleSpec(self, arm, pband = '', mag = '', err = 1.E-5,
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
            wlsel = (bandwl[pband][0] < self.wave[arm]) * (self.wave[arm] < bandwl[pband][1]) 
        else:
            wlsel = (pband[0] < self.wave[arm]) * (self.wave[arm] < pband[1]) 

        print '\tEstimating flux in band %s' %pband
        # First, we select the requested wavelengths for photband
        pb, pbw, pbe = self.oneddata[arm][wlsel], self.wave[arm][wlsel], self.onederro[arm][wlsel]
        bb, bbe = self.onedback[arm][wlsel], self.skyrms[arm][wlsel]

        # Second, only where telluric absorption is not strong, transmission >0.85
        # And exclude regions of high sky radiance       
        if arm in ['vis', 'nir']:        
            skyexl = self.skytel[arm][wlsel] > 0.85
            skyexl *= self.skyrad[arm][wlsel] < 5E2
        else:
            skyexl = self.skytel[arm][wlsel] > 0.

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
        elif self.modfl != '':
            wlsel2 = (bandwl[pband][0] < self.modwl) * (self.modwl < bandwl[pband][1]) 
            bandflux = np.array(self.modfl[wlsel2])
            fluxcomp = np.median(bandflux) * np.ones(1000)
        else:
            print ('\tNeed either a magnitude via mag or a defined ASCII Model')
            sys.exit()
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
        self.slitcorr[arm] = corrf
        self.linepars['sl_%s_%s'%(arm, pband)] = (arm.upper(), pband, corrf, corrfmax)

################################################################################

    def applyScale(self, arms, mdata = 0, usearm = ''):
        ''' Applies the previously defined scaling to the luminosity spectrum
        Uses the same arm, or if not defined the factor for the reddest arm that
        is defined. Can be overridden by usearm parameter. E.g., usearm=vis uses
        the vis factor for all bands in arms'''
        corfarm = {}
        corfa, corfea = [], []
        for arm in arms:
            corfs, corfes = [], []
            for slf in self.linepars.keys():
              if slf.startswith('sl'):  
                if self.linepars[slf][0] == arm.upper():
                    corfs.append(self.linepars[slf][2])
                    corfes.append(self.linepars[slf][3])
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
            self.lumspec[arm] *= corf
            self.lumerr[arm] *= corf
            if mdata != 0:
                print '\t\tMultiplying %s data with %.2f' %(arm, corf)
                self.oneddata[arm] *= corf
                self.onedback[arm] *= corf
                self.onederro[arm] *= corf
                
################################################################################
                
    def setMod(self, arms, norm, beta = 0.5, av = 0.15, red = 'smc'):
        ''' Defines a physical afterglow model given with a given norm at 6250 AA 
        in muJy (unreddened, A_V = 0), beta, av and reddening law (default smc)'''
        for arm in arms:
            wls = self.wave[arm]
            law = redlaw(wls/(1+self.redshift), red)
            modmuJy = norm * (wls/6250.)**(beta) * 10**(-0.4*av*law)
            self.model[arm] = Jyerg(modmuJy, wls)
            
################################################################################
            
    def setAscMod(self, arms, mfile, s = 3.0, order = 2):
        ''' Read in ASCII model (Lephare host output, which means two columns with
        WL (\AA) and AB mag) '''
        lines = [line.strip() for line in open(mfile)]
        self.modwl, self.modfl = [], []
        for line in lines:
            if line != '':
                if len(line.split()) == 2 \
                and isnumber(line.split()[0]) and isnumber(line.split()[1]):
                    self.modwl.append(float(line.split()[0]))
                    self.modfl.append(abflux(float(line.split()[1])))
        self.modwl, self.modfl = np.array(self.modwl), np.array(self.modfl)
        modergs = Jyerg(self.modfl, self.modwl)
        tck = interpolate.UnivariateSpline(self.modwl, modergs, s = s, k = order)
        for arm in arms:
            self.model[arm] = tck(self.wave[arm])
        print ('\t ASCII model set sucessfully')
        
################################################################################ 
        
    def scaleMod(self, arms, p = ''):
        for arm in arms:
            self.match[arm] = self.model[arm]/self.cont[arm]
            print '\tScaling %s spectrum to afterglow model' %arm
            if len(self.data[arm]) != 0:
                atmp = self.data[arm].transpose() * self.match[arm]
                btmp = self.erro[arm].transpose() * self.match[arm]
                self.data[arm] = atmp.transpose()
                self.erro[arm] = btmp.transpose()
            if len(self.oneddata[arm]) != 0:    
                self.oneddata[arm] *= self.match[arm] 
                self.onederro[arm] *= self.match[arm] 
                self.cont[arm] = np.array(self.model[arm])
            self.output[arm] = self.output[arm]+'_scale'
            if p != '':
                fig = plt.figure(figsize = (10,5))
                ax = fig.add_subplot(1, 1, 1)
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
                ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
                ax.plot(self.wave[arm], self.match[arm])
                ax.set_xlim(self.wlrange[arm][0], self.wlrange[arm][1])        
                ax.set_ylim(0.8, 4)        
                fig.savefig('%s_slitloss_%s.pdf' %(self.object, arm))
                
################################################################################ 
                
    def makeProf(self, arm, lim1 = 3.4E3, lim2 = 25E3, chop = 1, order = 0,
                 fwhm = '', mid = '', vac = 0, verbose = 0, orderfwhm = None,
                 line = '', meth = 'weight', profile = None):
        
        if profile == None:
            profile = self.profile[arm]
        else:
            self.profile[arm] = profile
            
        if orderfwhm == None:
            orderfwhm = max(0, order-1)


        print '\tCreating spatial profile for arm %s, %s profile' %(arm, profile)
        tracemid, tracesig, tracemide, tracesige, tracepic, tracewl = 6 * [np.array([])]
        tracebet, tracebete = [], []
        Pics = np.arange(len(self.wave[arm]))
        if line != '':
            vac = 1
            if line.upper() == 'OII':
                lim1 = emllist['[OII](3728)'][0]*(1+self.redshift)-8
                lim2 = emllist['[OII](3728)'][0]*(1+self.redshift)+8
            if line.upper()  == 'OIII':
                lim1 = emllist['[OIII](5007)'][0]*(1+self.redshift)-5
                lim2 = emllist['[OIII](5007)'][0]*(1+self.redshift)+5
            elif line in ('Ha', 'Halpha', 'ha'):
                lim1 = emllist['Halpha'][0]*(1+self.redshift)-5
                lim2 = emllist['Halpha'][0]*(1+self.redshift)+5
            elif line in ('Hb', 'Hbeta', 'hb'):
                lim1 = emllist['Hbeta'][0]*(1+self.redshift)-5
                lim2 = emllist['Hbeta'][0]*(1+self.redshift)+5
        
        if vac == 0:
            print '\t\tConverting limits to vacuum'
            lim1 = airtovac(lim1)
            lim2 = airtovac(lim2)
            
        wlsel = (lim1 < self.wave[arm]) * (self.wave[arm] < lim2) 
        sData = np.array_split(self.data[arm][wlsel], chop)
        sWave = np.array_split(self.wave[arm][wlsel], chop)
        sErro = np.array_split(self.erro[arm][wlsel], chop)
        sPics = np.array_split(Pics[wlsel], chop)
        pp = PdfPages('%s_profiles_%s.pdf' %(self.object, arm))

        for i in range(len(sData)):
            tellregs = [[7580, 7660], [9300, 9550],
                        [11000, 11500], [13000, 15000], [17600, 19600]]
            spic, swave = np.average(sPics[i]), np.average(sWave[i])
            inTel = 0
            for tellreg in tellregs:
                if tellreg[0] < swave < tellreg[1]:
                    inTel = 1
            if len(sData) in range(5):
                inTel = 0
            if inTel == 0:
                uErro = np.array(sErro[i]*self.mult).transpose()
                uData = np.array(sData[i]*self.mult).transpose()
                sprof, sprofe = [], []           
                
                for line, linee in zip(uData, uErro):
                    while True:
                        stdline, stdmed = np.median(linee), np.median(line)
                        if np.std(line) > 3*stdline:
                            stdline = np.std(line)
                        sel = (line < stdmed + 10*stdline) * (line > stdmed - 10*stdline)
                        if sel.all() == True or len(line) < 10:
                            break
                        line = np.array(line[sel])
                        linee = np.array(linee[sel])
                    
                    if meth == 'median':
                        sprof.append(  np.median(line) )
                        sprofe.append( np.std(line) / len(line)**0.5 )
                    
                    elif meth == 'weight':
                        mult =  1./min(linee[linee>0])
                        weight = (1. / (mult*linee)**2 )
                        sprof.append( sum (line * weight) / sum(weight) )
                        sprofe.append((sum(linee**2*weight)/ sum(weight)**2)**0.5)
     
                sprof, sprofe = np.array(sprof)/max(sprof), np.array(sprofe)
                sprof[:self.backrange[arm][0]] = 0
                sprof[-self.backrange[arm][1]:] = 0
    
                X = np.arange(len(sprof))
                if mid != '': mean, fm = mid, 1
                else: mean, fm = np.argmax(sprof), 0
                
                if fwhm != '':  sig, fw = fwhm/2.3548, 1
                else: sig, fw = 3, 0
                
                if profile == 'moffat':
                    params = onedmoffatfit(X, sprof, err = sprofe,
                                  params=[0, 1, mean, sig, 3.5],
                                  fixed=[1, 0, fm, fw, 1],
                                  minpars=[0, 0, 0, 0, 0],
                                  limitedmin=[0, 1, 1, 1, 1])    
                                  
                else:
                    params = onedgaussfit(X, sprof, err = sprofe,
                                  params=[0, 1, mean, sig],
                                  fixed=[1, 0, fm, fw],
                                  minpars=[0, 0, 0, 0],
                                  limitedmin=[0, 1, 1, 1])
                
                if verbose != 0:
                    print '\tTrace at x = %.i px' %spic
                    print '\t\tMean of trace y = %.2f +/- %.2f px' \
                                %(params[0][2] + self.datarange[arm][0], params[2][2])
                    if profile == 'moffat':
                        print '\t\tFWHM of trace y = %.2f px' \
                            %(params[0][3] * 2 * (2**(1./params[0][4]) - 1 )**0.5)
                    else:    
                        print '\t\tSigma of trace y = %.2f +/- %.2f px' %(params[0][3], params[2][3])
                    
                fig = plt.figure(figsize = (9,6))
                ax = fig.add_subplot(1, 1, 1)
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
                ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))                
                
                ax.errorbar(X, sprof, yerr = sprofe, fmt = 'o')
#                ax.plot(X, smooth(sprof, 7), '-', lw=2)
                ax.plot(params[-1], params[1])
                ax.set_xlabel(r'$\rm{Pixel}$')
                ax.set_ylabel(r'$\rm{Normalized\, flux}$')
                pp.savefig(fig) 
                plt.close(fig)

                # Use only certain wl in trace ftting     
                if lim1 < swave < lim2 and params[0][3] > 1.:
                    if mid == '':
                        tracemid = np.append(tracemid, params[0][2])
                        tracemide  = np.append(tracemide, max(params[2][2], params[0][2]/100.))
                    else:
                        tracemid = np.append(tracemid, mean)
                        tracemide = np.append(tracemide, mean/100.)
                    
                    if fwhm == '':
                        tracesig = np.append(tracesig, params[0][3])
                        tracesige = np.append(tracesige, max(params[2][3], params[0][3]/40.))
                    else:
                        tracesig = np.append(tracesig, sig)
                        tracesige= np.append(tracesige, sig/100.)
                    
                    if profile == 'moffat':
                        tracebet = np.append(tracebet, params[0][4])
                        tracebete = np.append(tracebete, max(params[2][4], params[0][4]/40.))
                    tracepic = np.append(tracepic, spic)
                    tracewl =  np.append(tracewl, swave)
        
        a = np.polyfit(tracepic, tracemid, order, w = 1./tracemide**2)
        b = np.polyfit(tracepic, tracesig, orderfwhm, w = 1./tracesige**2)
        if profile == 'moffat':
            c = np.polyfit(tracepic, tracebet, orderfwhm, w = 1./tracebete**2)
            self.trace[arm] = [a, b, c]
            nplot, figsize = 2, (9, 6)
        else:
            self.trace[arm] = [a, b]
            nplot, figsize = 2, (9, 6)
            
        fig = plt.figure(figsize = figsize)
        fig.subplots_adjust(hspace=0.05, wspace=0.0)
        fig.subplots_adjust(bottom=0.12, top=0.98, left=0.14, right=0.89)
        ax1 = fig.add_subplot(nplot, 1, 1)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax1.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))        
        
        ax1.xaxis.tick_top()
        ax1.errorbar(tracewl, tracemid + self.datarange[arm][0],
                     yerr= tracemide, fmt = 'o', capsize = 0)
        ax1.plot(self.wave[arm], 
                 np.polyval(self.trace[arm][0], Pics) + self.datarange[arm][0] )
        
        ax1.set_ylabel(r'$\rm{Mean\,of\,trace\,(px)}$')
        ax1.set_xlabel('')
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        ax2 = fig.add_subplot(nplot, 1, 2)
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))        
        ax3=ax2.figure.add_axes(ax2.get_position(), frameon = False)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        
        if profile == 'moffat':
            fwhm = tracesig * 2 * (2**(1./tracebet) - 1) **0.5
            fwhmerr = tracesige * 2 * (2**(1./tracebet) - 1) **0.5
            fitfwhm = np.polyval(self.trace[arm][1], Pics) * \
            2 * (2**(1./np.polyval(self.trace[arm][2], Pics)) - 1 )**0.5
            
#            ax4 = fig.add_subplot(nplot, 1, 3)
#            ax4.plot(self.wave[arm], polyval(self.trace[arm][1], Pics) )
#            ax4.errorbar(tracewl, tracesig, yerr= tracesige, fmt = 'o', capsize = 0)
#            ax5 = fig.add_subplot(nplot, 1, 4)
#            ax5.plot(self.wave[arm], polyval(self.trace[arm][2], Pics) )
#            ax5.errorbar(tracewl, tracebet, yerr= tracebete, fmt = 'o', capsize = 0)

#            plt.setp(ax1.get_xticklabels(), visible=False)
#            plt.setp(ax2.get_xticklabels(), visible=False)
#            plt.setp(ax3.get_xticklabels(), visible=False)

        else:
            fwhm = tracesig * 2.3538
            fwhmerr = tracesige * 2.3538
            fitfwhm = np.polyval(self.trace[arm][1], Pics) * 2.3538

        ax2.errorbar(tracewl, fwhm, yerr = fwhmerr, fmt = 'o', capsize = 0)
        ax2.errorbar(self.wave[arm], fitfwhm)
        
        ax2.set_ylabel(r'$\rm{FWHM\,of\,trace\,(px)}$')
        ax2.set_xlabel(r'$\rm{Observed\, wavelength\, (\AA)}$')
        pxsc = float(self.head[arm]['CD2_2'])
        ax3.set_ylim([ax2.get_ylim()[0]*pxsc, 
                      ax2.get_ylim()[1]*pxsc])
        ax3.set_ylabel(r'$\rm{FWHM\,(arcsec)}$')
        ax3.yaxis.tick_right()
        ax2.yaxis.tick_left()
        plt.setp(ax3.get_xticklabels(), visible=False)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim([min(self.wave[arm]), max(self.wave[arm])])
        pp.savefig(fig)
        pp.close()
    
################################################################################
    
    def extr1d(self, arms, opt = 1, n = 4, sig = '', errsc = 1., offset = 0):
        """Parameter opt = 1, use the derived profile to do opimal extraction
                     opt = 0, use an aperture of size n pixels
          Parameter n: everything out of +/- n/2.3548 arcsec is not extracted 
              for optimal size of the aperture
          Parameter sig: fix sigma (in pixels) of the profile"""
          
        for arm in arms:
            pxscale =  min(0.25, self.head[arm]['CDELT2'])
            weightdata, weighterro, skyrms, skysum, prof = [], [], [], [], []
            if self.trace[arm] == '':
                print '\tERROR: Trace necessary for extraction for %s arm' %arm
                sys.exit()
            if opt == 0:
                print '\tFixed aperture of %i pixels (%.2f arcsec)' \
                        %(n, n*pxscale) 
            else:
                print '\tOptimal extraction with profile/trace parameters'               
            self.output[arm] = self.output[arm]+'_1d'
            bins = np.arange(len(self.data[arm][0]))
            y0, y1 = 0, len(bins) - 1
            dy0, dy1 = self.backrange[arm][0], self.backrange[arm][1]

            for i in range(len(self.data[arm])):
                mid = np.polyval(self.trace[arm][0], i) + offset

                if opt == 0:
                    sky = np.append(self.data[arm][i][y0:y0+dy0],
                                 self.data[arm][i][y1-dy1:y1])
                    skye = np.append(self.erro[arm][i][y0:y0+dy0],
                                 self.erro[arm][i][y1-dy1:y1])
                else:
                    if sig == '':
                        sigma = np.polyval(self.trace[arm][1], i)
                    else:
                        sigma = float(sig)                    
                    
                    if self.profile[arm] == 'moffat':
                        alpha = np.polyval(self.trace[arm][1], i)
                        beta = np.polyval(self.trace[arm][2], i)
                        sigma = alpha * 2 * (2**(1./beta) - 1 )**0.5 / 2.3538
                        prof = onedmoffat(bins, 0, 1, mid, alpha, beta)
                    else:
                        prof = mlab.normpdf( bins, mid, sigma )                    
                    
                    if (int(mid + 3*sigma) > y1) or (int(mid-3*sigma) < y0):
                        print '\t\tError in sky regions - check profile'
                        sys.exit()                        
                    
                    sky = np.append(self.data[arm][i][y0 : int(mid-2*sigma)],
                                self.data[arm][i][int(mid + 2*sigma) : y1])
                    skye = np.append(self.erro[arm][i][y0 : int(mid-2*sigma)],
                                self.erro[arm][i][int(mid + 2*sigma) : y1])
                skyrms.append( np.median(skye) )
                skysum.append( np.median(sky) )
                
                if n > 0 and opt == 0:
                    prof = np.zeros(max(bins)+1) + 1./n
                    prof[:mid-n/2], prof[mid+n/2+1:] = 0, 0
                    if n % 2 == 0:
                        prof[mid-n/2], prof[mid+n/2] = 1./(2*n), 1./(2*n)
                else:
                    pxscale = min(0.3, self.head[arm]['CDELT2'])
                    lim = n / pxscale / 2.3548
                    prof[:int(mid-lim)], prof[int(mid+lim):] = 0, 0
                
                prof *= 1./sum(prof)
                weightd = sum((self.data[arm][i] * prof).transpose()) \
                       / sum(prof**2)
                weightdata.append(weightd)
                if len(self.erro[arm]) != 0:
                    weighte = (sum((self.erro[arm][i]**2 * prof**2)).transpose() \
                       / sum(prof**2)**2)**0.5
                    weighterro.append(weighte)
                    
            self.skyrms[arm] = np.array(skyrms)
            self.oneddata[arm] = np.array(weightdata)
            self.onederro[arm] = np.array(weighterro)/errsc
            self.onedback[arm] = np.array(skysum)
            
            if len(self.ebvcorr[arm]) != 0 and '_ebv' not in self.output[arm]:
                print '\t\tCorrecting %s data for E_B-V = %.2f with R_V = %.2f' %(arm, self.ebv, self.rv)
                if len(self.data[arm]) != 0:
                    atmp = self.data[arm].transpose() * self.ebvcorr[arm]
                    btmp = self.erro[arm].transpose() * self.ebvcorr[arm]
                    self.data[arm] = atmp.transpose()
                    self.erro[arm] = btmp.transpose()
                if len(self.oneddata[arm]) != 0:
                    self.oneddata[arm] *= self.ebvcorr[arm]
                    self.onederro[arm] *= self.ebvcorr[arm]
                self.output[arm] = self.output[arm]+'_ebv'
            for a in (self.oneddata[arm], self.onederro[arm]):
                a[abs(a) < 1E-22] = 1E-22
            
            if self.redshift != 0:
                # Luminosity distance
                ld = LDMP(self.redshift)*3.08568e24 
                self.lumspec[arm] = self.oneddata[arm] * 4 * np.pi * ld**2
                self.lumerr[arm] = self.onederro[arm] * 4 * np.pi * ld**2
                self.restwave[arm] = self.wave[arm]/(1+self.redshift)

################################################################################

    def write1d(self, arms, lim1=3.20E3, lim2=2.28E4, error = 1, lum=0):
        for arm in arms:
            f = open('%s.spec' %self.output[arm], 'w')
            f.write('#Object: %s\n' %self.object)
            f.write('#Fluxes in [10**-%s erg/cm**2/s/AA] \n' %(str(self.mult)[-2:]))
#            f.write('#Redshift: %.4f\n' %self.redshift)
            if 'helio' in self.output[arm].split('_') and 'vac' in self.output[arm].split('_'):
                f.write('#Wavelength is in vacuum and in a heliocentric reference\n')
            if 'ebv' in self.output[arm].split('_'):
                f.write('#Fluxes are corrected for Galactic foreground\n')
            for i in range(len(self.onederro[arm])):
                if lim1 < self.wave[arm][i] < lim2:
                    if error == 1:
                        f.write('%.6f\t%.4e\t%.4e' \
                        %(self.wave[arm][i], self.oneddata[arm][i]*self.mult, \
                                        self.onederro[arm][i]*self.mult))
                        if len(self.cont[arm]) > 0:
                            f.write('\t%.4e' %( self.cont[arm][i]*self.mult))
                        f.write('\n')
                    else:
                        f.write('%.6f\t%.4e\n' \
                        %(self.wave[arm][i], self.oneddata[arm][i]*self.mult))           
            f.close()
            if lum != 0:
              if self.lumspec[arm] != '':
                f = open('%s_lum.spec' %self.output[arm], 'w')
                f.write('#Object: %s\n' %self.object)
                f.write('#Redshift: %.4f\n' %self.redshift)
                if 'helio' in self.output[arm].split('_') and 'vac' in self.output[arm].split('_'):
                    f.write('#Wavelength is at rest, in vacuum and heliocentric\n')
                if 'ebv' in self.output[arm].split('_'):
                    f.write('#Luminosities are corrected for Galactic foreground and slitloss\n')
                for i in range(len(self.restwave[arm])):
                    f.write('%.6f\t%.4e\t%.4e\t%.3f\t%.1f\n' \
                    %(self.restwave[arm][i], self.lumspec[arm][i]*(1+self.redshift), 
                      self.lumerr[arm][i]*(1+self.redshift), 
                      self.skytel[arm][i], abs(self.skyrad[arm][i])))
                f.close()                
                
################################################################################
                
    def smooth1d(self, arm, smoothn=7, filt='median'):
        self.soneddata[arm] = np.array(smooth(self.oneddata[arm], smoothn, filt))
        
################################################################################  

    def setCont(self, arm, smoothn = 20, meth = 'median', datval = None,
               intpl = 'pf', order = 7, s1 = 1, sig = 2.5, absl = 1):
        print '\tRemoving absorption lines for continuum fit'
        wltmp = np.array(self.wave[arm])
        # Excluding Lya
        wlsela = wltmp > 912 * (self.redshift + 1)
        wlsel01 = wltmp > 1215.6 * (self.redshift + 1) * 1.05
        wlsel02 = wltmp < 1215.6 * (self.redshift + 1) * 0.945
        wlsel03 = wltmp > 3200
        # Limiting response in each arm
        if arm in ['uvb']:
            wlsel04 = wltmp > 3100
            wlsel05 = wltmp < 5750
            wlsel = wlsela * (wlsel01 ^ wlsel02) * wlsel03 * wlsel04 * wlsel05
        if arm in ['vis']:
            wlsel04 = wltmp > 5510
            wlsel05 = wltmp < 10410
            wlsel = wlsela * (wlsel01 ^ wlsel02) * wlsel03 * wlsel04 * wlsel05
        if arm in ['nir']:
            wlsel04 = wltmp > 10020
            wlsel05 = wltmp < 22500
            wlsel = wlsela * (wlsel01 ^ wlsel02) * wlsel03 * wlsel04 * wlsel05
        if arm in ['all']:
            wlsel04 = wltmp > 3500
            wlsel05 = wltmp < 22500
            wlsel = wlsela * (wlsel01 ^ wlsel02) * wlsel03 * wlsel04 * wlsel05        

        # Use only the brightest datsel percent of all values
        if datval:
            valsel = sorted(self.oneddata[arm])[int(datval*len(wltmp))]
            datsel = self.oneddata[arm] > valsel
            wlsel *= datsel
        
        
        # Excluing strong telluric lines
        if arm in ['vis', 'nir']:        
            telsel = self.skytel[arm] > 0.99
        else:
            telsel = self.skytel[arm] > 0.
            
        wlsel *= telsel
        print '\t\tExcluding %s number of points due to tellurics' %len(telsel[telsel==False])
        # Excluing strong absorption lines
        if absl in [1, '1', 'Y', 'y', 'YES', True]:
            for absline in linelist:
                lstrength = linelist[absline][1]
                if lstrength > 0.1:
                    wlline = linelist[absline][0]*(1+self.redshift)
                    wlsell = (wlline-2 < wltmp) ^ (wltmp < wlline + 2)
                    wlsela *= wlsell
                    wlsel *= wlsell
                for redi in self.intsys: 
                    for intline in self.intsys[redi]:
                        wlline = linelist[intline][0] * (1 + redi)
                        wlsell = (wlline - 10 < wltmp) ^ (wltmp < wlline + 10)
                        wlsela *= wlsell
                        wlsel *= wlsell
            print '\t\tExcluding %s number of points due to absorption lines' \
                    %len(wlsela[wlsela==False])

        self.cleanwav[arm] = self.wave[arm][wlsel]
        self.cleandat[arm] = self.oneddata[arm][wlsel]
        self.cleanerr[arm] = self.onederro[arm][wlsel]
        print '\t\tDownsample the spectrum for continuum fit (factor %s)' %smoothn
        wltmp, dattmp, errtmp = binspec(self.wave[arm][wlsel],
                                       self.oneddata[arm][wlsel]*self.mult,
                                       self.onederro[arm][wlsel]*self.mult,
                                       wl = smoothn, meth = meth)
        for j in range(30):
            if intpl == 'model':
                dattmppJy = ergJy(dattmp/self.mult, wltmp)
                errtmpJy = errtmp/dattmp*dattmppJy
                params = plfit(wltmp, dattmppJy, err = errtmpJy,
                        params = [1, 1, 0, self.redshift],
                        fixed = [False, False, False, True])
                contfit = Jyerg(pl(wltmp, params[0][0], params[0][1], params[0][2], 
                             params[0][3]), wltmp)
            elif intpl == 'fmmodel':
                dattmppJy = ergJy(dattmp/self.mult, wltmp)
                errtmpJy = errtmp/dattmp*dattmppJy
                params = plfmfit(wltmp, dattmppJy, err = 1.0*errtmpJy,
                         params = [10., 0.78, 0.15, self.redshift, 2.93, 
                                   1.00, 4.60, 2.35, 0.00, -0.22],
                          fixed = [False, True, False, True, True, 
                                   True, True, False, False, False],
                                   quiet = True)
                contfit =  Jyerg(plfm(wltmp, params[0][0], 
                    params[0][1], params[0][2], params[0][3],
                    params[0][4], params[0][5], params[0][6],
                    params[0][7], params[0][8]), wltmp)
            elif intpl == 'spline':
                tck = interpolate.UnivariateSpline(wltmp, dattmp, 
                                                   s = np.median(dattmp)**2*s1)
                contfit = tck(wltmp)
            elif intpl == 'pf':                
                coeff = np.polyfit(wltmp, dattmp, order) 
                contfit = np.polyval(coeff, wltmp)
            
            diff = (contfit - dattmp)/errtmp
            sel = (diff < sig) * (diff > -8)
            dattmp = np.array(dattmp[sel])
            errtmp = np.array(errtmp[sel])
            wltmp = np.array(wltmp[sel])
            #s = len(wltmp)/(s1 - 4)
            if len(sel[sel == False]) <= 20/smoothn:
                break
            else:
                print '\t\tExcluding %s number of points' %len(sel[sel==False])

        if intpl == 'model':
            self.cont[arm] = pl(self.wave[arm], params[0][0], 
                     params[0][1], params[0][2], params[0][3])/self.mult
        elif intpl == 'spline':
            self.cont[arm] = tck(self.wave[arm]) / self.mult
        elif intpl == 'pf':                
            self.cont[arm] = np.polyval(coeff, self.wave[arm])/self.mult 
        elif intpl == 'fmmodel':
            self.cont[arm] = plfm(self.wave[arm], params[0][0], 
                    params[0][1], params[0][2], params[0][3],
                    params[0][4], params[0][5], params[0][6],
                    params[0][7], params[0][8])/self.mult            
        self.woabs[arm] = np.array(self.cont[arm])
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))        
        
        ax.set_yscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.plot(self.wave[arm], self.oneddata[arm]*self.mult, 'b')
        ax.plot(self.wave[arm], self.cont[arm]*self.mult, 'r')
        ax.plot(wltmp, dattmp, 'o', color = 'yellow', ms = 1.9)
        #ax.set_xscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xlim(min(wltmp), max(wltmp))  
        ax.set_ylim(min(dattmp)*0.9, max(dattmp)*1.5)
        fig.savefig('%s_continuum_%s.pdf' %(self.object, arm))

################################################################################

    def sn(self, arm, x1 = 1, x2 = -1):
        if x2 == -1:
            sn = np.median(self.oneddata[arm]/self.onederro[arm])
            mfl = [np.median(self.oneddata[arm]), 
                   np.average(self.oneddata[arm])]
        else:
            x1 = self.wltopix(arm, x1)
            x2 = self.wltopix(arm, x2)
            sn = np.median(self.oneddata[arm][x1:x2]/self.onederro[arm][x1:x2])
            mfl = [np.median(self.oneddata[arm][x1:x2]),
                   np.average(self.oneddata[arm][x1:x2])]
        return sn, mfl

################################################################################

    def wltopix(self, arm, wl):
        dl = (self.wave[arm][-1]-self.wave[arm][0]) / (len(self.wave[arm]) - 1)
        pix = ((wl - self.wave[arm][0]) / dl)
        return max(0, int(round(pix)))

################################################################################  
      
    def fitCont(self, arm, wl1 = 4500, wl2 = 5800, norm = 0, av = 0, beta = 0,
                red = 'smc', binx = 1):
        print('\tFitting continuum with afterglow model')
        x1, x2 = self.wltopix(arm, wl1), self.wltopix(arm, wl2)
        fitx = self.cleanwav[arm][x1:x2+1]
        fity = self.cleandat[arm][x1:x2+1]
        fite = self.cleanerr[arm][x1:x2+1]
        fitfld = ergJy(fity, fitx)
        fitfle = fite/fity*fitfld
        binfitw, binfitd, binfite = binspec(fitx, fitfld, fitfle, wl = binx)
        binfite *= 3.5
        fixed0, fixed1, fixed2 = False, False, False
        if norm !=0: fixed0 = True
        if av != 0: fixed2 = True
        if beta != 0: fixed1 = True
        params = plfit(binfitw, binfitd, err = binfite, red = red,
                       params = [norm, beta, av, self.redshift],
                       fixed = [fixed0, fixed1, fixed2, True])
        self.cont[arm] = Jyerg(pl(self.wave[arm], params[0][0], 
                        params[0][1], params[0][2], params[0][3], red = red),
                        self.wave[arm])
                            
        fig = plt.figure(figsize = (8,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        
        ax.plot(binfitw, params[1], '-')
        ax.plot(binfitw, binfitd, 'o', ms = 4, mec='black', color='grey')
        fig.savefig('%s_%s_res.pdf' %(self.object, arm))

################################################################################ 
       
    def dlaabs(self, arm, nh, nherr = 0, z = ''):
        if z == '':
            z = self.redshift
        print '\tAdding DLA with log NH = %.2f +/- %.2f at z=%.3f' %(nh, nherr,z)
        nh, nhp, nhm = 10**nh, 10**(nh+nherr), 10**(nh-nherr) 
        self.nh = nh
        wl = self.wave[arm]/(1+z) * 1E-10
        dlam = dlafunc(wl, nh)
        if nherr != 0:
            dlamp = dlafunc(wl, nhp) 
            dlamm = dlafunc(wl, nhm)
        x1 = self.wltopix(arm, 1180 * (z + 1))
        x2 = self.wltopix(arm, 1260 * (z + 1))
        chim = sum(((dlam/self.mult - self.oneddata[arm])**2 \
            /(self.onederro[arm])**2)[x1:x2])
        if self.oneddla[arm] == '': 
            if nherr != 0: self.oneddla[arm] = [dlam, dlamp, dlamm]
            else: self.oneddla[arm] = [dlam]
        else:
            if nherr != 0: self.oneddla[arm] = [self.oneddla[arm][0]*dlam, 
                    self.oneddla[arm][1]*dlamp, self.oneddla[arm][2]*dlamm]
            else:
                self.oneddla[arm] = [self.oneddla[arm]*dlam]
        return chim
        
################################################################################
        
    def writevp(self, arm, lya = 0, scl = 1.0):
        if lya != 0:
            fname = self.output[arm]+'_vpfit_lya.txt'
            y = self.woabs[arm]*self.mult*self.match[arm]
            yerr = self.onederro[arm]*self.mult
        else:
            fname = self.output[arm]+'_vpfit.txt' 
            y = self.oneddata[arm]*self.mult
            yerr = self.onederro[arm]*self.mult            
        f = open(fname, 'w')
        f.write('RESVEL %.2f\n' %self.reso[arm])
        for wl, fl, er, cont in zip(self.wave[arm], y, yerr, 
                                    self.cont[arm]*self.mult):
            f.write('%.6f\t%.4e\t%.4e\t%.4e\n' %(wl, fl, er/scl, cont))
        f.close()
        return fname

################################################################################

    def telcor(self, arms):
        haws = highabswin()
        for arm in arms:
            print '\tCreating telluric model for arm %s' %arm
            telcor = np.ones(len(self.wave[arm]))
            wl = self.wave[arm]
            f = open('%s_telcor_%s.txt' %(self.object, arm), 'w')
            for i in range(len(wl)):
                for haw in haws:
                    if haw[0] < wl[i] and wl[i] < haw[1]: 
                        if (self.cont[arm][i] / self.oneddata[arm][i]) > 1:
                            telcor[i] = self.cont[arm][i] / self.oneddata[arm][i]
                f.write('%.3f\t%.3f\n' %(wl[i], telcor[i]))
            f.close()
            self.telcorr[arm] = telcor
            self.telwave[arm] = self.wave[arm]
            fig = plt.figure(figsize = (10,5))
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))            
            
            ax.plot(self.wave[arm], self.telcorr[arm], 'black')
            fig.savefig('%s_telcor_%s.pdf' %(self.object, arm))
            
################################################################################   
            
    def appltel(self, arms):
        for arm in arms:
            print '\tCorrecting for tellurics arm %s' %arm
            tck = interpolate.InterpolatedUnivariateSpline(self.telwave[arm], 
                                               self.telcorr[arm])
            tellcor = tck(self.wave[arm])
            self.oneddata[arm] *= tellcor

################################################################################

    def stacklines(self, lines, nsmooth=10):
        normspecs, normerrs = [], []
#        normvels = []
        normvels = np.arange(-2000, 2000, 20)
        for line in lines:
            if abslist.has_key(line):
                # Cutout -25 AA to 25 AA
                dx1, dx2 = 400, 300
                redwl = abslist[line][0]*(1+self.redshift)

                if 3100 < redwl < 5500: arm = 'uvb'
                elif 5500 < redwl < 10000: arm = 'vis'
                elif 10000 <  redwl < 25000: arm = 'nir'
                else:
                    print 'Line %s not in wavelength response' %line
                
                pix1, pix2 = self.wltopix(arm, redwl-dx1), self.wltopix(arm, redwl+dx1)
                x = (self.wave[arm][pix1:pix2]-redwl)/redwl * c
                y = self.mult * self.oneddata[arm][pix1:pix2]
                yerr = self.mult * self.onederro[arm][pix1:pix2]

                cont = np.median(np.append(y[ : dx2 ],  y[ : -dx2]))
#                normspecs.append(y/cont)
#                normvels.append(x)
                tck = interpolate.InterpolatedUnivariateSpline(x, y/cont)                
                normspecs.append(tck(normvels))
                tck = interpolate.InterpolatedUnivariateSpline(x, yerr/cont)                
                normerrs.append(tck(normvels))

        medspec = np.median(np.array(normspecs), axis = 0)
        mederr = np.average(np.array(normerrs), axis = 0)/len(lines)**0.5

        fig = plt.figure(figsize = (5,6))
        fig.subplots_adjust(left=0.14, right=0.95)
        ax = fig.add_subplot(1, 1, 1)
#        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))            
        ax.errorbar(normvels, medspec, mederr,
                    drawstyle = 'steps-mid', capsize = 0, color = 'black')
        ax.plot(normvels, smooth(medspec, window_len=nsmooth, window='hanning'), 
        'red', lw=1.5)
#        for ms in normspecs:
#            ax.plot(normvels, ms, 'grey')
        ax.set_xlim(-600, 600)
        ax.set_ylim(-0.3, 2.8)
        ax.set_xlabel(r'$\rm{Velocity\,(km\,s^{-1})}$')
        ax.set_ylabel(r'$\rm{Normalized\,flux}$')

        fig.savefig('%s_lines.pdf' %(self.object))   
                