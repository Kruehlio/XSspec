#!/usr/bin/env python
import os
import pyfits
import numpy as np

from matplotlib import rc
from matplotlib.patheffects import withStroke
from scipy import interpolate, constants

from .functions import (blur_image, ccmred)
from .astro import (airtovac, vactoair, absll, emll, isnumber, getebv, binspec)

from .postproc import (checkWL, scaleSpec, applyScale, fluxCor, 
                      telCor, applTel)
from .onedspec import makeProf, extr1d, write1d, smooth1d, writeVP
from .analysis import (setCont, fitCont, dlaAbs, stackLines, setMod,
                      setAscMod, scaleMod)

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

    def set1dFiles(self, arm, filen, mult = 10, errsc = 1., mode = 'txt', dAxis=1):
        wlkey, wlstart = 'NAXIS%i'%dAxis, 'CRVAL%i'%dAxis
        wlinc, wlpixst = 'CDELT%i'%dAxis, 'CRPIX%i'%dAxis
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
            pix = np.arange(self.head[arm][wlkey]) + 1
            self.wave[arm] = self.head[arm][wlstart] \
                + (pix-self.head[arm][wlpixst])*self.head[arm][wlinc]*mult
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
            self.inst = self.head[arm]['INSTRUME']
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
            pix = np.arange(self.head[arm][wlkey]) + 1
            self.wave[arm] = (self.head[arm][wlstart] \
                + (pix-self.head[arm][wlpixst])*self.head[arm][wlinc])*mult
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
            
    def scaleSpec(self, arm, **kwargs):
        scaleSpec(self, arm, **kwargs)  

    def applyScale(self, arms, **kwargs):
        applyScale(self, arms, **kwargs)

    def setMod(self, arms, norm, **kwargs):
        setMod(self, arms, norm, **kwargs)
                
    def setAscMod(self, arms, mfile, **kwargs):
        setAscMod(self, arms, mfile, **kwargs)

    def scaleMod(self, arms, p = ''):
        scaleMod(self, arms, p = '')
                
    def makeProf(self, arm, **kwargs):
        makeProf(self, arm, **kwargs)
    
    def extr1d(self, arms, **kwargs):
        extr1d(self, arms, **kwargs)

    def write1d(self, arms, **kwargs):
        write1d(self, arms, **kwargs)
                
    def smooth1d(self, arm, **kwargs):
        smooth1d(self, arm, **kwargs)
        
    def setCont(self, arm, **kwargs):
        setCont(self, arm, **kwargs)

    def fitCont(self, arm, **kwargs):
        fitCont(self, arm, **kwargs)

    def dlaabs(self, arm, nh, **kwargs):
        chim = dlaAbs(self, arm, nh, **kwargs)
        
    def fluxCor(self, arm, fluxf, countf):
        fluxCor(self, arm, fluxf, countf)        

    def checkWL(self, arms, **kwargs):
        checkWL(self, arms, **kwargs)
        
    def writevp(self, arm, **kwargs):
        fname = writeVP(self, arm, **kwargs)

    def telcor(self, arms):
        telCor(self, arms)
            
    def appltel(self, arms):
        applTel(self, arms)

    def stacklines(self, lines, **kwargs):
        stackLines(self, lines, **kwargs)
                