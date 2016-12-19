# -*- coding: utf-8 -*-
"""
Miscellaneous functions for astronomical use 
"""
__version__ = "0.2"

import os 
import pyfits
import operator
from xml.dom import minidom
from urllib2 import urlopen

from scipy import (special, log10, array, sqrt, sin, 
                   exp, log, average, 
                   arange, meshgrid, std)
from numpy.random import random_sample     
import numpy as np  
         
c, c1 = 2.99792458E5, 2.99792458E8

############################################

def geterrs(art, sigma = 1):
    ar = array(art)
    ar.sort()
    lenar = len(ar)
    mini = special.erfc(sigma/2**0.5)/2
    maxi = 1-mini
    bestval, minval, maxval = ar[int(lenar*0.5)], ar[int(lenar*mini)], ar[int(lenar*maxi)]
    return bestval, bestval-minval, maxval-bestval    

###########################################

def getebv(ra, dec, rv=3.08):
    """ Gets the Galactic foreground EB-V from the Schlafly and Finkbeiner
    maps using the http://irsa.ipac.caltech.edu/ interfact. In case they change
    their xml TagNames (Happend once) it might return completely crazy values.
    Be carful and reasonable.
    Arguments:
        ra = Right Ascension in degrees
        dec = Declination in degrees
        RV = Total-to-selective reddening (default 3.08)
    Returns Mean EBV, StdDec EBV, RefPixel EBV and Mean EBV*RV
    """
    
    if dec > 0: dec = '++'+str(dec)
    else: dec = '+'+str(dec)
    ra = str(ra)
    try:
        urlstring = 'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr='+ra+'+'+dec
        ebvall = minidom.parse(urlopen(urlstring))
        ebv = float(str(ebvall.getElementsByTagName('meanValueSandF')[0].firstChild.data).split()[0])
        std = float(str(ebvall.getElementsByTagName('stdSandF')[0].firstChild.data).split()[0])
        ref = float(str(ebvall.getElementsByTagName('refPixelValueSandF')[0].firstChild.data).split()[0])
        av = ebv*rv
    except:
        print "\t\tE_B-V can not be parsed"
        ebv, std, ref, av = '', '', '', ''
    return ebv, std, ref, av
    
    
################################################################################

def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]

#########################################################s#######################


def skylines(intensity=10, filen='~/tools/etc/sky_lines.dat'):
    """ Returns OH skylines lines above a given intensity, file from  
    http://www.eso.org/sci/facilities/paranal/instruments/isaac/tools/oh/index.html 
    from the ISAAC OH atlas and in vacuum wavelengths, averages skylines closer than
    0.2 AA
    """ 

    filen = os.path.expanduser(filen)
    fin = open(filen, 'r')
    lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
    fin.close()  
    sky, inten = [], []
    for line in lines:
        if line != []:
            if isnumber(line[0]) and line[1]:
                sky.append(float(line[0]))
                inten.append(float(line[1]))
    sky, inten = array(sky), array(inten)
    sky = sky[inten>intensity]
    inten = inten[inten>intensity]
    skynew, skipnext = [], 0
    for i in range(len(sky)):
        if i < len(sky)-1:
            if sky[i+1] - sky[i] < 0.4:
                skynew.append((sky[i]*inten[i] + sky[i+1]*inten[i+1])/(inten[i]+inten[i+1]))
                skipnext = 1
            elif skipnext == 0:
                skynew.append(sky[i])
            elif skipnext == 1:
                skipnext = 0
        elif skipnext == 0:
            skynew.append(sky[i])
    return array(skynew)  


#####################################################


def bootstrap(data, n_samples=10000):
    data = np.array(data)
    """ Given data, where axis 0 is considered to delineate points,
    return a list of arrays where each array is a set of bootstrap indexes.
    """
    return array([np.random.randint(data.shape[0], size = data.shape[0]) \
                    for a in arange(0,n_samples)])


#####################################################


def bootstraps(data):
    """ Given data, where axis 0 is considered to delineate points,
    return a bootstraped list.
    """
    return np.random.randint(data.shape[0],size=data.shape[0])


#####################################################


def errs(vals, sigma = 1, ul = 2):
    """ Median value and asymetric 1\sigma errorbars, 2s upper limitfor a list / array:
        Returns Median / 1s minus error / 1s plus error / 2s upper limit"""
    if len(vals) > 18:
        sigman = {1: 68.27, 2: 95.45, 3: 99.73, 4: 99.993665}
        vall = (1 - sigman[sigma] / 100)/2
        valh = 1 - vall
        valn = sorted(list(vals))
        valmed = valn[len(valn)/2]
        valmin = valn[int(round(vall*len(valn)))]-valmed
        valmax = valn[int(round(valh*len(valn)))]-valmed
        val2s  = valn[int(round(0.9545*len(valn)))]
        val3s  = valn[int(round(0.9973*len(valn)))]
    else:
        valmed, valmin, valmax, val2s = vals[0], 0, 0, vals[0]
    if ul == 2:
        return valmed, valmin, valmax, val2s
    elif ul == 3:
        return valmed, valmin, valmax, val3s
        

   
#####################################################
def ellShape(x, y, xm, xs, ym, ys, m=1, a=0.7, b=2):
    """ Returns the probability distribution for two quantities with mean xm, ym
    and sigma xs, ys, both quantities can be related if m != 1. m is the scatter
    of the relation (currently a linear relation)
    """
    x = arange(x[0]-x[2], x[1]+x[2], x[2])
    y = arange(y[0]-y[2], y[1]+y[2], y[2])
    zgrid = []
    for y1 in y:
        z = []
        for x1 in x:
            prob1 = exp(((x1-xm)**2)/(-2*xs**2))
            prob2 = exp(((y1-ym)**2)/(-2*ys**2))
            prob3 = m
            if m != 1:
                # Relates x to y linearly via y = a*x + b, b = 2, and a = 0.7 here
                # More complex functional forms are easily implemented
                # m is the scatter of the relation
                prob3 = exp(((a*x1 + b) - y1)**2 / (-2 * m**2))
            z.append(-1*prob1*prob2*prob3)
        zgrid.append(z)
    zgrid = array(zgrid)
    X, Y = meshgrid(x, y)
    return X, Y, zgrid


#####################################################
def getHead(name, head, ret='NA'):
    """ Returns fits header keyword 'head' from image 'name', if header keyword
    is not present, returns NA 
    """
    hdulist = pyfits.open(name)
    h = hdulist[0].header
    res = h.get(head, ret)
    hdulist.close()
    return res
    
#####################################################
def writeHead(name, head, value, comment=''):
    """ Writes fits header keyword 'head' with 'value' into image 'name' with
    comment 'comment' (default '')
    """
    hdulist = pyfits.open(name, mode='update')
    h = hdulist[0].header
    try:
        h[head] = (value, comment)
    except KeyError:
        h.update(head, value, comment = comment)
    hdulist.flush()
    hdulist.close(output_verify = 'silentfix')

        
#####################################################

def airtovac(wl):
    """ This corrects wavelengths from air to vac system wavelength in Angstrom
    Based in idlastro library
    """
    wl = array([wl])
    wlm, wln = wl[wl >= 2000], wl[wl < 2000]
    sigma2 = (1.E4/wlm)**2
    fact = 1. + 5.792105E-2/(238.0185-sigma2) + 1.67917E-3/(57.362-sigma2)
    wlv = wlm * fact
    return np.append(wln, wlv)

#####################################################

def vactoair(wl):
    """ This corrects wavelengths from vac to air system, wavelength in Angstrom
    Based in idlastro library
    """
    wl = array([wl])
    wlm, wln = wl[wl >= 2000], wl[wl < 2000]
    sigma2 = (1.E4/wlm)**2
    fact = 1. + 5.792105E-2/(238.0185-sigma2) + 1.67917E-3/(57.362-sigma2)
    wlv = wlm / fact
    return np.append(wln, wlv)
#####################################################

def LDMP(z, H0=67.3, WM=0.315, WV=0.685, v = 0):
    """ Cosmo calculator given the cosmological parameters z, H0, WM, WV returns
    the luminosity distance, angular seperation, and possibly many more. Based on
    http://www.astro.ucla.edu/~wright/CC.python
    """

    c = 299792.458 # velocity of light in km/sec
    Tyr = 977.8    # coefficent for converting 1/H into Gyr

    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    n = 5000
    i = arange(n)
    
    if not hasattr(z, '__iter__'):
        z = np.array([float(z)])

    zage_Gyra = np.array([])
    for zs in z:
        az = 1.0 / (1 + zs)
        a = az * (i + 0.5) / n
        adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = sum(1./adot)
        zage = az*age/n
        zage_Gyr = (Tyr/H0)*zage
        zage_Gyra = np.append(zage_Gyra, zage_Gyr)

    if v == 'age':
        return zage_Gyra
        
    DTT, DCMR = 0.0, 0.0
    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    a = az+(1-az)*(i+0.5)/n
    adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
    DTT = sum(1./adot)
    DCMR = sum(1./(a*adot))

    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n
    age = DTT+zage
    age_Gyr = age*(Tyr/H0)
    DTT_Gyr = (Tyr/H0)*DTT
    DCMR_Gyr = (Tyr/H0)*DCMR
    DCMR_Mpc = (c/H0)*DCMR
    # tangential comoving distance
    ratio = 1.00
    x = sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0: ratio =  0.5*(exp(x)-exp(-x))/x
        else: ratio = sin(x)/x
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.
    DCMT = ratio*DCMR
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806
    DA_Gyr = (Tyr/H0)*DA
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
    DL_Gyr = (Tyr/H0)*DL
    # comoving volume computation
    ratio = 1.00
    x = sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0: ratio = (0.125*(np.exp(2.*x)-np.exp(-2.*x))-x/2.)/(x*x*x/3.)
        else: ratio = (x/2. - np.sin(2.*x)/4.)/(x*x*x/3.)
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/5. + (2./105.)*y*y
    VCM = ratio*DCMR*DCMR*DCMR/3.
    V_Gpc = 4.*np.pi*((0.001*c/H0)**3)*VCM
    DL_cm = DL_Mpc * 3.08568E24

    if v == 1:
        print '\tH_0 = %1.1f' % H0 + ', Omega_M = ' + '%1.2f' % WM + ', Omega_vac = %1.2f' % WV + ', z = ' + '%1.3f' % z
        print '\tIt is now %1.3f' % age_Gyr + ' Gyr since the Big Bang.'
        print '\tAge at redshift z was %1.3f' % zage_Gyr + ' Gyr.'
        print '\tLight travel time was %1.3f' % DTT_Gyr + ' Gyr.'
        print '\tComoving radial distance is \t%1.1f' % DCMR_Mpc + ' Mpc or ' + '%1.1f' % DCMR_Gyr + ' Gly.'
        print '\tComoving volume within redshift z ' + '%1.1f' % V_Gpc + ' Gpc^3.'
        print '\tAngular size distance D_A is ' + '%1.1f' % DA_Mpc + ' Mpc or %1.1f' % DA_Gyr + ' Gly.'
        print '\tAngular scale of %.2f' % kpc_DA + ' kpc/".'
        print '\tLuminosity distance D_L is %1.1f' % DL_Mpc + ' Mpc or ' + '%1.4e' % DL_cm + ' cm.'
        print '\tDistance modulus, m-M, is %1.2f mag' % (5*log10(DL_Mpc*1e6)-5)
        print '\tK-correction for equal effective wavelength %1.2f' %(-2.5*log10(1+z))
        return(DL_Mpc, kpc_DA)
    elif v == 2:
        return(DL_Mpc, kpc_DA)
    else:
        return(DL_Mpc)


def isnumber(num):
    """ Checks whether argument is number"""    
    try:
        float(num)
        return True
    except ValueError:
        return False
        
###########################################
def ergJy(erg, wl):
    """ Converts erg/cm^2/s/AA into myJy """
    return (erg / c * wl**2 * 1E17 / 10)
    
###########################################
    
def Jyerg(jy, wl):
    """ Converts muJy into erg/cm^2/s/AA """
    return (jy * c / wl**2 / 1E17 * 10)
###########################################
def abflux(ab):
    try:
        return 10**((23.9 - float(ab))/2.5)
    except TypeError:
        return 10**((23.9 - ab)/2.5)
###########################################
def fluxab(flux):
    try:
        return -2.5*log10(float(flux))+23.9
    except TypeError:
        return -2.5*log10(flux)+23.9
###########################################
def abfluxerr(ab, err):
    ab, err = float(ab), float(err)
    flux = abflux(ab)
    return [abflux(ab-err)-flux, flux-abflux(ab+err)]
###########################################
def fluxaberr(flux, err):
    flux, err = float(flux), float(err)
    return (fluxab(flux-err)-fluxab(flux+err))/2.
###########################################
def addzero(val, n):
    if isnumber(val):
        val = float(val)
        if n == 1: valr = '%.0f' %val
        if n == 2: valr = '%.2f' %val
        if n == 3: valr = '%.3f' %val
        if float(val) < 10:
            if n == 1: valr = '0%.0f' %val
            if n == 2: valr = '0%.2f' %val
            if n == 3: valr = '0%.3f' %val
    return valr
###########################################
def deg2sexa(ra, dec):
    retra = ra
    if isnumber(ra):
        ra = float(ra)
        hours = int(ra/15)
        minu = int((ra/15.-hours)*60)
        seco = float((((ra/15.-hours)*60)-minu)*60)
        retra = '%s:%s:%s' %(addzero(hours,1), addzero(minu,1), addzero(seco,3))
    retdec = dec
    if isnumber(dec):
        dec = float(dec)
        degree = int(dec)
        minutes = int((dec-degree)*60)
        seconds = float((((dec-degree)*60)-minutes)*60)
        if dec < 0:
            retdec = '-%s:%s:%s' %(addzero(-1*degree,1), addzero(-1*minutes,1), addzero(-1*seconds,2))
        else:
            retdec = '+%s:%s:%s' %(addzero(degree,1), addzero(minutes,1), addzero(seconds,2))
    return retra, retdec
###########################################

def sexa2deg(ra, dec):
    if isnumber(ra):
        retra = ra
    else:
        ra = ra.split(':')
        retra = (float(ra[0])+float(ra[1])/60.+float(ra[2])/3600.)*15

    if isnumber(dec):
        retdec = dec
    else:
        dec = dec.split(':')
        if dec[0][0] == '-':
            retdec = float(dec[0])-float(dec[1])/60.-float(dec[2])/3600.
        else:
            retdec = float(dec[0])+float(dec[1])/60.+float(dec[2])/3600.
    return round(float(retra),6), round(float(retdec), 6)


##########################################################

def smooth(x, window_len=11, window='hanning'):
       if x.ndim != 1:
           raise ValueError, "smooth only accepts 1 dimension arrays."
       if x.size < window_len:
           raise ValueError, "Input vector needs to be bigger than window size."
       if window_len<3:
           return x
       if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
           raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

       s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
       if window == 'flat': #moving average
           w=np.ones(window_len,'d')
       else:
           w=eval('np.'+window+'(window_len)')
       y=np.convolve(w/w.sum(),s,mode='same')
       return y[window_len:-window_len+1]

##########################################################

def binspec(xs, ys, yerr = '', wl=3, meth = 'average', clip = 3, do_weight = 1):
    wavebin, databin = [], []
    if len(xs) != len(ys):
        print '\tLength of input arrays must be equal, but are %i and %i' %(len(xs), len(ys))
    xs, ys = array(xs), array(ys)
    if len(yerr) != 0: yerr, errobin = array(yerr), []
        
    for i in range(len(xs)/int(wl)):
        data, wave = array(ys[i*wl:(i+1)*wl]), array(xs[i*wl:(i+1)*wl])
        wavebin.append(np.nanmean(wave))
        meddata = np.nanmedian(data)
        if len(yerr) == 0:
            stderr = std(data)
        else:
            erro = yerr[i*wl:(i+1)*wl]
            stderr = average(erro)

        if clip != 0:
            while True:
                meddata = np.nanmedian(data)
                if len(yerr) == 0: 
                    stderr = np.nanstd(data)
                else: 
                    stderr = np.nanmean(erro)  
                    # If Standarddeviation >> Error in spectrum prevent excluding all
                    if np.nanstd(data) > 3*stderr:
                        stderr = np.nanstd(data)
                sel = (data < meddata + clip*stderr) * (data > meddata - clip*stderr)
                if sel.all() == True or len(data) <= wl/2:
                    break  
                data = array(data[sel])
                if len(yerr) != 0: erro = array(erro[sel])

        if do_weight in [0, 'False', False, 'N', 'no', 'n', 'No'] or len(yerr) == 0:
            if meth == 'median':
                databin.append(np.nanmedian(ys[i*wl:(i+1)*wl]))
                if len(yerr) != 0:
                    errobin.append(np.nansum(yerr[i*wl:(i+1)*wl]**2)**0.5 / wl)
            elif meth == 'average':
                databin.append(np.nanmean(ys[i*wl:(i+1)*wl]))
                if len(yerr) != 0:
                    errobin.append(np.nansum(yerr[i*wl:(i+1)*wl]**2)**0.5 / wl)
        else:
            mult = 1./min(erro[erro > 0])
            weight = (1. / (mult*erro)**2 )
            databin.append( np.nansum (data * weight) / np.nansum(weight) )
            errobin.append((np.nansum(erro**2)**0.5)/len(erro))  
    if len(yerr) != 0:
        return array(wavebin), array(databin), array(errobin)
    else:
        return array(wavebin), array(databin)
        
        
##########################################################
        
        
def AvlawsFM(y, law,  rv = 3.08):
    # y is wavelength in nm
    if law == 'smc':	
        c1, c2, c3, c4, x0, gam = -4.959, 2.264, 0.389, 0.461, 4.6, 1.
    x = 1./(y*1.e-3)
    D = x**2/((x**2 - x0**2)**2 + x**2*gam**2)
    F = 0.5392*(x-5.9)**2 + 0.05644*(x-5.9)**3
    F[x<5.9] = 0.
    extlaw = c1 + c2*x + c3*D + c4*F
    return (extlaw/rv+1.)    
    
    
##########################################################

def Avlaws(y, law):
    # y is wavelength in nm
    if law == 'mw':
        rv, mult = 3.08, 1.35649
        a1, l1, b1, n1 = 165.,47.,90.,2.
        a2, l2, b2, n2 = 14.,80.,4.,6.5
        a3, l3, b3, n3 = 0.045, 220.,-1.95,2.
        a4, l4, b4, n4 = 0.002,9700.,-1.95,2.
        a5, l5, b5, n5 = 0.002,18000.,-1.8,2.
        a6, l6, b6, n6 = 0.012,25000.,0.,2.
    if law == 'smc':
        rv, mult = 2.93, 1.3875
        a1, l1, b1, n1 = 185.,42.,90.,2.
        a2, l2, b2, n2 = 27.,80.,5.5,4.0
        a3, l3, b3, n3 = 0.005, 220.,-1.95,2.
        a4, l4, b4, n4 = 0.010,9700.,-1.95,2.
        a5, l5, b5, n5 = 0.012,18000.,-1.8,2.
        a6, l6, b6, n6 = 0.030,25000.,0.,2.
    if law == 'lmc':
        rv, mult = 3.16, 1.3165
        a1, l1, b1, n1 = 175.,46.,90.,2.
        a2, l2, b2, n2 = 19.,80.,5.5,4.5
        a3, l3, b3, n3 = 0.028, 220.,-1.95,2.
        a4, l4, b4, n4 = 0.005,9700.,-1.95,2.
        a5, l5, b5, n5 = 0.006,18000.,-1.8,2.
        a6, l6, b6, n6 = 0.020,25000.,0.,2.
    if law == 'sn':
        rv, mult = 1, 1
        a1, l1, b1, n1 =157.8,67.7,136.97,2.56
        a2, l2, b2, n2 =157.4,79.73,23.59,5.87
        a3, l3, b3, n3 =0.2778, 272.8,-1.87,0.87
        a4, l4, b4, n4 =0.005,9700.,-1.95,2.
        a5, l5, b5, n5 =0.006,18000.,-1.8,2.
        a6, l6, b6, n6 =0.020,25000.,0.,2.
    if law in ('smc', 'lmc', 'mw', 'sn'):
        c1 = a1/((y/l1)**n1+(l1/y)**n1+b1)
        c2 = a2/((y/l2)**n2+(l2/y)**n2+b2)
        c3 = a3/((y/l3)**n3+(l3/y)**n3+b3)
        c4 = a4/((y/l4)**n4+(l4/y)**n4+b4)
        c5 = a5/((y/l5)**n5+(l5/y)**n5+b5)
        c6 = a6/((y/l6)**n6+(l6/y)**n6+b6)
        extlaw = rv * (c1+c2+c3+c4+c5+c6) * mult/rv
    elif law == 'cal':
        rv = 4.05
        #p11 = 1/0.11
        #ff11 = 2.659*(-2.156+1.509*p11-0.198*p11**2+0.011*p11**3)+rv
        #p12 = 1/0.12
        #ff12 = 2.659*(-2.156+1.509*p12-0.198*p12**2+0.011*p12**3)+rv
        #slope1 = (ff12-ff11)/100.
        ff99 = 2.659*(-1.857+1.040/2.19)+rv
        ff100 = 2.659*(-1.857+1.040/2.2)+rv
        slope2 = (ff100-ff99)/100.
        p = 1./(y*1E-3)
        if y > 2200:
            ff=(ff99+(y-2190.)*slope2)/rv
        elif (y <= 2200) and (y>630):
            ff=(2.659*(-1.857+1.040*p)+rv)/rv
        elif (y <= 630) and (y>90):
            ff=(2.659*(-2.156+1.509*p-0.198*p**2+0.011*p**3)+rv)/rv
        else:
            ff = 0
        extlaw = ff
    else:
        extlaw = y*0
    return extlaw

def voigt_m(x, y):
    z = x + 1j*y
    return special.wofz(z).real

def Voigt(nu, alphaD, alphaL, nu_0, A, a=0, b=0):
   # The Voigt line shape in terms of its physical parameters
   f = sqrt(log(2))
   x = (nu-nu_0)/alphaD * f
   y = alphaL/alphaD * f
   backg = a + b*nu
   V = A*f/(alphaD*sqrt(np.pi)) * voigt_m(x, y) + backg
   return V
# Half width and amplitude
#c1 = 1.0692
#c2 = 0.86639
#hwhm = 0.5*(c1*alphaL+np.sqrt(c2*alphaL**2+4*alphaD**2))
#f = np.sqrt(ln2)
#y = alphaL/alphaD * f
#amp = A/alphaD*np.sqrt(ln2/np.pi)*voigt(0,y)

def emll():
    return {'Lyalpha' : [1215.670, 1], '[OII](3726)' : [3727.092, 1],
            '[OII](3729)' : [3729.875, 1], '[OII](3728)' : [3728.30, 1],
             'Htheta' : [3798.98, 2], 'Heta' : [3836.47, 2],
            '[NeIII](3869)' : [3869.81, 2], '[NeIII](3968)' : [3968.53, 2],
            'HeI' : [3889.00, 2], 'Hzeta' : [3890.15, 2],
            'Hepsilon' : [3971.20, 2], '[SII](4072)' : [4072.30, 2],
            'Hdelta' : [4102.89, 2], 'Hgamma' : [4341.69, 2],
            '[OIII](4363)' : [4364.44, 2],  'HeII(4686)' : [4687.311 ,2],
            'Hbeta' : [4862.68, 1], '[OIII](4931)' : [4932.603, 2],
            '[OIII](4959)' : [4960.30, 1], '[OIII](5007)' : [5008.24, 1],
            'HeII(5411)' : [5413.030 ,2],
            '[NII](5755)':[5756.24, 2], 
            '[OI](6301)' : [6302.046, 2], '[OI](6364)' : [6365.536, 2],
            '[NI](6528)' : [6529.03, 2], '[NII](6548)' : [6549.86, 2],
            'Halpha' : [6564.61, 1], '[NII](6584)' : [6585.27, 2],
            '[SII](6717)' : [6718.29, 2], '[SII](6731)' : [6732.67, 2]}

def absll():
    """ Returns absorption lines and their average equivalent width
    in GRB-DLAs from Lise's paper"""
    
    
    return {'H2J0_1092': [1092.1952, 0.005], 'H2J0_1077': [1077.138, 0.0116],
    'H2J0_1062': [1062.882, 0.0178], 'H2J0_1049': [1049.3674, 0.02319],   
    'H2J0_1036': [1036.545, 0.0268], 'H2J0_1024': [1024.3739, 0.0287],
    'H2J0_1012': [1012.813, 0.0297], 'H2J0_1008': [1008.5519, 0.0153],
    'H2J0_1001': [1001.823, 0.0267], 'H2J0_991': [991.37, 0.0261],
    'H2J0_985': [985.633, 0.0240], 'H2J0_981': [981.43, 0.020],
            'Lyg_972': [972.536, 0], 'CIII_977': [977.020, 0],
            'SiII_989': [989.873, 0], 'NII_989': [989.799, 0],
            'SiII_1012': [1012.502, 0], 'SiII_1020': [1020.6989, 0],
            'MgII_1026': [1026.1134, 0], 'Lyb_1025': [1025.728, 7.],
            'ArI_1066': [1066.660, 0], 'FeII_1081': [1081.8748, 0],
            'NII_1083': [1083.990, 0], 'FeII_1125': [1125.4478, 0],
            'SiII_1190': [1190.254, 0], 'SiII_1193': [1193.2897, 0],
            'SiII_1206': [1206.500, 0], 'Lya_1215': [1215.67, 73],
             'NV_1238': [1238.82, 0.14], 'NV_1242': [1242.80, 0.07],
             'SII_1250': [1250.58, 0.15],'SII_1253': [1253.52, 0.24],
             'SII_1259': [1259.52, 0.05], 'SiII_1260': [1260.42, 1.26],
             'SiII^*_1264': [1264.74, 0.66], 'CI_1277': [1277.25, 0.09],
             'OI_1302': [1302.17, 0.5], 'SiII_1304': [1304.37, 2.29],
             'OI^*_1304': [1304.86, 0.05], 'SiII^*_1309': [1309.28, 0.27],
            'NiII_1317': [1317.22, 0.11], 'CI_1328': [1328.83, 0.08],
            'CII_1334': [1334.53, 1.73], 'CII^*_1335': [1335.71, 0.5],
            'CII_1347': [1347.24, 0.20], 'OI_1355': [1355.60, 0.12],
            'NiII_1370': [1370.13, 0.13], 'SiIV_1393': [1393.76, 0.95],
            'SiIV_1402': [1402.77, 0.68], 'GaII_1414': [1414.40, 0.05],
            'NiII_1415': [1415.72, 0.16],  'CO_1419': [1419.0, 0.06],
            'NiII_1454': [1454.84, 0.08], 'ZnI_1457': [1457.57, 0.08],
            'CO_1477': [1477.565, 0.01], 'CO_1509': [1509.748, 0.01],
            'CO_1544': [1544.448, 0.01], 'CO_1392': [1392.525, 0.01],
            'SiII_1526': [1526.71, 0.93], 'SiII^*_1533': [1533.43, 0.42],
            'CoII_1539': [1539.47, 0.03],  'CIV_1548': [1548.20, 2.18],
            'CIV_1550': [1550.77, 2.18],'CI_1560': [1560.31, 0.09],
            'FeII^*_1570': [1570.25, 0.08], 'FeII^*_1602': [1602.49, 0.08],
            'FeII_1608': [1608.45, 0.85], 'FeII_1611': [1611.20, 0.18],
            'FeII^*_1618': [1618.47, 0.03], 'FeII^*_1621': [1621.69, 0.11],
            'FeII^*_1629': [1629.16, 0.05], 'FeII^*_1631': [1631.13, 0.1],
            'FeII^*_1634': [1634.35, 0.05], 'FeII^*_1636': [1636.33, 0.05],
            'FeII^*_1639': [1639.40, 0.5], 'CI_1656': [1656.93, 0.14],
            'AlII_1670': [1670.79, 1.04], 'SiI_1693': [1693.29, 0.07]	,#1693.03	0.07 ± 0.02
            'NiII_1703': [1703.41, 0.14], 'NiII_1709': [1709.60, 0.08],#1709.45	0.08 ± 0.02
            'NiII_1741': [1741.55, 0.14],  'MgI_1747': [1747.79, 0.05],#1748.02	0.05 ± 0.01	b
            'NiII_1751': [1751.92, 0.09], 'NiII_1804': [1804.47, 0.03],#1805.31	0.03 ± 0.01	a
            'SI_1807': [1807.31, 0.01], 'SiII_1808': [1808.01, 0.29],#			a
            'SiII^*_1816': [1816.93, 0.12], 'SiII^*_1817': [1817.45, 0.12],#			a, b
            'MgI_1827': [1827.93, 0.09], 'NiII_1842': [1842.89, 0.12],#	1844.27	0.12 ± 0.02	b
            'AlIII_1854': [1854.72, 0.89], 'AlIII_1862': [1862.79, 0.68],#	1863.56	0.68 ± 0.02
            'SiIII_1892': [1892.03, 0.10], 'CoII_1941': [1941.29, 0.07],#	1941.97	0.07 ± 0.01	a, b
            'CoII_2012': [2012.17, 0.06], 'CrII_2017': [2017.57, 0.08],#2017.61	0.08 ± 0.02	b
            'ZnII_2026': [2026.14, 0.60], 'CrII_2026': [2026.27, 0.05],#2025.97	0.60 ± 0.02	a
            'MgI_2026': [2026.48, 0.05], 'CrII_2056': [2056.26, 0.19],#	2055.51	0.19 ± 0.02
            'CrII_2062': [2062.23, 0.05], 'ZnII_2062': [2062.66, 0.53],#		a
            'CrII_2066': [2066.16, 0.12],#2066.29	0.12 ± 0.02	a
            'NiII^*_2166': [2166.23, 0.26],#	2167.65	0.26 ± 0.02	a
            'FeI_2167': [2167.45, 0.05]	,#
            'NiII^*_2175': [2175.22, 0.07],#	2175.83	0.07 ± 0.01
            #'MnI_2185': [2185.59, 0.51],#2186.96	0.51 ± 0.02
            'NiII^*_2217': [2217.2, 0.24],#2217.02	0.24 ± 0.01
            'NiII_2223': [2223.0, 0.07],#2224.98	0.07 ± 0.01
            'FeII_2249': [2249.88, 0.31],#	2250.02	0.31 ± 0.02
            'FeII_2260': [2260.78, 0.38],#2261.00	0.38 ± 0.02
            'NiII^*_2316': [2316.7, 0.19],#2316.36	0.19 ± 0.02
            'FeII^{*}_2328': [2328.11, 0.10],#	2327.65	0.10 ± 0.02
            'FeII^*_2333': [2333.52, 0.34],#	2333.30	0.34 ± 0.02
            'FeII_2344': [2344.21, 1.74],#2346.01	1.74 ± 0.02	a
            'FeII^*_2345': [2345.00, 0.05],#
            #'FeII\,^4F_{9/2}_2348': [2348.834, 0.26],#	2349.60	0.26 ± 0.02	a
            'FeII^{*}_2348': [2348.834, 0.26],#	2349.60	0.26 ± 0.02	a
            'FeII^{*}_2349': [2349.02, 0.26],#	2349.60	0.26 ± 0.02	a
            'FeII^*_2359': [2359.83, 0.28],#	2360.01	0.28 ± 0.01
            'FeII^*_2365': [2365.55, 0.18],#	2365.40	0.18 ± 0.01
            'FeII_2374': [2374.46, 1.00],#2374.46	1.00 ± 0.02
            'FeII^*_2381': [2381.49, 0.05],#			a
            'FeII_2382': [2382.77, 1.65],#2383.59	1.65 ± 0.02	a
            'FeII^*_2383': [2383.79, 0.05],#			a
            'FeII^*_2389': [2389.36, 0.18],#	2389.40	0.18 ± 0.02	a
            'FeII^*_23961': [2396.15, 0.05],#			a
            'FeII^*_23963': [2396.36, 0.71],#	2396.50	0.71 ± 0.02	a
            'FeII^*_2399': [2399.98, 0.05],#			a
            'FeII^*_2405': [2405.16, 0.40],#	2406.04	0.40 ± 0.02	a
            'FeII^*_2407': [2407.39, 0.05],#			a
            'FeII^*_24112': [2411.25, 0.05],#			a
            'FeII^*_24118': [2411.80, 0.51],#	2411.68	0.51 ± 0.03	a
            'FeII^*_2414': [2414.05, 0.05],#			a
            'MnII_2576': [2576.88, 0.45],#2577.04	0.45 ± 0.02
            'FeII_2586': [2586.65, 1.33],#2586.49	1.33 ± 0.02	a
            'MnII_2594': [2594.50, 0.45],#2594.36	0.45 ± 0.02
            'FeII^*_2599': [2599.15, 0.05],#	2600.23	1.85 ± 0.03	a
            'FeII_2600': [2600.17, 1.85],#			a
            'MnII_2606': [2606.46, 0.56],#2607.53	0.56 ± 0.01	a
            'FeII^*_2607': [2607.87, 0.05],#			a
            'FeII^*_2612': [2612.65, 0.51],#	2613.11	0.51 ± 0.01	a
            'FeII^*_2614': [2614.61, 0.05],#
            'FeII^*_2618': [2618.40, 0.06],#	2618.51	0.06 ± 0.01	a
            'FeII^*_2621': [2621.19, 0.05],#			a
            'FeII^*_2622': [2622.45, 0.05],#			a
            'FeII^*_2626': [2626.45, 0.05],#	2629.82	0.90 ± 0.02	a
            'FeII^*_2629': [2629.08, 0.90],#			a
            'FeII^*_2631': [2631.83, 0.05],#			a
            'FeII^*_2632': [2632.11, 0.05],#			a
            'FeII^*_2740': [2740.4, 0.07],#739.45	0.07 ± 0.01	b
            'FeII^*_2747': [2747.9, 0.16],#749.50	0.16 ± 0.01	b
            'FeII^*_2756': [2756.28, 0.08],#	2756.50	0.08 ± 0.01	b
            'MgII_2796': [2796.35, 1.71],#2796.21	1.71 ± 0.02
            'MgII_2803': [2803.53, 1.47],#2803.50	1.47 ± 0.02
            'MgI_2852': [2852.96, 0.78],#	2852.97	0.78 ± 0.01
            'TiII_3073': [3073.88, 0.08],#3076.01	0.08 ± 0.01
            'CaII_3934': [3934.78, 0.76],#3933.97	0.76 ± 0.02
            'CaII_3969': [3969.59, 0.66],#3969.98	0.66 ± 0.02
            'CaI_4227': [4227.92, 0.11],#	4226.93	0.11 ± 0.02
            'MgH_5209': [5209.45, 0.09],
            'NaID_5890' : [5891.63, 0.0],
            'NaID_5896' : [5897.534, 0.0],           
            'DIB_4428': [4430.1, 0.09],
            'DIB_5705': [5706.7, 0.01],
            'DIB_5781': [5782.2, 0.06],
            'DIB_5797': [5798.7, 0.01],
            'DIB_6284': [6286.0, 0.06],
            'DIB_6614': [6615.5, 0.02]}#5210.75	0.09 ± 0.02	b
def sortabsll():
    absdic = absll()
    sortabs = sorted(absdic.iteritems(), key=operator.itemgetter(1))
    return sortabs
    
def sortemll():
    absdic = emll()
    sortem = sorted(absdic.iteritems(), key=operator.itemgetter(1))
    return sortem

def highabswin():
    return [[5600, 5750], [6850, 6930], [7180, 7300], [7550, 7700], [8150, 8330], 
            [8950, 9200], [9270, 9800], [10950, 11500],
            [13300, 14800], [17800, 19600], [19900, 20200], [24100, 26000]]

def highabswin2():
    return [[7590, 7650], [9300, 9400], [10610, 11500],
            [13300, 14800], [17800, 19600], [19900, 20200], [24000, 26000]]

