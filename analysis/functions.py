# -*- coding: utf-8 -*-

import os
import subprocess
import numpy

from pylab import median, mgrid, exp, pi, array
from scipy import signal
from ..utils.astro import isnumber, Avlaws

C1 = 2.99792458E8


################################################################################


def skylines(intensity = 10, filen = '~/tools/etc/sky_lines.dat'):
    """ Returns OH skylines lines above a given intensity, file from  
    http://www.eso.org/sci/facilities/paranal/instruments/isaac/tools/oh/index.html 
    from the ISAAC OH atlas and in vacuum wavelengths, averages skylines closer than
    0.2 AA""" 
    # (9849)
    filen = os.path.expanduser(filen)
    fin = open(filen, 'r')
    lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
    fin.close()  
    sky, wlprev, intprev = [], 0, 0
    for line in lines:
        if line != []:
            if isnumber(line[0]):
                if float(line[-1]) > intensity:
                    wlline = float(line[0])
                    intens = float(line[1])
                    if (wlline - wlprev) < 0.2: 
                        sky.pop()
                        sky.append((wlline*intens + wlprev*intprev)/(intens+intprev))
                    else:
                        sky.append(wlline)
                wlprev, intprev = float(line[0]), float(line[1])
    return sky  


################################################################################


def transNIR(intensity = -0.2, filen = '~/tools/etc/cptrans_zm_43_15.dat'):
    """ Returns absorption lines above a given intensity, file from  
    http://www.gemini.edu/?q=node/10789 and in vacuum wavelengths""" 
    # (9849)
    filen = os.path.expanduser(filen)
    fin = open(filen, 'r')
    lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
    fin.close()  
    absor = []
    for line in lines:
        if line != []:
            if isnumber(line[0]):
                if float(line[0])*1E4 > 9849 and float(line[0])*1E4 < 25000:
                    if float(line[-1])-1 < intensity:
                        absor.append(float(line[0])*1E4)
    return absor


################################################################################  


def tell(intensity = -0.2, filen = '~/tools/etc/tel_lines_uves.dat'):
    """ Returns telluric absorption lines from UVES above a given intensity, 
    file found somewhere on the web, and apparently the wavelengths are in air"""
    filen = os.path.expanduser(filen)
    fin = open(filen, 'r')
    lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
    fin.close()
    tell = []
    for line in lines:
        if line != []:
            if isnumber(line[0]):
                if float(line[-2]) < intensity:
                    tell.append(float(line[3]))
    return tell


################################################################################


def ccmred(wl, ebv, rv = 3.08):
    """ Returns correction factors for wavelengths given in AA and E_B-V based on 
    Cardelli Redenning law and R_V = 3.08 """

    corr = []
    x = 10000./wl  #Inverse micron
    for inwl in x:
        if 0.3 < inwl <= 1.1:
            a =  0.574 * inwl**1.61
            b = -0.527 * inwl**1.61
        elif 1.1 < inwl <= 3.3:
            y = inwl - 1.82
            a = 1 + 0.104*y - 0.609*y**2 + 0.701*y**3 + 1.137*y**4\
                -1.718*y**5 - 0.827*y**6 + 1.647*y**7 - 0.505*y**8
            b = 0 + 1.952*y + 2.908*y**2 - 3.989*y**3 - 7.985*y**4\
              + 11.102*y**5 + 5.491*y**6 - 10.805*y**7 + 3.347*y**8
        elif 3.3 < inwl <= 5.9:
            fa, fb, y = 0, 0, inwl
            a =  1.752 - 0.316*y - (0.104 / ((y-4.67)**2 + 0.341)) + fa
            b = -3.090 + 1.825*y + (1.206 / ((y-4.62)**2 + 0.263)) + fb
        alambda = rv * ebv * (a + b/rv)
        corr.append(10**(0.4 * alambda))
    return array(corr)


################################################################################


def gauss_kern(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


################################################################################
def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='same')
    return(improc)


################################################################################


def smooth(x, window_len=11, window='flat', rms=0):
    if window == 'median':
        movmed, movrms = [], []
        for i in range(len(x)):
            low = max(0, i-window_len/2)
            high = min(len(x), i+window_len/2)
            movmed.append(median(x[low:high]))
            movrms.append(numpy.std(x[low:high]))
        if rms == 0:
            return array(movmed)
        else:
            return array(movmed), array(movrms)

    else:
       if x.ndim != 1:
           raise ValueError, "smooth only accepts 1 dimension arrays."
       if x.size < window_len:
           raise ValueError, "Input vector needs to be bigger than window size."
       if window_len<3:
           return x
       if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
           raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
       s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
       #print(len(s))
       if window == 'flat': #moving average
           w=numpy.ones(window_len,'d')
       else:
           w=eval('numpy.'+window+'(window_len)')
       y = numpy.convolve(w/w.sum(),s,mode='same')
       return array(y[window_len:-window_len+1])


################################################################################


def dlafunc(wl, nh):
    lya_rest, lyb_rest = 1215.67E-10, 1025.7222E-10
    nua_rest, nub_rest = C1/lya_rest, C1/lyb_rest
    nh = nh*1E4
    # Oscillator strength Ly_alpha
    f_a = 0.4162
    # Oscillator strength Ly_beta
    f_b = 0.0791
    
    gu, gl = 3., 1.
    gub, glb = 3., 1.

    delta_cla = 1.503E9
    delta_clb = 1.503E9*lya_rest**2/lyb_rest**2
    
    
    # Delta a Lyman alpha decay rate 6.26E8
    delta_a = 3 * (gl/gu) * f_a * delta_cla
    # Delta beta Lyman alpha decay rate 1.6725e+08
    delta_b = 3 * (glb/gub) * f_b * delta_clb

    const = 3 * lya_rest**2 * delta_a**2 / 8./ pi
    constb = 3 * lyb_rest**2 * delta_b**2 / 8./ pi

    nu = C1 / wl
    
    sigma_a1 = const * ((nu/nua_rest)**4/\
            (4*pi**2*(nu-nua_rest)**2 + delta_a**2/4.*(nu/nua_rest)**6))
            
    sigma_a2 = constb * ((nu/nub_rest)**4/\
            (4*pi**2*(nu-nub_rest)**2 + delta_b**2/4.*(nu/nub_rest)**6))    
    
    tau_a1 = nh*sigma_a1#*sigma_a2
    tau_a2 = nh*sigma_a2#*sigma_a2

    dla = exp(-1*tau_a1)*exp(-1*tau_a2)
    return dla


################################################################################    


def redlaw(y, law):
    y = y / 10. # y is wavelength in nm, but spectra always AA
    ext = Avlaws(y, law)
    return ext


###########################################


def addzero(val, n):
    val = float(val)
    if val < 10:
        if n == 1: val = '0%.0f' %(val)
        if n == 2: val = '0%.2f' %(val)
        if n == 3: val = '0%.3f' %(val)
    else:
        if n == 1: val = '%.0f' %val
        if n == 2: val = '%.2f' %val
        if n == 3: val = '%.3f' %val
    return val
    

###########################################


def deg2sexa(ra, dec):
    hours = int(ra/15)
    minu = int((ra/15.-hours)*60)
    seco = float((((ra/15.-hours)*60)-minu)*60)
    retra = '%s:%s:%s' %(addzero(hours,1), addzero(minu,1), addzero(seco,3))

    degree = int(dec)
    minutes = int((dec-degree) * 60)
    seconds = float((((dec-degree) * 60) - minutes) * 60)
    if 60.01 >= abs(seconds) >= 59.99:
        if seconds < 0:
            minutes -= 1
        else:
            seconds += 1
        seconds = 0 
        
    if degree < 0:
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
        if float(dec[0]) < 0:
            retdec = float(dec[0])-float(dec[1])/60.-float(dec[2])/3600.
        else:
            retdec = float(dec[0])+float(dec[1])/60.+float(dec[2])/3600.
    return round(float(retra),6), round(float(retdec), 6)
    


###########################################


def checkExec(execlist):
    """ Checks for executables from a list using which. Returns None if not
    available"""
    
    for execcall in execlist:
        proc = subprocess.Popen(['which', execcall], stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE)
        if proc.stdout.read() != None:
            return execcall
    return None