import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate, constants

from .functions import dlafunc, smooth, redlaw
from ..utils.astro import absll, emll, binspec, ergJy, Jyerg, isnumber, abflux
from ..utils.fitter import plfit, plfmfit, pl, plfm

abslist, emllist = absll(), emll()
linelist = dict(emllist.items() + abslist.items())


def setCont(s2d, arm, smoothn = 20, meth = 'median', datval = None,
           intpl = 'pf', order = 7, s1 = 1, sig = 2.5, absl = 1):
    print '\tRemoving absorption lines for continuum fit'
    wltmp = np.array(s2d.wave[arm])
    # Excluding Lya
    wlsela = wltmp > 912 * (s2d.redshift + 1)
    wlsel01 = wltmp > 1215.6 * (s2d.redshift + 1) * 1.05
    wlsel02 = wltmp < 1215.6 * (s2d.redshift + 1) * 0.945
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
        valsel = sorted(s2d.oneddata[arm])[int(datval*len(wltmp))]
        datsel = s2d.oneddata[arm] > valsel
        wlsel *= datsel
    
    
    # Excluing strong telluric lines
    if arm in ['vis', 'nir']:        
        telsel = s2d.skytel[arm] > 0.99
    else:
        telsel = s2d.skytel[arm] > 0.
        
    wlsel *= telsel
    print '\t\tExcluding %s number of points due to tellurics' %len(telsel[telsel==False])
    # Excluing strong absorption lines
    if absl in [1, '1', 'Y', 'y', 'YES', True]:
        for absline in linelist:
            lstrength = linelist[absline][1]
            if lstrength > 0.1:
                wlline = linelist[absline][0]*(1+s2d.redshift)
                wlsell = (wlline-2 < wltmp) ^ (wltmp < wlline + 2)
                wlsela *= wlsell
                wlsel *= wlsell
            for redi in s2d.intsys: 
                for intline in s2d.intsys[redi]:
                    wlline = linelist[intline][0] * (1 + redi)
                    wlsell = (wlline - 10 < wltmp) ^ (wltmp < wlline + 10)
                    wlsela *= wlsell
                    wlsel *= wlsell
        print '\t\tExcluding %s number of points due to absorption lines' \
                %len(wlsela[wlsela==False])

    s2d.cleanwav[arm] = s2d.wave[arm][wlsel]
    s2d.cleandat[arm] = s2d.oneddata[arm][wlsel]
    s2d.cleanerr[arm] = s2d.onederro[arm][wlsel]
    print '\t\tDownsample the spectrum for continuum fit (factor %s)' %smoothn
    wltmp, dattmp, errtmp = binspec(s2d.wave[arm][wlsel],
                                   s2d.oneddata[arm][wlsel]*s2d.mult,
                                   s2d.onederro[arm][wlsel]*s2d.mult,
                                   wl = smoothn, meth = meth)
    for j in range(30):
        if intpl == 'model':
            dattmppJy = ergJy(dattmp/s2d.mult, wltmp)
            errtmpJy = errtmp/dattmp*dattmppJy
            params = plfit(wltmp, dattmppJy, err = errtmpJy,
                    params = [1, 1, 0, s2d.redshift],
                    fixed = [False, False, False, True])
            contfit = Jyerg(pl(wltmp, params[0][0], params[0][1], params[0][2], 
                         params[0][3]), wltmp)
        elif intpl == 'fmmodel':
            dattmppJy = ergJy(dattmp/s2d.mult, wltmp)
            errtmpJy = errtmp/dattmp*dattmppJy
            params = plfmfit(wltmp, dattmppJy, err = 1.0*errtmpJy,
                     params = [10., 0.78, 0.15, s2d.redshift, 2.93, 
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
        s2d.cont[arm] = pl(s2d.wave[arm], params[0][0], 
                 params[0][1], params[0][2], params[0][3])/s2d.mult
    elif intpl == 'spline':
        s2d.cont[arm] = tck(s2d.wave[arm]) / s2d.mult
    elif intpl == 'pf':                
        s2d.cont[arm] = np.polyval(coeff, s2d.wave[arm])/s2d.mult 
    elif intpl == 'fmmodel':
        s2d.cont[arm] = plfm(s2d.wave[arm], params[0][0], 
                params[0][1], params[0][2], params[0][3],
                params[0][4], params[0][5], params[0][6],
                params[0][7], params[0][8])/s2d.mult            
    s2d.woabs[arm] = np.array(s2d.cont[arm])
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))        
    
    ax.set_yscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.plot(s2d.wave[arm], s2d.oneddata[arm]*s2d.mult, 'b')
    ax.plot(s2d.wave[arm], s2d.cont[arm]*s2d.mult, 'r')
    ax.plot(wltmp, dattmp, 'o', color = 'yellow', ms = 1.9)
    #ax.set_xscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xlim(min(wltmp), max(wltmp))  
    ax.set_ylim(min(dattmp)*0.9, max(dattmp)*1.5)
    fig.savefig('%s_continuum_%s.pdf' %(s2d.object, arm))

################################################################################  
      
def fitCont(s2d, arm, wl1 = 4500, wl2 = 5800, norm = 0, av = 0, beta = 0,
            red = 'smc', binx = 1):
    print('\tFitting continuum with afterglow model')
    x1, x2 = s2d.wltopix(arm, wl1), s2d.wltopix(arm, wl2)
    fitx = s2d.cleanwav[arm][x1:x2+1]
    fity = s2d.cleandat[arm][x1:x2+1]
    fite = s2d.cleanerr[arm][x1:x2+1]
    fitfld = ergJy(fity, fitx)
    fitfle = fite/fity*fitfld
    binfitw, binfitd, binfite = binspec(fitx, fitfld, fitfle, wl = binx)
    binfite *= 3.5
    fixed0, fixed1, fixed2 = False, False, False
    if norm !=0: fixed0 = True
    if av != 0: fixed2 = True
    if beta != 0: fixed1 = True
    params = plfit(binfitw, binfitd, err = binfite, red = red,
                   params = [norm, beta, av, s2d.redshift],
                   fixed = [fixed0, fixed1, fixed2, True])
    s2d.cont[arm] = Jyerg(pl(s2d.wave[arm], params[0][0], 
                    params[0][1], params[0][2], params[0][3], red = red),
                    s2d.wave[arm])
                        
    fig = plt.figure(figsize = (8,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
    
    ax.plot(binfitw, params[1], '-')
    ax.plot(binfitw, binfitd, 'o', ms = 4, mec='black', color='grey')
    fig.savefig('%s_%s_res.pdf' %(s2d.object, arm))

################################################################################  

def dlaAbs(s2d, arm, nh, nherr = 0, z = ''):
    if z == '':
        z = s2d.redshift
    print '\tAdding DLA with log NH = %.2f +/- %.2f at z=%.3f' %(nh, nherr,z)
    nh, nhp, nhm = 10**nh, 10**(nh+nherr), 10**(nh-nherr) 
    s2d.nh = nh
    wl = s2d.wave[arm]/(1+z) * 1E-10
    dlam = dlafunc(wl, nh)
    if nherr != 0:
        dlamp = dlafunc(wl, nhp) 
        dlamm = dlafunc(wl, nhm)
    x1 = s2d.wltopix(arm, 1180 * (z + 1))
    x2 = s2d.wltopix(arm, 1260 * (z + 1))
    chim = sum(((dlam/s2d.mult - s2d.oneddata[arm])**2 \
        /(s2d.onederro[arm])**2)[x1:x2])
    if s2d.oneddla[arm] == '': 
        if nherr != 0: s2d.oneddla[arm] = [dlam, dlamp, dlamm]
        else: s2d.oneddla[arm] = [dlam]
    else:
        if nherr != 0: s2d.oneddla[arm] = [s2d.oneddla[arm][0]*dlam, 
                s2d.oneddla[arm][1]*dlamp, s2d.oneddla[arm][2]*dlamm]
        else:
            s2d.oneddla[arm] = [s2d.oneddla[arm]*dlam]
    return chim


################################################################################  
    
def stackLines(s2d, lines, nsmooth=10):
    c = constants.c/1E3
    normspecs, normerrs = [], []
#        normvels = []
    normvels = np.arange(-2000, 2000, 20)
    for line in lines:
        if abslist.has_key(line):
            # Cutout -25 AA to 25 AA
            dx1, dx2 = 400, 300
            redwl = abslist[line][0]*(1+s2d.redshift)

            if 3100 < redwl < 5500: arm = 'uvb'
            elif 5500 < redwl < 10000: arm = 'vis'
            elif 10000 <  redwl < 25000: arm = 'nir'
            else:
                print 'Line %s not in wavelength response' %line
            
            pix1, pix2 = s2d.wltopix(arm, redwl-dx1), s2d.wltopix(arm, redwl+dx1)
            x = (s2d.wave[arm][pix1:pix2]-redwl)/redwl * c
            y = s2d.mult * s2d.oneddata[arm][pix1:pix2]
            yerr = s2d.mult * s2d.onederro[arm][pix1:pix2]

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
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))            
    ax.errorbar(normvels, medspec, mederr,
                drawstyle = 'steps-mid', capsize = 0, color = 'black')
    ax.plot(normvels, smooth(medspec, window_len=nsmooth, window='hanning'), 
    'red', lw=1.5)
    ax.set_xlim(-600, 600)
    ax.set_ylim(-0.3, 2.8)
    ax.set_xlabel(r'$\rm{Velocity\,(km\,s^{-1})}$')
    ax.set_ylabel(r'$\rm{Normalized\,flux}$')

    fig.savefig('%s_lines.pdf' %(s2d.object))

################################################################################
            
def setMod(s2d, arms, norm, beta = 0.5, av = 0.15, red = 'smc'):
    ''' Defines a physical afterglow model given with a given norm at 6250 AA 
    in muJy (unreddened, A_V = 0), beta, av and reddening law (default smc)'''
    for arm in arms:
        wls = s2d.wave[arm]
        law = redlaw(wls/(1+s2d.redshift), red)
        modmuJy = norm * (wls/6250.)**(beta) * 10**(-0.4*av*law)
        s2d.model[arm] = Jyerg(modmuJy, wls)
            
################################################################################
            
def setAscMod(s2d, arms, mfile, s = 3.0, order = 2):
    ''' Read in ASCII model (Lephare host output, which means two columns with
    WL (\AA) and AB mag) '''
    lines = [line.strip() for line in open(mfile)]
    s2d.modwl, s2d.modfl = [], []
    for line in lines:
        if line != '':
            if len(line.split()) == 2 \
            and isnumber(line.split()[0]) and isnumber(line.split()[1]):
                s2d.modwl.append(float(line.split()[0]))
                s2d.modfl.append(abflux(float(line.split()[1])))
    s2d.modwl, s2d.modfl = np.array(s2d.modwl), np.array(s2d.modfl)
    modergs = Jyerg(s2d.modfl, s2d.modwl)
    tck = interpolate.UnivariateSpline(s2d.modwl, modergs, s = s, k = order)
    for arm in arms:
        s2d.model[arm] = tck(s2d.wave[arm])
    print ('\t ASCII model set sucessfully')
        
################################################################################ 
        
def scaleMod(s2d, arms, p = ''):
    for arm in arms:
        s2d.match[arm] = s2d.model[arm]/s2d.cont[arm]
        print '\tScaling %s spectrum to afterglow model' %arm
        if len(s2d.data[arm]) != 0:
            atmp = s2d.data[arm].transpose() * s2d.match[arm]
            btmp = s2d.erro[arm].transpose() * s2d.match[arm]
            s2d.data[arm] = atmp.transpose()
            s2d.erro[arm] = btmp.transpose()
        if len(s2d.oneddata[arm]) != 0:    
            s2d.oneddata[arm] *= s2d.match[arm] 
            s2d.onederro[arm] *= s2d.match[arm] 
            s2d.cont[arm] = np.array(s2d.model[arm])
        s2d.output[arm] = s2d.output[arm]+'_scale'
        if p != '':
            fig = plt.figure(figsize = (10,5))
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
            ax.plot(s2d.wave[arm], s2d.match[arm])
            ax.set_xlim(s2d.wlrange[arm][0], s2d.wlrange[arm][1])        
            ax.set_ylim(0.8, 4)        
            fig.savefig('%s_slitloss_%s.pdf' %(s2d.object, arm))
            
################################################################################ 
       
