import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from matplotlib.backends.backend_pdf import PdfPages

from tom.astro import airtovac, emll, LDMP
from .fitter import onedmoffatfit, onedgaussfit, onedmoffat
from .functions import smooth
emllist = emll()


def makeProf(s2d, arm, lim1 = 3.4E3, lim2 = 25E3, chop = 1, order = 0,
             fwhm = '', mid = '', vac = 0, verbose = 0, orderfwhm = None,
             line = '', meth = 'weight', profile = None,
             pmin=0, pmax=-1):
    
    if profile == None:
        profile = s2d.profile[arm]
    else:
        s2d.profile[arm] = profile
        
    if orderfwhm == None:
        orderfwhm = max(0, order-1)

    print '\tCreating spatial profile for arm %s, %s profile' %(arm, profile)
    tracemid, tracesig, tracemide, tracesige, tracepic, tracewl = 6 * [np.array([])]
    tracebet, tracebete = [], []
    Pics = np.arange(len(s2d.wave[arm]))
    if line != '':
        vac = 1
        if line.upper() == 'OII':
            lim1 = emllist['[OII](3728)'][0]*(1+s2d.redshift)-8
            lim2 = emllist['[OII](3728)'][0]*(1+s2d.redshift)+8
        if line.upper()  == 'OIII':
            lim1 = emllist['[OIII](5007)'][0]*(1+s2d.redshift)-5
            lim2 = emllist['[OIII](5007)'][0]*(1+s2d.redshift)+5
        elif line in ('Ha', 'Halpha', 'ha'):
            lim1 = emllist['Halpha'][0]*(1+s2d.redshift)-5
            lim2 = emllist['Halpha'][0]*(1+s2d.redshift)+5
        elif line in ('Hb', 'Hbeta', 'hb'):
            lim1 = emllist['Hbeta'][0]*(1+s2d.redshift)-5
            lim2 = emllist['Hbeta'][0]*(1+s2d.redshift)+5
    
    if vac == 0:
        print '\t\tConverting limits to vacuum'
        lim1 = airtovac(lim1)
        lim2 = airtovac(lim2)
        
    wlsel = (lim1 < s2d.wave[arm]) * (s2d.wave[arm] < lim2) 
    sData = np.array_split(s2d.data[arm][wlsel], chop)
    sWave = np.array_split(s2d.wave[arm][wlsel], chop)
    sErro = np.array_split(s2d.erro[arm][wlsel], chop)
    sPics = np.array_split(Pics[wlsel], chop)
    
    pp = PdfPages('%s_profiles_%s.pdf' %(s2d.object, arm))

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
            uErro = np.array(sErro[i]*s2d.mult).transpose()
            uData = np.array(sData[i]*s2d.mult).transpose()
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
            sprof[:s2d.backrange[arm][0]] = 0
            sprof[-s2d.backrange[arm][1]:] = 0
            sprof[:pmin] = 0
            sprof[pmax:] = 0

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
                            %(params[0][2] + s2d.datarange[arm][0], params[2][2])
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
        s2d.trace[arm] = [a, b, c]
        nplot, figsize = 2, (9, 6)
    else:
        s2d.trace[arm] = [a, b]
        nplot, figsize = 2, (9, 6)
        
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(hspace=0.05, wspace=0.0)
    fig.subplots_adjust(bottom=0.12, top=0.98, left=0.14, right=0.89)
    ax1 = fig.add_subplot(nplot, 1, 1)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))        
    
    ax1.xaxis.tick_top()
    ax1.errorbar(tracewl, tracemid + s2d.datarange[arm][0],
                 yerr= tracemide, fmt = 'o', capsize = 0)
    ax1.plot(s2d.wave[arm], 
             np.polyval(s2d.trace[arm][0], Pics) + s2d.datarange[arm][0] )
    
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
        fitfwhm = np.polyval(s2d.trace[arm][1], Pics) * \
        2 * (2**(1./np.polyval(s2d.trace[arm][2], Pics)) - 1 )**0.5
        
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
        fitfwhm = np.polyval(s2d.trace[arm][1], Pics) * 2.3538

    ax2.errorbar(tracewl, fwhm, yerr = fwhmerr, fmt = 'o', capsize = 0)
    ax2.errorbar(s2d.wave[arm], fitfwhm)
    
    ax2.set_ylabel(r'$\rm{FWHM\,of\,trace\,(px)}$')
    ax2.set_xlabel(r'$\rm{Observed\, wavelength\, (\AA)}$')

    if s2d.inst == 'FORS2':
        pxsc = float(s2d.head[arm]['HIERARCH ESO INS PIXSCALE'])\
            * float(s2d.head[arm]['HIERARCH ESO DET WIN1 BINY'])
    else:    
        pxsc = float(s2d.head[arm]['CD2_2'])

    ax3.set_ylim([ax2.get_ylim()[0]*pxsc, 
                  ax2.get_ylim()[1]*pxsc])
    ax3.set_ylabel(r'$\rm{FWHM\,(arcsec)}$')
    ax3.yaxis.tick_right()
    ax2.yaxis.tick_left()
    plt.setp(ax3.get_xticklabels(), visible=False)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([min(s2d.wave[arm]), max(s2d.wave[arm])])
    pp.savefig(fig)
    pp.close()
    
    
def extr1d(s2d, arms, opt = 1, n = 4, sig = '', errsc = 1., offset = 0):
    """Parameter opt = 1, use the derived profile to do opimal extraction
                 opt = 0, use an aperture of size n pixels
      Parameter n: everything out of +/- n/2.3548 arcsec is not extracted 
          for optimal size of the aperture
      Parameter sig: fix sigma (in pixels) of the profile"""
      
    for arm in arms:
        if s2d.inst == 'FORS2':
            pxscale = float(s2d.head[arm]['HIERARCH ESO INS PIXSCALE'])\
                * float(s2d.head[arm]['HIERARCH ESO DET WIN1 BINY'])
        else:    
            pxscale = float(s2d.head[arm]['CD2_2'])
        
        weightdata, weighterro, skyrms, skysum, prof = [], [], [], [], []
        if s2d.trace[arm] == '':
            print '\tERROR: Trace necessary for extraction for %s arm' %arm
            raise SystemExit
        if opt == 0:
            print '\tFixed aperture of %i pixels (%.2f arcsec)' \
                    %(n, n*pxscale) 
        else:
            print '\tOptimal extraction with profile/trace parameters'               
        s2d.output[arm] = s2d.output[arm]+'_1d'
        bins = np.arange(len(s2d.data[arm][0]))
        y0, y1 = 0, len(bins) - 1
        dy0, dy1 = s2d.backrange[arm][0], s2d.backrange[arm][1]

        for i in range(len(s2d.data[arm])):
            mid = np.polyval(s2d.trace[arm][0], i) + offset

            if opt == 0:
                sky = np.append(s2d.data[arm][i][y0:y0+dy0],
                             s2d.data[arm][i][y1-dy1:y1])
                skye = np.append(s2d.erro[arm][i][y0:y0+dy0],
                             s2d.erro[arm][i][y1-dy1:y1])
            else:
                if sig == '':
                    sigma = np.polyval(s2d.trace[arm][1], i)
                else:
                    sigma = float(sig)                    
                
                if s2d.profile[arm] == 'moffat':
                    alpha = np.polyval(s2d.trace[arm][1], i)
                    beta = np.polyval(s2d.trace[arm][2], i)
                    sigma = alpha * 2 * (2**(1./beta) - 1 )**0.5 / 2.3538
                    prof = onedmoffat(bins, 0, 1, mid, alpha, beta)
                else:
                    prof = mlab.normpdf( bins, mid, sigma )                    
                
                if (int(mid + 3*sigma) > y1) or (int(mid-3*sigma) < y0):
                    print '\t\tError in sky regions - check profile'
                    raise SystemExit
                
                sky = np.append(s2d.data[arm][i][y0 : int(mid-2*sigma)],
                            s2d.data[arm][i][int(mid + 2*sigma) : y1])
                skye = np.append(s2d.erro[arm][i][y0 : int(mid-2*sigma)],
                            s2d.erro[arm][i][int(mid + 2*sigma) : y1])
            skyrms.append( np.median(skye) )
            skysum.append( np.median(sky) )
            
            if n > 0 and opt == 0:
                prof = np.zeros(max(bins)+1) + 1./n
                prof[:int(round(mid-n/2))], prof[int(round(mid+n/2+1)):] = 0, 0
                if n % 2 == 0:
                    prof[int(round(mid-n/2))], prof[int(round(mid+n/2))] = 1./(2*n), 1./(2*n)
            else:
                lim = n / pxscale / 2.3548
                prof[:int(round(mid-lim))], prof[int(round(mid+lim)):] = 0, 0
            
            prof *= 1./sum(prof)
            weightd = sum((s2d.data[arm][i] * prof).transpose()) \
                   / sum(prof**2)
            weightdata.append(weightd)
            if len(s2d.erro[arm]) != 0:
                weighte = (sum((s2d.erro[arm][i]**2 * prof**2)).transpose() \
                   / sum(prof**2)**2)**0.5
                weighterro.append(weighte)
                
        s2d.skyrms[arm] = np.array(skyrms)
        s2d.oneddata[arm] = np.array(weightdata)
        s2d.onederro[arm] = np.array(weighterro)/errsc
        s2d.onedback[arm] = np.array(skysum)
        
        if len(s2d.ebvcorr[arm]) != 0 and '_ebv' not in s2d.output[arm]:
            print '\t\tCorrecting %s data for E_B-V = %.2f with R_V = %.2f' \
                %(arm, s2d.ebv, s2d.rv)
            if len(s2d.data[arm]) != 0:
                atmp = s2d.data[arm].transpose() * s2d.ebvcorr[arm]
                btmp = s2d.erro[arm].transpose() * s2d.ebvcorr[arm]
                s2d.data[arm] = atmp.transpose()
                s2d.erro[arm] = btmp.transpose()
            if len(s2d.oneddata[arm]) != 0:
                s2d.oneddata[arm] *= s2d.ebvcorr[arm]
                s2d.onederro[arm] *= s2d.ebvcorr[arm]
            s2d.output[arm] = s2d.output[arm]+'_ebv'
        for a in (s2d.oneddata[arm], s2d.onederro[arm]):
            a[abs(a) < 1E-22] = 1E-22
        
        if s2d.redshift != 0:
            # Luminosity distance
            ld = LDMP(s2d.redshift)*3.08568e24 
            s2d.lumspec[arm] = s2d.oneddata[arm] * 4 * np.pi * ld**2
            s2d.lumerr[arm] = s2d.onederro[arm] * 4 * np.pi * ld**2
            s2d.restwave[arm] = s2d.wave[arm]/(1+s2d.redshift)    

def write1d(s2d, arms, lim1=3.20E3, lim2=2.28E4, error = 1, lum=0):

    for arm in arms:
        f = open('%s.spec' %s2d.output[arm], 'w')
        f.write('#Object: %s\n' %s2d.object)
        f.write('#Fluxes in [10**-%s erg/cm**2/s/AA] \n' %(str(s2d.mult)[-2:]))
#            f.write('#Redshift: %.4f\n' %self.redshift)
        if 'helio' in s2d.output[arm].split('_') and 'vac' in s2d.output[arm].split('_'):
            f.write('#Wavelength is in vacuum and in a heliocentric reference\n')
        if 'ebv' in s2d.output[arm].split('_'):
            f.write('#Fluxes are corrected for Galactic foreground\n')
        for i in range(len(s2d.onederro[arm])):
            if lim1 < s2d.wave[arm][i] < lim2:
                if error == 1:
                    f.write('%.6f\t%.4e\t%.4e' \
                    %(s2d.wave[arm][i], s2d.oneddata[arm][i]*s2d.mult, \
                                    s2d.onederro[arm][i]*s2d.mult))
                    if len(s2d.cont[arm]) > 0:
                        f.write('\t%.4e' %( s2d.cont[arm][i]*s2d.mult))
                    f.write('\n')
                else:
                    f.write('%.6f\t%.4e\n' \
                    %(s2d.wave[arm][i], s2d.oneddata[arm][i]*s2d.mult))           
        f.close()
        
        if lum != 0:
          if s2d.lumspec[arm] != '':
            f = open('%s_lum.spec' %s2d.output[arm], 'w')
            f.write('#Object: %s\n' %s2d.object)
            f.write('#Redshift: %.4f\n' %s2d.redshift)
            if 'helio' in s2d.output[arm].split('_') and 'vac' in s2d.output[arm].split('_'):
                f.write('#Wavelength is at rest, in vacuum and heliocentric\n')
            if 'ebv' in s2d.output[arm].split('_'):
                f.write('#Luminosities are corrected for Galactic foreground and slitloss\n')
            for i in range(len(s2d.restwave[arm])):
                f.write('%.6f\t%.4e\t%.4e\t%.3f\t%.1f\n' \
                %(s2d.restwave[arm][i], s2d.lumspec[arm][i]*(1+s2d.redshift), 
                  s2d.lumerr[arm][i]*(1+s2d.redshift), 
                  s2d.skytel[arm][i], abs(s2d.skyrad[arm][i])))
            f.close()                
 
def smooth1d(s2d, arm, smoothn=7, filt='median'):
    s2d.soneddata[arm] = np.array(smooth(s2d.oneddata[arm], smoothn, filt))
    
               
def writeVP(s2d, arm, lya = 0, scl = 1.0):
    if lya != 0:
        fname = s2d.output[arm]+'_vpfit_lya.txt'
        y = s2d.woabs[arm]*s2d.mult*s2d.match[arm]
        yerr = s2d.onederro[arm]*s2d.mult
    else:
        fname = s2d.output[arm]+'_vpfit.txt' 
        y = s2d.oneddata[arm]*s2d.mult
        yerr = s2d.onederro[arm]*s2d.mult            
    f = open(fname, 'w')
    f.write('RESVEL %.2f\n' %s2d.reso[arm])
    for wl, fl, er, cont in zip(s2d.wave[arm], y, yerr, 
                                s2d.cont[arm]*s2d.mult):
        f.write('%.6f\t%.4e\t%.4e\t%.4e\n' %(wl, fl, er/scl, cont))
    f.close()
    return fname