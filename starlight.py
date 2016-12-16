# -*- coding: utf-8 -*-

""" Spectrum class for running starlight on spectra. Particularly for 
 long-slit spectra """

import os
import numpy as np
import scipy as sp

import shutil
import time
import platform
import matplotlib.pyplot as plt

from .astro import binspec
from matplotlib.backends.backend_pdf import PdfPages

SL_BASE = os.path.join(os.path.dirname(__file__), "etc/Base.BC03.15lh")
SL_CONFIG = os.path.join(os.path.dirname(__file__), "etc/XS_SLv01.config")
SL_CONFIG = os.path.join(os.path.dirname(__file__), "etc/XS_SLv01_15lh.config")

SL_MASK = os.path.join(os.path.dirname(__file__), "etc/Masks.EmLines.SDSS.gm")
SL_BASES = os.path.join(os.path.dirname(__file__), "etc/bases")

if platform.platform().startswith('Linux'):
    SL_EXE = os.path.join(os.path.dirname(__file__), "etc/starlight")
else:
    SL_EXE = os.path.join(os.path.dirname(__file__), "etc/starlight_mac")


def asciiout(s2d, wl, spec, err=[], resample=1, name='', div=-17,
             frame = 'rest', fmt='spec'):

    """ Write the given spectrum into a ascii file. 
    Returns name of ascii file, writes ascii file.

    Parameters
    ----------
    wl : np.array
        wavelength array
    spec : np.array
        spectrum array
    err : np.array
        possible error array (default None)
    resample : int
        wavelength step in AA to resample
    name : str
        Name to use in fits file name
    """
    asciiout = '%s_%s.%s' %(s2d.inst, name, fmt)

    if s2d.redshift != None and frame == 'rest':
        print ('\tMoving to restframe for starlight fit')
        wls = wl / (1+s2d.redshift)
        divspec = spec * (1+s2d.redshift) / 10**div
        if len(err) == len(spec):
            diverr = err * (1+s2d.redshift) / 10**div
    else:
        divspec = spec / 10**div
        if len(err) == len(spec):
            diverr = err / 10**div
            
    if resample not in [False, 0, 'False']:
        
        outwls = np.arange(int(wls[1]), int(wls[-2]), resample)
        s = sp.interpolate.interp1d(wls, divspec, fill_value='extrapolate')
        outspec = s(outwls)
        if len(err) == len(spec):
            t = sp.interpolate.interp1d(wls, diverr, fill_value='extrapolate')
            outerr = t(outwls)
        fmt = '%.1f %.3f %.3f 0\n'
    else:
        outwls, outspec, outerr = \
            np.copy(wl), np.copy(spec) / 10**div, np.copy(err) / 10**div
        fmt = '%.3f %.3e %.3e \n'
        
    f = open(asciiout, 'w')
    if fmt == 'spec':
        f.write('#Fluxes in [10**-%s erg/cm**2/s/AA] \n' %(-17+div))
        f.write('#Wavelength is in vacuum and in a heliocentric reference\n')
        if s2d.ebvGalCorr != 0:
            f.write('#Fluxes are corrected for Galactic foreground\n')

    for i in range(len(outwls)):
        if len(err) == len(spec):
            f.write(fmt %(outwls[i], outspec[i], outerr[i]))
        if len(err) != len(spec):
            f.write('%.2f %.3f\n' %(outwls[i], outspec[i]))            
    f.close()
#        logger.info('Writing ascii file took %.2f s' %(time.time() - t1))
    return asciiout   




class StarLight:
    """ StarLight class for fitting """

    def __init__(self, filen, inst, verbose=0, minwl=3700, maxwl=9400,
                 run=1, red='GD3'):
        self.specfile = filen
        self.minwl=minwl
        self.maxwl=maxwl
        root, ext = os.path.splitext(filen)
        self.output = root+'_sl_%i_%i_out%s' %(minwl, maxwl, ext)
        self.sllog = root+'_sl_log'+ext
        self.seed = np.random.randint(1E6, 9E6)
        self.cwd = os.getcwd()
        self.inst = inst
        self.red = red
        
        shutil.copy(SL_BASE, self.cwd)
        shutil.copy(SL_CONFIG, self.cwd)
        
        if not os.path.isdir(os.path.join(self.cwd, 'bases')):
            shutil.copytree(SL_BASES, os.path.join(self.cwd, 'bases'))
        
        if not os.path.isfile(SL_EXE):
            print ('ERROR: STARLIGHT executable not found')
            raise SystemExit
            
        if run == 1:
            self._makeGrid()
            self._runGrid()


    def _makeGrid(self, name='xshoot_grid.in'):

        headkey = ['[Number of fits to run]',
               '[base_dir]', '[obs_dir]', '[mask_dir]', '[out_dir]',
               '[seed]', '[llow_SN]', '[lupp_SN]', '[Olsyn_ini]',
               '[Olsyn_fin]', '[Odlsyn]', '[fscale_chi2]', '[FIT/FXK]', 
               '[IsErrSpecAvailable]', '[IsFlagSpecAvailable]']
        speckey = ['spectrum', 'config', 'bases', 'masks', 'red', 'v0_start',
                  'vd_start', 'output']
       
        header = {'[Number of fits to run]': '1',
               '[base_dir]': self.cwd+'/bases/',
               '[obs_dir]' :self.cwd+'/', 
               '[mask_dir]' : os.path.split(SL_MASK)[0]+'/', 
               '[out_dir]': self.cwd+'/',
               '[seed]': self.seed, 
               '[llow_SN]': 4500, 
               '[lupp_SN]': 4800, 
               '[Olsyn_ini]': self.minwl,
               '[Olsyn_fin]': self.maxwl, 
               '[Odlsyn]':1.0, 
               '[fscale_chi2]':1.0, 
               '[FIT/FXK]': 'FIT',
               '[IsErrSpecAvailable]':'1', 
               '[IsFlagSpecAvailable]':'1'}

        specline = {'spectrum': self.specfile, 
            'config': os.path.split(SL_CONFIG)[-1], 
            'bases': os.path.split(SL_BASE)[-1], 
            'masks': os.path.split(SL_MASK)[-1], 
            'red' : self.red, 
            'v0_start': 0,
            'vd_start': 150, 
            'output': self.output}
            
        f = open(name, 'w')
        for head in headkey:
            f.write('%s  %s\n' %(header[head], head))
        for spec in speckey:
            f.write('%s   ' %(specline[spec]))
        f.write('\n')
        self.grid = name
        
    def _runGrid(self, cleanup=True):
        t1 = time.time()
        slarg = [SL_EXE, '<', self.grid, '>', self.sllog]
        os.system(' '.join(slarg))
        # Cleanup
        if cleanup == True:
            shutil.rmtree('bases')
            os.remove(os.path.join(self.cwd, os.path.split(SL_BASE)[-1]))
            os.remove(os.path.join(self.cwd, os.path.split(SL_CONFIG)[-1]))
            os.remove(self.grid)
        return time.time()-t1

       
       
    def modOut(self, minwl, maxwl, plot=1, chop=8):
        
        starwl, starfit = np.array([]), np.array([])
        datawl, data, gas, stars = 4*[np.array([])]
        success, run, norm = 0, 0, 1

        try:
            f = open(self.output)
            output = f.readlines()
            f.close()
            os.remove(self.sllog)
            run = 1
        except IOError:
            pass
        
        if run == 1:
            for out in output:
              outsplit =   out.split()
              if outsplit[1:] == ['[fobs_norm', '(in', 'input', 'units)]']:
                  norm = float(outsplit[0])
                  success = 1
              if outsplit[1:] == ['Run', 'aborted:(']:
                  break
              if len(outsplit) == 4:
                try:  
                  outsplit = [float(a) for a in outsplit]  
                  if float(outsplit[0]) >= minwl and float(outsplit[0]) <= maxwl:
                      starfit = np.append(starfit, outsplit[2])
                      starwl = np.append(starwl, outsplit[0])
                      if outsplit[3] != -2:
                          data = np.append(data, outsplit[1])
                          gas = np.append(gas, outsplit[1]-outsplit[2] )
                          stars = np.append(stars, outsplit[2])
                          datawl = np.append(datawl, outsplit[0])                    
                except ValueError:
                    pass
            
              if len(outsplit) == 3:
                 if outsplit[1] == '[v0_min':
                    v0 = float(outsplit[0])
                 if outsplit[1] == '[vd_min':
                    vd = float(outsplit[0])       
                 if outsplit[1] == '[AV_min':
                    av = float(outsplit[0])       

            if plot == 1:
              pdfname =   '%s_starlight.pdf' %(self.inst)
              print ('\tPlotting best starlight fit to %s' %pdfname)        
              pp = PdfPages(pdfname)
              sdatawl = np.array_split(datawl, chop)
              sstarwl = np.array_split(starwl, chop)
              sdata = np.array_split(data, chop)
              sstarfit = np.array_split(starfit, chop)
              sgas = np.array_split(gas, chop)
              for i in range(chop):
                fig1 = plt.figure(figsize = (7,4.4))
                fig1.subplots_adjust(bottom=0.15, top=0.97, left=0.15, right=0.96)
                ax = fig1.add_subplot(1, 1, 1)
                dl = np.median(sgas[i]) - np.median(sstarfit[i]) \
                    + 1.5*np.max(sgas[i]) 
                dl = 0    
                ax.plot(sdatawl[i], 0*sdatawl[i], '--', color ='grey')
                ax.plot(sdatawl[i], sdata[i]+dl, '-', color ='black', label = 'Original spectrum')
                ax.plot(sstarwl[i], sstarfit[i]+dl, '-', lw=1.2,
                            color ='blue', label='Stellar component')
                ax.plot(sdatawl[i], sgas[i], '-', color ='red', label = 'Gas component')
                ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1}) + const.}$',
                               fontsize=18)
                
                ax.set_xlabel(r'Restframe wavelength $(\AA)$', fontsize=18)
                ax.set_ylim(np.min(sgas[i]), 1.2*np.max(sdata[i])+dl)
                ax.set_xlim(np.min(sdatawl[i]), np.max(sdatawl[i]))
                legend = ax.legend(frameon=True, prop={'size':15},
                                    scatterpoints=1, loc = 2)
                rect = legend.get_frame()
                rect.set_facecolor("0.9")
                rect.set_linewidth(0.0)
                rect.set_alpha(0.5)
                pp.savefig(fig1) 
                plt.close(fig1)
              pp.close()

        print ('\tDelta v = %.1f km/s' %v0)        
        print ('\tVelocity dispersion sigma = %.1f km/s' %vd)        
        print ('\tStellar extinction A_V = %.3f mag' %av)        

        return datawl, data, stars, norm, success
        
        
        
def runStar(s2d, arm, ascii, minfit, maxfit, plot=1, verbose=1):

    """ Convinience function to run starlight on an ascii file returning its
    spectral fit and bring it into original rest-frame wavelength scale again
    
    Parameters
    ----------
        ascii : str
            Filename of spectrum in Format WL SPEC ERR FLAG

    Returns
    ----------
        data : np.array (array of zeros if starlight not sucessfull)
            Original data (resampled twice, to check for accuracy)
            
        star : np.array (array of zeros if starlight not sucessfull)
            Starlight fit

        success : int
            Flag whether starlight was executed successully
    """
    
    if verbose == 1:
        print ('\tStarting starlight')
    t1 = time.time()
    sl = StarLight(filen=ascii, inst=s2d.inst, minwl=minfit, maxwl=maxfit)
    datawl, data, stars, norm, success = \
        sl.modOut(plot=plot, minwl=minfit, maxwl=maxfit)
    zerospec = np.zeros(s2d.wave[arm].shape)

    if success == 1:
        if verbose == 1:
            print ('\tRunning starlight took %.2f s' %(time.time() - t1))
        s = sp.interpolate.interp1d(datawl*(1+s2d.redshift), 
                                data*norm/(1+s2d.redshift), fill_value='extrapolate')
        t = sp.interpolate.interp1d(datawl*(1+s2d.redshift), 
                                stars*norm/(1+s2d.redshift), fill_value='extrapolate')
        return s(s2d.wave[arm]), t(s2d.wave[arm]), success
    
    else:
        if verbose ==1:
            print ('\tStarlight failed in %.2f s' %(time.time() - t1))
        return zerospec, zerospec, success
        

def substarlight(s2d, arm, minfit=3700, maxfit=5300, verbose=1):
    """ Convinience function to subtract a starlight fit from a single
    spectrum
    
    Parameters
    ----------
        arm : float
    """
    print '\n\t######################'
    print ('\tInitializing starlight')
    print '\t######################\n'

    s2d.stars = {}

    if arm == 'uvbvis':
        uvbsel = s2d.wave['uvb'] < 5600
        vissel = s2d.wave['vis'] >= 5600
        s2d.wave[arm] = np.append(s2d.wave['uvb'][uvbsel], s2d.wave['vis'][vissel])
        s2d.oneddata[arm] = np.append(s2d.oneddata['uvb'][uvbsel], s2d.oneddata['vis'][vissel])
        s2d.onederro[arm] = np.append(s2d.onederro['uvb'][uvbsel], s2d.onederro['vis'][vissel])
        
    dl = s2d.wave[arm][1]-s2d.wave[arm][0]
    if dl < 0.5:
        wav, dat, ero = binspec(s2d.wave[arm], s2d.oneddata[arm], 
                        s2d.onederro[arm], wl = int(1/dl), meth='median')
    else:
        wav, dat, ero = s2d.wave[arm], s2d.oneddata[arm], s2d.onederro[arm]
    
    ascii = asciiout(s2d=s2d, wl=wav, spec=dat, err=ero, 
                     name='%s' %(arm), fmt='txt')
                      
    data, stars, success = runStar(s2d, arm, ascii, 
               minfit=minfit, maxfit=maxfit, verbose=1)
    
    os.remove(ascii)

    if success == 1:
        s2d.stars[arm] = stars*1E-17
        if arm == 'uvbvis':
            s = sp.interpolate.interp1d(s2d.wave[arm], 
                            stars, fill_value='extrapolate')
            s2d.stars['uvb'] = s(s2d.wave['uvb'])*1E-17
            s2d.stars['vis'] = s(s2d.wave['vis'])*1E-17
        

