#!/usr/bin/env python

import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

molecBase = os.path.expanduser('~/bin/molecfit')
molecCall = os.path.join(molecBase, 'bin/molecfit')
transCall = os.path.join(molecBase, 'bin/calctrans')

class molecFit():
    ''' Sets up and runs a molecfit'''
    
    def __init__(self, s2d, header = '', output='.'):
        ''' Reads in the required information to set up the parameter files'''
        self.s2d = s2d
        self.molecparfile = {}
        self.params = {'basedir': molecBase,
           'listname': 'none', 'trans' : 1,
           'columns': 'LAMBDA FLUX ERR NULL',
           'default_error': 0.01, 'wlgtomicron' : 0.0001,
           'vac_air': 'vac', 'list_molec': [], 'fit_molec': [],
           'wrange_exclude': 'none',
           'output_dir': os.path.abspath(os.path.expanduser(output)),
           'plot_creation' : '', 'plot_range': 1,
           'ftol': 0.01, 'xtol': 0.01, 
           'relcol': [], 'flux_unit': 2,
           'fit_back': 0, 'telback': 0.1, 'fit_cont': 1, 'cont_n': 3,
           'cont_const': 1.0, 'fit_wlc': 1, 'wlc_n': 0, 'wlc_const': 0.0,
           'fit_res_box': 1, 'relres_box': 0.5, 'kernmode': 0, 'fit_res_gauss': 1,
           'res_gauss': 1.0, 'fit_res_lorentz': 0, 'res_lorentz': 0.5, 'kernfac': 30.0,
           'varkern': 1, 'kernel_file': 'none'}
        self.headpars = {'utc': 'UTC', 'telalt': 'HIERARCH ESO TEL ALT', 
             'rhum' : 'HIERARCH ESO TEL AMBI RHUM',
             'obsdate' : 'MJD-OBS',
             'temp' : 'HIERARCH ESO TEL AMBI TEMP',
             'm1temp' : 'HIERARCH ESO TEL TH M1 TEMP',
             'geoelev': 'HIERARCH ESO TEL GEOELEV',
             'longitude': 'HIERARCH ESO TEL GEOLON',
             'latitude': 'HIERARCH ESO TEL GEOLAT',
             'pixsc' : 'CD2_2' } 
        self.atmpars = {'ref_atm': 'equ.atm', 
            'gdas_dir': os.path.join(molecBase, 'data/profiles/grib'),
             'gdas_prof': 'none', 'layers': 1, 'emix': 5.0, 'pwv': -1.} 

    def updateParams(self, arm, paramdic, headpars):
        ''' Sets Parameters for molecfit execution'''
        
        for key in paramdic.keys():
            print '\t\tMolecfit: Setting parameter %s to %s' %(key,  paramdic[key])
            if key in self.params.keys():
                self.params[key] = paramdic[key]
            elif key in self.headpars.keys():
                self.headpars[key] = paramdic[key]
            elif key in self.atmpars.keys():
                self.atmpars[key] = paramdic[key]
            else:
                print '\t\tWarning: Parameter %s not known to molecfit'
                self.params[key] = paramdic[key]
        
    def setParams(self, arm):
        ''' Writes Molecfit parameters into file '''
        wrange = os.path.join(molecBase, 'examples/config/include_xshoo_%s.dat' % arm)
        prange = os.path.join(molecBase, 'examples/config/exclude_xshoo_%s.dat' % arm)
        self.params['wrange_include'] =  wrange
        self.params['prange_exclude'] =  prange

        if arm == 'vis':
            self.params['list_molec'] = ['H2O', 'O2']
            self.params['fit_molec'] = [1, 1]
            self.params['relcol'] = [1.0, 1.0]
            self.params['wlc_n'] = 0

        elif arm == 'nir':
            self.params['list_molec'] = ['H2O', 'CO2', 'CO', 'CH4', 'O2']
            self.params['fit_molec'] = [1, 0, 0, 0, 0]
            self.params['relcol'] = [1.0, 1.06, 1.0, 1.0, 1.0]
            self.params['wlc_n'] = 1

        print '\tPreparing Molecfit params file'
        self.molecparfile[arm] = os.path.abspath('./%s_molecfit_%s.par' % (self.s2d.object, arm))
        f = open(self.molecparfile[arm], 'w')
        f.write('## INPUT DATA\n')
        f.write('filename: %s.spec\n' % (os.path.abspath(self.s2d.output[arm])))
        f.write('output_name: %s_%s_molecfit\n' % (self.s2d.object, arm))

        for param in self.params.keys():
            if hasattr(self.params[param], '__iter__'):
                f.write('%s: %s \n' % (param, ' '.join([str(a) for a in self.params[param]]) ))
            else:
                f.write('%s: %s \n' % (param, self.params[param] ))

        f.write('\n## HEADER PARAMETERS\n')
        if arm == 'vis':
            slitkey = 'HIERARCH ESO INS OPTI4 NAME'
        elif arm == 'nir':
            slitkey = 'HIERARCH ESO INS OPTI5 NAME'
        slitw = self.s2d.head[arm][slitkey].split('x')[0]
        f.write('slitw: %s\n' %(slitw))
        for headpar in self.headpars.keys():
            f.write('%s: %s \n' % (headpar,  self.s2d.head[arm][self.headpars[headpar]]))

        f.write('\n## ATMOSPHERIC PROFILES\n')
        for atmpar in self.atmpars.keys():
            f.write('%s: %s \n' % (atmpar, self.atmpars[atmpar]))
        
        f.write('\nend\n')
        f.close()
            
    def runMolec(self, arm):
        t1 = time.time()
        print '\tRunning molecfit'
        print '\t%s' %(' '.join([molecCall, self.molecparfile[arm]]))
        runMolec = subprocess.Popen([molecCall, self.molecparfile[arm]],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
                                    
        runMolec.wait()
        runMolecRes = runMolec.stdout.readlines()
        molecpar = os.path.abspath('./%s_molecfit_%s.output' % (self.s2d.object, arm))
        f = open(molecpar, 'w')
        f.write(''.join(runMolecRes))
        f.close()
        
        if runMolecRes[-1].strip() == '[ INFO  ] No errors occurred':
            print '\tMolecfit sucessful in %.0f s' % (time.time()-t1)
        else:
            print runMolecRes[-1].strip()
        
        t1 = time.time()
        print '\tRunning calctrans'
        runTrans = subprocess.Popen([transCall, self.molecparfile[arm]],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        runTrans.wait()
        runTransRes = runTrans.stdout.readlines()
        runtranspar = os.path.abspath('./%s_calctrans_%s.output' % (self.s2d.object, arm))
        f = open(runtranspar, 'w')
        f.write(''.join(runTransRes))
        f.close()        
        
        if runTransRes[-1].strip() == '[ INFO  ] No errors occurred':
            print '\tCalctrans sucessful in %.0f s' % (time.time()-t1)
        else:
            print runTransRes[-1].strip()
            
    def updateSpec(self, arms, tacfile = ''):
        
        ''' Read in Calctrans output und update spectrum2d class with telluric
        correction spectrum '''
        print '\tUpdating the spectra with telluric-corrected data'
        for arm in arms:
            if tacfile == '':
                tacfilearm = self.s2d.output[arm] + '_TAC.spec'
            if os.path.isfile(tacfilearm):
                f = open(tacfilearm, 'r')
                tacspec = [item for item in f.readlines() if not item.startswith('#')]
                f.close()
                wl, rawspec, rawspece, transm, tcspec, tcspece = 6 * [np.array([])]
                for spec in tacspec:
                    column = [float(ent) for ent in spec.split()]
                    wl = np.append(wl, column[0])
                    rawspec = np.append(rawspec, column[1])
                    rawspece = np.append(rawspece, column[2])
                    transm = np.append(transm, column[3])
                    tcspec = np.append(tcspec, column[4])
                    tcspece = np.append(tcspece, column[5])
                self.s2d.oneddata[arm] = tcspec/self.s2d.mult
                self.s2d.onederro[arm] = tcspece/self.s2d.mult
                self.s2d.wave[arm] = wl
                self.s2d.telcorr[arm] = transm
                self.s2d.output[arm] += '_tac'
                
            # Plot the fit regions
            wrange = os.path.join(molecBase, 'examples/config/include_xshoo_%s.dat' % arm)
            g = open(wrange, 'r')
            fitregs = [reg for reg in g.readlines() if not reg.startswith('#')]
            g.close()
            
            pp = PdfPages('%s_tellcor_%s.pdf' % (self.s2d.object, arm) )
            print '\tPlotting telluric-corrected data for arm %s' %arm

            for fitreg in fitregs:
                mictowl = 1./self.params['wlgtomicron']
                if float(fitreg.split()[1])*mictowl < self.s2d.wave[arm][-1]:
                    x1 = self.wltopix(arm, float(fitreg.split()[0])*mictowl)
                    x2 = self.wltopix(arm, float(fitreg.split()[1])*mictowl)
                    
        
                    fig = plt.figure(figsize = (9.5, 7.5))
                    fig.subplots_adjust(hspace=0.05, wspace=0.0, right=0.97)
        
                    ax1 = fig.add_subplot(2, 1, 1)
                    wlp, tcspecp, transmp = wl[x1:x2], tcspec[x1:x2], transm[x1:x2] 

                    if len(tcspecp[transmp>0.90]) > 3:
                        cont = np.median(tcspecp[transmp>0.90])/np.median(transmp[transmp>0.90])
                        ax1.plot(wlp, transmp * cont, '-' ,color = 'firebrick', lw = 2)
                    else:
                        ax1.errorbar(wlp, tcspec[x1:x2], tcspece[x1:x2],
                        capsize = 0, color = 'firebrick', fmt = 'o', ms = 4,
                        mec = 'grey', lw = 0.8, mew = 0.5)
                    
                    ax1.errorbar(wlp, rawspec[x1:x2], rawspece[x1:x2],
                        capsize = 0, color = 'black', fmt = 'o', ms = 4,
                        mec = 'grey', lw = 0.8, mew = 0.5)
        
                    ax2 = fig.add_subplot(2, 1, 2)
                    if len(tcspecp[transmp>0.90]) > 3:
                        ax2.errorbar(wlp, rawspec[x1:x2] / (transmp * cont), 
                            yerr = rawspece[x1:x2] / (transmp * cont),
                            capsize = 0, color = 'black', fmt = 'o',  ms = 4,
                            mec = 'grey', lw = 0.8, mew = 0.5)
                    else:
                        ax2.plot(wlp, transmp, '-' ,color = 'firebrick', lw = 2)

        
                    for ax in [ax1, ax2]:
                        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
                        ax.set_xlim(float(fitreg.split()[0])*mictowl, float(fitreg.split()[1])*mictowl)
                        ax.set_ylim(ymin=-0.05)
        
                    ax1.xaxis.set_major_formatter(plt.NullFormatter())
                    ax2.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
                    ax1.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$' \
                                 %(str(self.s2d.mult))[-2:])
                    if len(tcspecp[transmp>0.90]) > 3:
                        ax2.set_ylim(ymax = max(rawspec[x1:x2] / (transmp * cont))*1.05)
                        ax2.set_ylabel(r'Ratio')
                    else:
                        ax2.set_ylim(ymax=1.05)
                        ax2.set_ylabel(r'Transmission')
                    ax1.set_ylim(ymax = max(tcspec[x1:x2])*1.05)
        
                    pp.savefig(fig)
                    plt.close(fig)  
            pp.close()            
        return self.s2d
        
    def wltopix(self, arm, wl):
        dl = (self.s2d.wave[arm][-1]-self.s2d.wave[arm][0]) / (len(self.s2d.wave[arm]) - 1)
        pix = ((wl - self.s2d.wave[arm][0]) / dl) + 1
        return max(0, int(round(pix)))       