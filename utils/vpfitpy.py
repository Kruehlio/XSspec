#!/usr/bin/env python
#
# Copyright (c) 2011 Dark Cosmology Center
# All Rights Reserved.

import os
from tom.astro import absll
absl = absll()

class vpfit:
    def __init__(self, sfile = '', red = ''):
        self.specfile = sfile
        self.redshift = red
        self.wlrange = 4
        self.fort13 = ''
        self.vpfile = ''        
        self.lines = []
        
    def setSpecfile(self, sfile):
        self.specfile = sfile
        
    def setRed(self, red):
        self.redshift = red
        
    def setLines(self, lines):
        self.lines = lines        
        
    def makeFort13(self):
        self.fort13 = os.path.splitext(self.specfile)[0]+'_vpfort.13' 
        f = open(self.fort13, 'w')
        f.write('  *\n')
        for line in self.lines:
            wlmid = absl[line][0] * (1+self.redshift)
            wlmin, wlmax = wlmid - self.wlrange, wlmid + self.wlrange
            f.write('%s\t1\t%.1f\t%.1f\n' %(self.specfile, wlmin, wlmax))
        f.write('  *\n')
        tieletb = 'a'
        lines = []
        for line in self.lines:
            if line not in lines:
                vpline = line.split('_')[0].strip('^')
                f.write('%s\t15\t%.4f\t16%s\t0.00\t1.00E+00\n' \
                    %(vpline, self.redshift, tieletb))
                tieletb = 'A'
                lines.append(line)
        f.close()

    def makeVPfile(self):
        self.vpfile = os.path.splitext(self.fort13)[0]+'.vpin' 
        f = open(self.vpfile, 'w')
        f.write('F\n\n%s\nn\nn\n' %(self.fort13))
        f.close()
    
    def runVP(self):
        os.system('vpfit < %s' %self.vpfile)
    
    def readVPout(self):
        pass
    
    def plotVPout(self):
        pass
        