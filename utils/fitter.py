# -*- coding: utf-8 -*-

"""
===========
Spectrum fitter
===========
"""
import numpy
from numpy.ma import median
from numpy import pi, sqrt, log, array, exp
from mpfit import mpfit
from scipy import special, interpolate
from ..analysis.functions import redlaw
""" 
The version of mpfit I use can be found here:
    http://code.google.com/p/agpy/source/browse/trunk/mpfit
"""

def moments(data,circle,rotate,vheight,estimator=median,**kwargs):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above.
    
    If using masked arrays, pass estimator=numpy.ma.median
    """
    total = numpy.abs(data).sum()
    Y, X = numpy.indices(data.shape) # python convention: reverse x,y numpy.indices
    y = numpy.argmax((X*numpy.abs(data)).sum(axis=1)/total)
    x = numpy.argmax((Y*numpy.abs(data)).sum(axis=0)/total)
    col = data[int(y),:]
    # FIRST moment, not second!
    width_x = numpy.sqrt(numpy.abs((numpy.arange(col.size)-y)*col).sum()/numpy.abs(col).sum())
    row = data[:, int(x)]
    width_y = numpy.sqrt(numpy.abs((numpy.arange(row.size)-x)*row).sum()/numpy.abs(row).sum())
    width = ( width_x + width_y ) / 2.
    height = estimator(data.ravel())
    amplitude = data.max()-height
    mylist = [amplitude,x,y]
    if numpy.isnan(width_y) or numpy.isnan(width_x) or numpy.isnan(height) \
        or numpy.isnan(amplitude):
        raise ValueError("something is nan")
    if vheight==1:
        mylist = [height] + mylist
    if circle==0:
        mylist = mylist + [width_x,width_y]
        if rotate==1:
            mylist = mylist + [0.] #rotation "moment" is just zero...
            # also, circles don't rotate.
    else:  
        mylist = mylist + [width]
    return mylist

def twodgaussian(inpars, circle=False, rotate=True, vheight=True, shape=None):
    """Returns a 2d gaussian function of the form:
        x' = numpy.cos(rota) * x - numpy.sin(rota) * y
        y' = numpy.sin(rota) * x + numpy.cos(rota) * y
        (rota should be in degrees)
        g = b + a * numpy.exp ( - ( ((x-center_x)/width_x)**2 +
        ((y-center_y)/width_y)**2 ) / 2 )

        inpars = [b,a,center_x,center_y,width_x,width_y,rota]
                 (b is background height, a is peak amplitude)

        where x and y are the input parameters of the returned function,
        and all other parameters are specified by this function

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

        You can choose to ignore / neglect some of the above input parameters 
            unumpy.sing the following options:
            circle=0 - default is an elliptical gaussian (different x, y
                widths), but can reduce the input by one parameter if it's a
                circular gaussian
            rotate=1 - default allows rotation of the gaussian ellipse.  Can
                remove last parameter by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an
                additive constant for the Gaussian function.  Can remove first
                parameter by setting this to 0
            shape=None - if shape is set (to a 2-parameter list) then returns
                an image with the gaussian defined by inpars
        """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight == 1:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_y, center_x = inpars.pop(0),inpars.pop(0),inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle == 1:
        width = inpars.pop(0)
        width_x, width_y = float(width), float(width)
        rotate = 0
    else:
        width_x, width_y = inpars.pop(0),inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate == 1:
        rota = inpars.pop(0)
        rota = pi/180. * float(rota)
        rcen_x = center_x * numpy.cos(rota) - center_y * numpy.sin(rota)
        rcen_y = center_x * numpy.sin(rota) + center_y * numpy.cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError("There are still input parameters:" + str(inpars) + \
                " and you've input: " + str(inpars_old) + \
                " circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )
            
    def rotgauss(x,y):
        if rotate==1:
            xp = x * numpy.cos(rota) - y * numpy.sin(rota)
            yp = x * numpy.sin(rota) + y * numpy.cos(rota)
        else:
            xp = x
            yp = y
        g = height+amplitude*numpy.exp(
            -(((rcen_x-xp)/width_x)**2+
            ((rcen_y-yp)/width_y)**2)/2.)
        return g
    if shape is not None:
        return rotgauss(*numpy.indices(shape))
    else:
        return rotgauss

def gaussfit(data, err=None,params=(),autoderiv=True, return_all=False,circle=False,
        fixed=numpy.repeat(False,7),limitedmin=[False,False,False,False,True,True,True],
        limitedmax=[False,False,False,False,False,False,True],
        usemoment=numpy.array([],dtype='bool'),
        minpars=numpy.repeat(0,7),maxpars=[0,0,0,0,0,0,360],
        rotate=1,vheight=1,quiet=True,returnmp=False,
        returnfitimage=False,**kwargs):
    """
    Gaussian fitter with the ability to fit a variety of different forms of
    2-dimensional gaussian.
    
    Input Parameters:
        data - 2-dimensional data array
        err=None - error array with same size as data array
        params=[] - initial input parameters for Gaussian function.
            (height, amplitude, x, y, width_x, width_y, rota)
            if not input, these will be determined from the moments of the system, 
            assuming no rotation
        autoderiv=1 - use the autoderiv provided in the lmder.f function (the
            alternative is to us an analytic derivative with lmdif.f: this method
            is less robust)
        return_all=0 - Default is to return only the Gaussian parameters.  
                   1 - fit params, fit error
        returnfitimage - returns (best fit params,best fit image)
        returnmp - returns the full mpfit struct
        circle=0 - default is an elliptical gaussian (different x, y widths),
            but can reduce the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove
            last parameter by setting rotate=0.  numpy.expects angle in DEGREES
        vheight=1 - default allows a variable height-above-zero, i.e. an
            additive constant for the Gaussian function.  Can remove first
            parameter by setting this to 0
        usemoment - can choose which parameters to use a moment estimation for.
            Other parameters will be taken from params.  Needs to be a boolean
            array.
    Output:
        Default output is a set of Gaussian parameters with the same shape as
            the input parameters
        Warning: Does NOT necessarily output a rotation angle between 0 and 360 degrees.
    """
    usemoment=numpy.array(usemoment,dtype='bool')
    params=numpy.array(params,dtype='float')
    if usemoment.any() and len(params)==len(usemoment):
        moment = numpy.array(moments(data,circle,rotate,vheight,**kwargs),dtype='float')
        params[usemoment] = moment[usemoment]
    elif params == [] or len(params)==0:
        params = (moments(data,circle,rotate,vheight,**kwargs))
    if vheight==0:
        vheight=1
        params = numpy.concatenate([[0],params])
        fixed[0] = 1

    for i in xrange(len(params)): 
        if params[i] > maxpars[i] and limitedmax[i]: params[i] = maxpars[i]
        if params[i] < minpars[i] and limitedmin[i]: params[i] = minpars[i]

    if err is None:
        errorfunction = lambda p: numpy.ravel((twodgaussian(p,circle,rotate,vheight)\
                (*numpy.indices(data.shape)) - data))
    else:
        errorfunction = lambda p: numpy.ravel((twodgaussian(p,circle,rotate,vheight)\
                (*numpy.indices(data.shape)) - data)/err)
    def mpfitfun(data,err):
        if err is None:
            def f(p,fjac=None): 
                return [0,numpy.ravel(data-twodgaussian(p,circle,rotate,vheight)\
                    (*numpy.indices(data.shape)))]
        else:
            def f(p,fjac=None): 
                return [0,numpy.ravel((data-twodgaussian(p,circle,rotate,vheight)\
                    (*numpy.indices(data.shape)))/err)]
        return f
                    
    parinfo = [ {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"AMPLITUDE",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                     'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],
                     'parname':"XSHIFT",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                     'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],
                     'parname':"YSHIFT",'error':0},
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],
                     'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],
                     'parname':"XWIDTH",'error':0} ]
    if vheight == 1:
        parinfo.insert(0,{'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                          'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                          'parname':"HEIGHT",'error':0})
    if circle == 0:
        parinfo.append({'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],
                        'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5],
                        'parname':"YWIDTH",'error':0})
        if rotate == 1:
            parinfo.append({'n':6,'value':params[6],'limits':[minpars[6],maxpars[6]],
                            'limited':[limitedmin[6],limitedmax[6]],'fixed':fixed[6],
                            'parname':"ROTATION",'error':0})

    mp = mpfit(mpfitfun(data,err),parinfo=parinfo,quiet=quiet)

    if returnmp:
        returns = (mp)
    elif return_all == 0:
        returns = mp.params
    elif return_all == 1:
        returns = mp.params,mp.perror
    if returnfitimage:
        fitimage = twodgaussian(mp.params,circle,rotate,vheight)\
                    (*numpy.indices(data.shape))
        returns = (returns,fitimage)
    return returns

def onedmoments(Xax,data,vheight=True,estimator=median,negamp=None,
        veryverbose=False, **kwargs):
    """Returns (height, amplitude, x, width_x)
    the gaussian parameters of a 1D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above.
    
    If using masked arrays, pass estimator=numpy.ma.median
    'estimator' is used to measure the background level (height)

    negamp can be used to force the peak negative (True), positive (False),
    or it will be "autodetected" (negamp=None)
    """

    dx = numpy.mean(Xax[1:] - Xax[:-1]) # assume a regular grid
    integral = (data*dx).sum()
    height = estimator(data)
    
    # try to figure out whether pos or neg based on the minimum width of the pos/neg peaks
    Lpeakintegral = integral - height*len(Xax)*dx - (data[data>height]*dx).sum()
    Lamplitude = data.min()-height
    Lwidth_x = 0.5*(numpy.abs(Lpeakintegral / Lamplitude))
    Hpeakintegral = integral - height*len(Xax)*dx - (data[data<height]*dx).sum()
    Hamplitude = data.max()-height
    Hwidth_x = 0.5*(numpy.abs(Hpeakintegral / Hamplitude))
    Lstddev = Xax[data<data.mean()].std()
    Hstddev = Xax[data>data.mean()].std()
    #print "Lstddev: %10.3g  Hstddev: %10.3g" % (Lstddev,Hstddev)
    #print "Lwidth_x: %10.3g  Hwidth_x: %10.3g" % (Lwidth_x,Hwidth_x)

    if negamp: # can force the guess to be negative
        xcen,amplitude,width_x = Xax[numpy.argmin(data)],Lamplitude,Lwidth_x
    elif negamp is None:
        if Hstddev < Lstddev: 
            xcen,amplitude,width_x, = Xax[numpy.argmax(data)],Hamplitude,Hwidth_x
        else:                                                                   
            xcen,amplitude,width_x, = Xax[numpy.argmin(data)],Lamplitude,Lwidth_x
    else:  # if negamp==False, make positive
        xcen,amplitude,width_x = Xax[numpy.argmax(data)],Hamplitude,Hwidth_x

    if veryverbose:
        print "negamp: %s  amp,width,cen Lower: %g, %g   Upper: %g, %g  Center: %g" %\
                (negamp,Lamplitude,Lwidth_x,Hamplitude,Hwidth_x,xcen)
    mylist = [amplitude,xcen,width_x]
    if numpy.isnan(width_x) or numpy.isnan(height) or numpy.isnan(amplitude):
        raise ValueError("something is nan")
    if vheight:
        mylist = [height] + mylist
    return mylist

def plfm(wl, norm, beta = 1., av = 0.1, z = 1., rv = 3.08,
         gamma = 0.922, x0 = 4.592, c1 = '', c2 = 2.35, c3 = 3.26, 
         c4 = 0.41, re = 0):
    ebv, i = av/rv, -1
    if c1 == '':    
        c1 = 2.030 - 3.007*c2
    xcutuv = 10000.0/2700.0
    ancpointsx = [0, 0.377, 0.820, 1.667, 1.828, 2.141, 2.4333]
    ancpointsy = [0, 0.265/3.1*rv, 0.829/3.1*rv,
                  -4.22809e-01 + 1.00270*rv + 2.13572e-04*rv**2,
                  -5.13540e-02 + 1.00216*rv - 7.35778e-05*rv**2,  
                  +7.00127e-01 + 1.00184*rv - 3.32598e-05*rv**2,
                  +1.19456 + 1.01707*rv - 5.46959e-03*rv**2 - 4.45636e-05*rv**3]
    for uvwl in range(2700, 2400,-10):
        x2 = 10000./uvwl
        yuv = c1 + c2*x2 + c3*(x2**2/((x2**2-x0**2)**2 + x2**2*gamma**2)) + rv
        ancpointsx.append(x2)
        ancpointsy.append(yuv)
    fmspline = interpolate.splrep(array(ancpointsx), array(ancpointsy), s = 0)

    x = 10000./(array(wl)/(1+z))
    law = x * 0 + 1.
    for xs in x:
        i += 1
        if xs >= xcutuv:
            if xs >= 5.9:
                fuv = 0.5392*(xs-5.9)**2 + 0.05644*(xs-5.9)**3 
            else:
                fuv = 0
            law[i] = c1+c2*xs+c3*(xs**2/((xs**2-x0**2)**2+xs**2*gamma**2))+c4*fuv + rv
        else:
            law[i] = interpolate.splev(xs, fmspline)
       
    ret = norm * (wl/5.0E3)**(beta) * exp(-1/1.086*ebv*law)
    if re == 1:
        ret =  ebv * law
    return ret
    
def plfmfit(xax, data, err=None, params = [1.,1.,0.1,1.,3.08,0.922,4.592,2.35,3.26,0.41], 
        fixed = [False, False, False, True, False, True, True, False, False, False],
        limitedmin = [True, False, True, True, True, True, True, True, True, True], 
        limitedmax = [False, False, False, False, True, False, False, False, False, True],
        minpars=[0,0,1E-6,0,2,0,0,0,-1,-1], maxpars=[0,0,0,0,5,0,0,0,0,1], 
        quiet=True, shh=False, veryverbose=False):

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0, (y-plfm(x,*p))]
        else:
            def f(p,fjac=None): return [0, (y-plfm(x,*p))/err]
        return f

    if xax == []:
        xax = numpy.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                     'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                     'parname':"Norm",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"Spectral index",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                     'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],
                     'parname':"AV",'error':0} ,
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                     'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],
                     'parname':"Redshift",'error':0},  
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],
                     'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],
                     'parname':"RV",'error':0} , 
                {'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],
                     'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5],
                     'parname':"Gamma",'error':0}, 
                {'n':6,'value':params[6],'limits':[minpars[6],maxpars[6]],
                     'limited':[limitedmin[6],limitedmax[6]],'fixed':fixed[6],
                     'parname':"X0",'error':0}, 
                {'n':7,'value':params[7],'limits':[minpars[7],maxpars[7]],
                     'limited':[limitedmin[7],limitedmax[7]],'fixed':fixed[7],
                     'parname':"c2",'error':0} ,
                {'n':8,'value':params[8],'limits':[minpars[8],maxpars[8]],
                     'limited':[limitedmin[8],limitedmax[8]],'fixed':fixed[8],
                     'parname':"c3",'error':0} ,
                {'n':9,'value':params[9],'limits':[minpars[9],maxpars[9]],
                     'limited':[limitedmin[9],limitedmax[9]],'fixed':fixed[9],
                     'parname':"c4",'error':0} ]                     
    mp = mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    dof = len(data)-len(mpp)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    return mpp, plfm(xax,*mpp), mpperr, [chi2, dof]
    
def pl(wl, norm, beta, av, z, red = 'smc'):
    law = redlaw(wl/(1+z), red)
    return norm * (wl/5.0E3)**(beta) * exp(-1/1.086*av*law)

def plfit(xax, data, err=None, red = 'smc',
          params = [1,1,1,1], fixed = [False,False,False,True],
         limitedmin = [False,False,True,True], limitedmax = [False,False,False,True],
         minpars=[0,0,0,0], maxpars=[0,0,0,15], quiet=True,shh=False,veryverbose=False):

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0,(y-pl(x,*p, red = red))]
        else:
            def f(p,fjac=None): return [0,(y-pl(x,*p, red = red))/err]
        return f

    if xax == []:
        xax = numpy.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                     'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                     'parname':"Norm",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"Spectral index",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                     'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],
                     'parname':"AV",'error':0} ,
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                     'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],
                     'parname':"Redshift",'error':0}  ]

    mp = mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    dof = len(data) - len(mpp)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    return mpp, pl(xax,*mpp, red = red), mpperr, [chi2, dof]

def voigt_m(x, y):
    z = x + 1j*y            
    I = special.wofz(z).real
    return I

def onedvoigt(xo, alphaD, alphaL, x_0, A, a=0, b=0):
   """
   Returns a 1d Voigt profile
   """
   f = sqrt(log(2))
   x = (xo-x_0)/alphaD * f
   y = alphaL/alphaD * f
   backg = a + b*xo
   V = A*f/(alphaD*sqrt(pi)) * voigt_m(x, y) + backg
   return V 

def onedvoigtfit(xax, data, err=None, 
                 params = [1, 1, 1, 1, 0, 0],
                 fixed = [False,False,False,False,True,True],
                 limitedmin = [False,False,False,False,False,False],
                 limitedmax = [False,False,False,False,False,False],
                 minpars=[0,0,0,0,0,0],
                 maxpars=[0,0,0,0,0,0], quiet=True, shh=True,
                 veryverbose=False):
                     
    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0,(y-onedvoigt(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedvoigt(x,*p))/err]
        return f
    if xax == []:
        xax = numpy.arange(len(data))
    
    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                     'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                     'parname':"alphaD",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"alphaL",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                     'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],
                     'parname':"x_0",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                     'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],
                     'parname':"A",'error':0},
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],
                     'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],
                     'parname':"a_back",'error':0} ,
                {'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],
                     'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5],
                     'parname':"b_back",'error':0}]

    mp = mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    dof = len(data)-len(mpp)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    return mpp,onedvoigt(xax,*mpp),mpperr,[chi2, dof]
    
def lin(x, a, b):
    return a + b*x

def linfit(xax, data, err = None, params = [1, 1], fixed = [False, False],
            limitedmin=[False,False], limitedmax=[False,False], 
            minpars=[0,0], maxpars=[0,0], quiet=True, shh=True,
            veryverbose=False):
                
    def mpfitfun(x, y, err):
        if err is None:
            def f(p,fjac=None): return [0,(y-lin(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-lin(x,*p))/err]
        return f                
    if xax == []:
        xax = numpy.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                     'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                     'parname':"Intersect",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"Slope",'error':0} ]

    mp = mpfit(mpfitfun(xax,data,err), parinfo=parinfo, quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    dof = len(data)-len(mpp)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    retax = numpy.arange(xax[0], xax[-1], 0.1)
    return mpp, lin(retax,*mpp), mpperr, [chi2, dof], retax
    
    
#==============================================================================
# Moffatprofile    
#==============================================================================
    
def onedmoffat(x, H, A, dx, alpha, beta):
    """
    Returns a 1-dimensional moffat of form
    H + A * (1 + ((x-dx)/alpha)**2)**(-beta)    """
    return H + A * (1 + ((x-dx)/alpha)**2)**(-beta)    
    
def onedmoffatfit(xax, data, err = None,
        params=[0,1,0,1,3],fixed=[False,False,False,False,False],
        limitedmin=[False,False,False,True,True],
        limitedmax=[False,False,False,False,False], minpars=[0,0,0,0,0],
        maxpars=[0,0,0,0,0], quiet=True, shh=True,
        veryverbose=False,
        vheight=True, negamp=False,
        usemoments=False):

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0,(y-onedmoffat(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedmoffat(x,*p))/err]
        return f

    if xax == []:
        xax = numpy.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                     'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                     'parname':"HEIGHT",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"AMPLITUDE",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                     'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],
                     'parname':"SHIFT",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                     'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],
                     'parname':"ALPHA",'error':0},
                 {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],
                     'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4],
                     'parname':"BETA",'error':0}]

    mp = mpfit(mpfitfun(xax,data,err), parinfo=parinfo, quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    try:
        dof = len(data) - len(mpp)
    except TypeError:
        dof = len(data)
    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    retax = numpy.arange(xax[0], xax[-1], 0.1)
    return mpp, onedmoffat(retax,*mpp), mpperr, [chi2, dof], retax
    
#==============================================================================
# Onedimensional Gaussian    
#==============================================================================
    
    
def onedgaussian(x, H, A, dx, w):
    """
    Returns a 1-dimensional gaussian of form
    H+A*numpy.exp(-(x-dx)**2/(2*w**2))
    """
    return H + A*numpy.exp(-(x-dx)**2 / (2*w**2))


def onedgaussfit(xax, data, err = None,
        params=[0,1,0,1],fixed=[False,False,False,False],
        limitedmin=[False,False,False,True],
        limitedmax=[False,False,False,False], minpars=[0,0,0,0],
        maxpars=[0,0,0,0], quiet=True, shh=True,
        veryverbose=False,
        vheight=True, negamp=False,
        usemoments=False):

    retax = numpy.arange(xax[0], xax[-1], 0.1)
    
    if numpy.any(numpy.isnan(data)):
        return [[numpy.nan, numpy.nan, numpy.nan, numpy.nan], numpy.nan,
                [numpy.nan, numpy.nan, numpy.nan, numpy.nan], numpy.nan]
                

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0,(y-onedgaussian(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedgaussian(x,*p))/err]
        return f

    if xax == []:
        xax = numpy.arange(len(data))

    if vheight is False: 
        height = params[0]
        fixed[0] = True
    
    if usemoments:
        params = onedmoments(xax,data,vheight=vheight,negamp=negamp,veryverbose=veryverbose)
        if vheight is False: params = [height]+params
        if veryverbose: print "OneD moments: h: %g  a: %g  c: %g  w: %g" % tuple(params)
    
    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                     'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],
                     'parname':"HEIGHT",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                     'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],
                     'parname':"AMPLITUDE",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                     'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],
                     'parname':"SHIFT",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                     'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],
                     'parname':"WIDTH",'error':0}]

    mp = mpfit(mpfitfun(xax,data,err), parinfo=parinfo, quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    try:
        dof = len(data) - len(mpp)
    except TypeError:
        dof = len(data)
    
    if mp.status == 0:
        raise Exception(mp.errmsg)
    
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    return mpp, onedgaussian(retax,*mpp), mpperr, [chi2, dof], retax


def onedtwogaussian(x, H, A1, dx1, w1, A2, dx2, w2):
    """
    Returns two 1-dimensional gaussian of form
    H+A*numpy.exp(-(x-dx)**2/(2*w**2))
    """
    g1 = A1 * numpy.exp(-(x-dx1)**2 / (2*w1**2))
    g2 = A2 * numpy.exp(-(x-dx2)**2 / (2*w2**2))
    return H + g1 + g2

def onedtwogaussfit(xax, data, err=None,
        params=[0,1,0,1,1,0,1],
        fixed=[False,False,False,False,False,False,False],
        limitedmin=[False,False,False,True,False,False,True],
        limitedmax=[False,False,False,False,False,False,False], 
        minpars=[0,0,0,0,0,0,0],
        maxpars=[0,0,0,0,0,0,0], 
        tied = ['','','','','','',''],
        quiet=True, shh=True,
        veryverbose=False,
        vheight=True, negamp=False,
        usemoments=False):

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0,(y-onedtwogaussian(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedtwogaussian(x,*p))/err]
        return f

    if xax == []:
        xax = numpy.arange(len(data))

    if vheight is False: 
        height = params[0]
        fixed[0] = True
    if usemoments:
        params = onedmoments(xax,data,vheight=vheight,negamp=negamp, veryverbose=veryverbose)
        if vheight is False: params = [height]+params
        if veryverbose: print "OneD moments: h: %g  a: %g  c: %g  w: %g" % tuple(params)
    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                 'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0], 'tied':tied[0], 
                 'parname':"HEIGHT",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                 'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1], 'tied':tied[1], 
                 'parname':"AMPLITUDE1",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                 'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2], 'tied':tied[2], 
                 'parname':"SHIFT1",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                 'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3], 'tied':tied[3], 
                 'parname':"WIDTH1",'error':0},
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],
                 'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4], 'tied':tied[4], 
                 'parname':"AMPLITUDE2",'error':0},
                {'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],
                 'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5], 'tied':tied[5], 
                 'parname':"SHIFT2",'error':0},
                {'n':6,'value':params[6],'limits':[minpars[6],maxpars[6]],
                 'limited':[limitedmin[6],limitedmax[6]],'fixed':fixed[6], 'tied':tied[6], 
                 'parname':"WIDTH2",'error':0}]

    mp = mpfit(mpfitfun(xax,data,err), parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    dof = len(data)-len(mpp)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    retax = numpy.arange(xax[0], xax[-1], 0.1)
    return mpp, onedtwogaussian(retax,*mpp), mpperr, [chi2, dof], retax
    

def onedthreegaussian(x, H, A1, dx1, w1, A2, dx2, w2, A3, dx3, w3):
    """
    Returns two 1-dimensional gaussian of form
    H+A*numpy.exp(-(x-dx)**2/(2*w**2))
    """
    g1 = A1 * numpy.exp(-(x-dx1)**2 / (2*w1**2))
    g2 = A2 * numpy.exp(-(x-dx2)**2 / (2*w2**2))
    g3 = A3 * numpy.exp(-(x-dx3)**2 / (2*w3**2))
    return H + g1 + g2 + g3

def onedthreegaussfit(xax, data, err=None,
        params = [0,1,0,1,1,0,1,1,0,1],
        fixed = 10*[False],
        limitedmin = [False] + 3*[False, True, True],
        limitedmax = 10*[False], 
        minpars = 10*[0],
        maxpars = 10*[0], 
        tied = 10*[''],
        quiet=True, shh=True,
        veryverbose=False,
        vheight=True, negamp=False,
        usemoments=False):

    def mpfitfun(x,y,err):
        if err is None:
            def f(p,fjac=None): return [0,(y-onedthreegaussian(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedthreegaussian(x,*p))/err]
        return f

    if xax == []:
        xax = numpy.arange(len(data))

    if vheight is False: 
        height = params[0]
        fixed[0] = True
    if usemoments:
        params = onedmoments(xax,data,vheight=vheight,negamp=negamp, veryverbose=veryverbose)
        if vheight is False: params = [height]+params
        if veryverbose: print "OneD moments: h: %g  a: %g  c: %g  w: %g" % tuple(params)
    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],
                 'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0], 'tied':tied[0], 
                 'parname':"HEIGHT",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],
                 'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1], 'tied':tied[1], 
                 'parname':"AMPLITUDE1",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],
                 'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2], 'tied':tied[2], 
                 'parname':"SHIFT1",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],
                 'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3], 'tied':tied[3], 
                 'parname':"WIDTH1",'error':0},
                {'n':4,'value':params[4],'limits':[minpars[4],maxpars[4]],
                 'limited':[limitedmin[4],limitedmax[4]],'fixed':fixed[4], 'tied':tied[4], 
                 'parname':"AMPLITUDE2",'error':0},
                {'n':5,'value':params[5],'limits':[minpars[5],maxpars[5]],
                 'limited':[limitedmin[5],limitedmax[5]],'fixed':fixed[5], 'tied':tied[5], 
                 'parname':"SHIFT2",'error':0},
                {'n':6,'value':params[6],'limits':[minpars[6],maxpars[6]],
                 'limited':[limitedmin[6],limitedmax[6]],'fixed':fixed[6], 'tied':tied[6], 
                 'parname':"WIDTH2",'error':0},
                 {'n':7,'value':params[7],'limits':[minpars[7],maxpars[7]],
                 'limited':[limitedmin[7],limitedmax[7]],'fixed':fixed[7], 'tied':tied[7], 
                 'parname':"AMPLITUDE3",'error':0},
                {'n':8,'value':params[8],'limits':[minpars[8],maxpars[8]],
                 'limited':[limitedmin[8],limitedmax[8]],'fixed':fixed[8], 'tied':tied[8], 
                 'parname':"SHIFT3",'error':0},
                {'n':9,'value':params[9],'limits':[minpars[9],maxpars[9]],
                 'limited':[limitedmin[9],limitedmax[9]],'fixed':fixed[9], 'tied':tied[9], 
                 'parname':"WIDTH3",'error':0}]

    mp = mpfit(mpfitfun(xax,data,err), parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm
    dof = len(data)-len(mpp)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    if (not shh) or veryverbose:
        print "Fit status: ",mp.status
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print parinfo[i]['parname'],p," +/- ",mpperr[i]
        print "Chi2: ",chi2," Reduced Chi2: ",chi2/dof," DOF:",dof
    retax = numpy.arange(xax[0], xax[-1], 0.1)
    return mpp, onedthreegaussian(retax,*mpp), mpperr, [chi2, dof], retax    
    
