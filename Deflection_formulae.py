# Copyright (c) 2025, Lawrence Livermore National Security, LLC and
#
# LLNL-CODE-853603
#
# SPDX-License-Identifier: (BSD-3-Clause)
#------------------------------------------------------------------------------
# Analytic_Deflection_of_Asteroids_by_NEDs project
#------------------------------------------------------------------------------

import math
import sys
import numpy as np
import scipy.integrate
from Fit_data import *

# Fit with constant a and b
# a = math.log(5750.)
# b = math.log(3.16e-4)

def Original_Formula_dim(x, yp) :
    """
    Implement the original dimensionless deflection formula including 
    the low fluence case
    x  - HOB or standoff (m) / radius of asteroid (m)
    yp - Yield (kt)/(b HOB^2)
    """
    
    # mu_1 is cosine of angle where melt sets in
    # mu_t is the tangency angle
    mu_1 = ( 2.0 *(1.0 + x) + (1.0 - yp)*x*x )/( 2.0 *(1.0 + x) )
    mu_t = 1.0/(1.0 + x)
    
    # Mass_melted is the dimensionless part of the melted mass. It is missing the 
    # np.pi*density*Lambda_D*R**2 factor
    Mass_melted = np.where( (yp <= 1.0) | (x == 0.0), 0.0, (x*x/(1.0 + x))* \
        np.where( mu_1 > mu_t, (yp - 1.0 - np.log(yp)), ((2.0/(x + 1e-20))*( 1.0 + np.log(yp)) - \
            (1.0 + 2.0/(x + 1e-20))*np.log(1.0 + 2.0/(x + 1e-20))) ) )
    # Edep is the dimensionless part of the energy dep. It is missing the Y/2 factor
    Edep = 0.5*np.where( yp <= 1.0, 0.0, np.where( mu_1 > mu_t, 1.0 - \
        np.sqrt( 1.0 - (x*x/yp) * ( (yp - 1.0)/( 2.0*(1.0 + x) ) )**2  *(4.0*(1.0 + x)/\
            (x*x*(yp - 1.0) + 1e-20) - 1.0)), 1.0 - np.sqrt( 1.0 - (1.0 + x)**(-2)) ) )
    deltaV = np.sqrt(2.0*Mass_melted*Edep)
    return deltaV

def Original_Formula_dim_errs(x, yp) :
    """
    Implement the original dimensionless deflection formula including 
    the low fluence case
    x  - HOB or standoff (m) / radius of asteroid (m)
    yp - Yield (kt)/(b HOB^2)
    """
    
    x = np.asarray(x)
    yp = np.asarray(yp)
    # mu_1 is cosine of angle where melt sets in
    # mu_t is the tangency angle
    mu_1 = ( 2.0 *(1.0 + x) + (1.0 - yp)*x*x )/( 2.0 *(1.0 + x) )
    mu_t = 1.0/(1.0 + x)
    
    # Mass_melted is the dimensionless part of the melted mass. It is missing the 
    # np.pi*density*Lambda_D*R**2 factor
    Mass_melted = np.where( (yp <= 1.0) | (x == 0.0), 0.0, (x*x/(1.0 + x))* \
        np.where( mu_1 > mu_t, (yp - 1.0 - np.log(yp)), ((2.0/(x + 1e-20))*( 1.0 + np.log(yp)) - \
            (1.0 + 2.0/(x + 1e-20))*np.log(1.0 + 2.0/(x + 1e-20))) ) )
    # Edep is the dimensionless part of the energy dep. It is missing the Y/2 factor
    f =  1.0 - (x*x/yp) * ( (yp - 1.0)/( 2.0*(1.0 + x) ) )**2  *(4.0*(1.0 + x)/(x*x*(yp - 1.0) + 1e-20) - 1.0)
    Edep = 0.5*np.where( yp <= 1.0, 0.0, np.where( mu_1 > mu_t, 1.0 - \
        np.sqrt( f ), 1.0 - np.sqrt( 1.0 - (1.0 + x)**(-2)) ) )
    deltaV = np.sqrt(2.0*Mass_melted*Edep)
    derHi = (x/(1.0 + x)*( 1.0 - np.sqrt( 1.0 - np.power((1.0 + x),-2)))/(deltaV*yp + 1e-20))
    derLo = (0.5/(deltaV + 1e-20))*( x*x/((1.0 + x) + 1e-20)*((1. - 1.0/yp)*(1. - np.sqrt(f)) - 0.5*(yp - 1.0 - np.log(yp))*np.power(f, -0.5)*(-x*x*(yp*yp - 1.)/(4.0*((x+1.)*yp)**2 + 1e-20)*(4.0*(1.0 + x)/(x*x*(yp - 1.0) + 1e-20) - 1.0) + 1.0/((1.0 + x)*yp + 1e-20))))
    derDeltaV_dyp = np.where( (yp <= 1.0) | (x == 0.0), 0.0, np.where( mu_1 > mu_t, derLo, derHi))
    return [deltaV, derDeltaV_dyp]

def OriginalModel(d, Yield, Rad, den, z=AB2):
    """
    evaluate the original model extended for low fluences
    d is standoff in m
    Y is yield in kt
    Rad is radius in m
    den is density in g/cm^3
    z = [[A_coef],[B_coeff]], defaults to 2 keV fits
      constant ceofficients were a =  8.6945, b = -7.95216, 
      exponential of those are   a = 5970,    b = 3.519e-04
    """
    Standoff = d
    # dimensionless HOB
    x = d/Rad
    Fluence = np.log10(Yield/(4*math.pi*Standoff*Standoff))
    Fluence = np.minimum(Fluence, 0.0) # This gives better behavior for small d
    # calculate fit coefficients from the fluence
    A = (z[0][2]*Fluence + z[0][1])*Fluence + z[0][0]
    B = (z[1][2]*Fluence + z[1][1])*Fluence + z[1][0]
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    deltaV = np.exp(A)*np.sqrt(Yield/den) * np.power(Rad,-2) * Original_Formula_dim(x, y)
    return deltaV

def OriginalModel_Por_errs(d, Yield, Rad, den, por, z=AB_Si_1_2keV, covariance=AB_Si_1_2keV_covariance):
    """
    evaluate the original model extended for low fluences
    d is standoff in m
    Y is yield in kt
    Rad is radius in m
    den is density in g/cm^3
    z = [[A_coef],[B_coeff]], defaults to 2 keV fits
      constant coefficients were a =  8.6945, b = -7.95216, 
      exponential of those are   a = 5970,    b = 3.519e-04
    """
    Standoff = d
    # dimensionless HOB
    x = d/Rad
    Fluence = np.log10(Yield/(4*math.pi*Standoff*Standoff))
    Fluence = np.minimum(Fluence, 1.0) # This gives better behavior for small d
    # calculate fit coefficients from the fluence
    A = ((z[0][2] + z[0][5]*por)*Fluence + (z[0][1] + z[0][4]*por))*Fluence + z[0][0] + z[0][3]*por
    B = ((z[1][2] + z[1][5]*por)*Fluence + (z[1][1] + z[1][4]*por))*Fluence + z[1][0] + z[1][3]*por
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    # if d, rad are arrays then x is
    # if Yield, d are arrays then Fluence is
    # if Fluence or por are arrays then A, B are, => A, B array status hinges on d, Yield, por
    # if Yield, B, den, d are arrays then y is - d, Yield, por determine y
    dimlessDeltaV, dimlessDerDeltaV = Original_Formula_dim_errs(x, y)
    # if A, B, x, y are scalar then den, Rad can make deltaV an array
    deltaV = np.exp(A)*np.sqrt(Yield/den) * np.power(Rad,-2) * dimlessDeltaV
    derDeltaVdA = deltaV
    derDeltaVdB = -np.exp(A)*np.power(Yield/den,0.5) * np.power(Rad,-2) * dimlessDerDeltaV*np.exp(B)
    derivVector = np.array([derDeltaVdA, derDeltaVdA*Fluence, derDeltaVdA*Fluence*Fluence, \
                            derDeltaVdA*por, derDeltaVdA*por*Fluence, derDeltaVdA*por*Fluence*Fluence, \
                            derDeltaVdB, derDeltaVdB*Fluence, derDeltaVdB*Fluence*Fluence, \
                            derDeltaVdB*por, derDeltaVdB*por*Fluence, derDeltaVdB*por*Fluence*Fluence])
    if len(derivVector.shape) == 1:
        derivVector = derivVector[:, np.newaxis]
    derivVectorT = np.transpose(derivVector)
    term = np.matvec(covariance, derivVectorT)
    terms = np.zeros(derivVector.shape[1])
    for i in range(derivVector.shape[1]):
        terms = derivVector[:,i] * term[i,:]
    dvError = np.vecdot(derivVectorT, term)
    
    return [deltaV, dvError]

# other model variants

def Modified_Formula_dim(x, yp) :
    """
    Implement the deflection formula with the incident angle dependent melt depth 
    x - HOB or standoff (m) / radius of asteroid (m)
    yp - Yield (kt)/(b HOB^2)
    """
    
    # mu_1 is cosine of tangency angle, mu_t is the tangency angle
    mu_1 = ( 1.0 + (1.0 + x)**2 - yp*x*x )/( 2.0 *(1.0 + x) )
    mu_t = 1.0/(1.0 + x)

    # Note that when yp <= 1.0 that mu_1 >= 1.0
    # Mass_melted is the dimensionless part of the melted mass. 
    # It is missing the np.pi*density*Lambda_D*R**2 factor
    Mass_melted = np.where( (yp <= 1.0) | (x == 0.0), 0.0, (2.0/(9.0*(1.0 + x)))* \
        np.where( mu_1 > mu_t, x*x*(np.sqrt(yp)*(9.0*(2.0 + x) - yp*x) - \
        (2.0*(9.0 + 4.0*x) + 3.0*(3.0 + x)*np.log(yp + 1e-20))), \
        (np.power(x*(2.0+x), 1.5)*(8.0 + 3.0*np.log(yp*x/(x + 2.0) + 1e-20)) - \
            x*x*( 2.0*(4.0*x + 9.0) + 3.0*(x + 3.0)*np.log(yp + 1e-20)) ) ) )
    # Edep is the dimensionless part of the energy dep. It is missing the Y/2 factor
    Edep = 0.5*np.where( yp <= 1.0, 0.0, np.where( mu_1 > mu_t, 1.0 - \
        np.sqrt( 1.0 - (x*x/yp) * ( (yp - 1.0)/( 2.0*(1.0 + x) ) )**2  *\
        (4.0*(1.0 + x)/(x*x*(yp - 1.0)) - 1.0)), 1.0 - np.sqrt( 1.0 - (1.0 + x)**(-2)) ) )
    deltaV = np.sqrt(2.0*Mass_melted*Edep)
    return deltaV

def Modified_Formula_dim_errs(x, yp) :
    """
    Implement the deflection formula with the incident angle dependent melt depth 
    x - HOB or standoff (m) / radius of asteroid (m)
    yp - Yield (kt)/(b D^2)
    """
    
    # mu_1 is cosine of tangency angle, mu_t is the tangency angle
    mu_1 = ( 1.0 + (1.0 + x)**2 - yp*x*x )/( 2.0 *(1.0 + x) )
    mu_t = 1.0/(1.0 + x)

    # Note that when yp <= 1.0 that mu_1 >= 1.0
    # Mass_melted is the dimensionless part of the melted mass. 
    # It is missing the np.pi*density*Lambda_D*R**2 factor
    Mass_melted = np.where( (yp <= 1.0) | (x == 0.0), 0.0, (2.0/(9.0*(1.0 + x)))* \
        np.where( mu_1 > mu_t, x*x*(np.sqrt(yp)*(9.0*(2.0 + x) - yp*x) - \
        (2.0*(9.0 + 4.0*x) + 3.0*(3.0 + x)*np.log(yp + 1e-20))), \
        (np.power(x*(2.0+x), 1.5)*(8.0 + 3.0*np.log(yp*x/(x + 2.0) + 1e-20)) - \
            x*x*( 2.0*(4.0*x + 9.0) + 3.0*(x + 3.0)*np.log(yp + 1e-20)) ) ) )
    # Edep is the dimensionless part of the energy dep. It is missing the Y/2 factor
    f =  1.0 - (x*x/yp) * np.power(( (yp - 1.0)/( 2.0*(1.0 + x) ) ),2)  *(4.0*(1.0 + x)/(x*x*(yp - 1.0) + 1e-20) - 1.0)
    Edep = 0.5*np.where( yp <= 1.0, 0.0, np.where( mu_1 > mu_t, 1.0 - \
        np.sqrt( f ), 1.0 - np.sqrt( 1.0 - (1.0 + x)**(-2)) ) )
    deltaV = np.sqrt(2.0*Mass_melted*Edep)
    derHi = (3.0/(9.0*(1.0 + x))*( 3*np.power(x*(x + 2.0),1.5) - x*x*(x + 3.0) )*( 1.0 - np.sqrt( 1.0 - np.power((1.0 + x),-2)))/(deltaV*yp + 1e-20))
    derLo = (1.0/(deltaV + 1e-20))*( x*x/(9.0*(1.0 + x) + 1e-20)*((1.0/yp)*(0.5*np.sqrt(yp)*(9.0*(x + 2.0) - 3.0*x*yp) - 3.0*(x + 3.0))*(1. - np.sqrt(f)) - 0.5*np.power(f, -0.5)*(np.sqrt(yp)*(9.0*(x + 2.0) - x*yp) - 2.0*(4.0*x + 9.0) - 3.0*(x + 3.0)*np.log(yp))*(-x*x*(yp*yp - 1.)/(4.0*np.power((x+1.)*yp,2) + 1e-20)*(4.0*(1.0 + x)/(x*x*(yp - 1.0) + 1e-20) - 1.0) + 1.0/((1.0 + x)*yp + 1e-20))))
    derDeltaV_dyp = np.where( (yp <= 1.0) | (x == 0.0), 0.0, np.where( mu_1 > mu_t, derLo, derHi))
    return [deltaV, derDeltaV_dyp]

def ModifiedModel(d, Yield, Rad, den, z=AB_mod_2):
    """
    evaluate the model modified to account for angle 
    of incidence in computing melt depth
    d     is standoff in m
    Yield is yield in kt
    Rad   is radius in m
    den   is density in g/cm^3
    z = [[A_coef],[B_coeff]], defaults to 2 keV fits
      constant coefficients were a = 9.31946,    b = -7.49352, 
      exponential of those are   a = 1.1153e+04, b = 5.5668e-04
    """
    Standoff = d
    # dimensionless HOB
    x = d/Rad
    Fluence = np.log10(Yield/(4*math.pi*Standoff*Standoff))
    Fluence = min(Fluence, 0.0) # This gives better behavior for small d
    # calculate fit coefficients from the fluence
    A = (z[0][2]*Fluence + z[0][1])*Fluence + z[0][0]
    B = (z[1][2]*Fluence + z[1][1])*Fluence + z[1][0]
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    deltaV = np.exp(A)*np.sqrt(Yield/den) * Rad**(-2) * Modified_Formula_dim(x, y)
    return deltaV

def ModifiedModel_Por_errs(d, Yield, Rad, den, por, z=AB_mod_Si_1_2keV, covariance=AB_mod_Si_1_2keV_covariance):
    """
    evaluate the model modified to account for angle 
    of incidence in computing melt depth
    d     is standoff in m
    Yield is yield in kt
    Rad   is radius in m
    den   is density in g/cm^3
    z = [[A_coef],[B_coeff]], defaults to 2 keV fits
      constant coefficients were a = 9.31946,    b = -7.49352, 
      exponential of those are   a = 1.1153e+04, b = 5.5668e-04
    """
    Standoff = d
    # dimensionless HOB
    x = d/Rad
    Fluence = np.log10(Yield/(4*math.pi*Standoff*Standoff))
    Fluence = np.minimum(Fluence, 0.0) # This gives better behavior for small d
    # calculate fit coefficients from the fluence
    A = ((z[0][2] + z[0][5]*por)*Fluence + (z[0][1] + z[0][4]*por))*Fluence + z[0][0] + z[0][3]*por
    B = ((z[1][2] + z[1][5]*por)*Fluence + (z[1][1] + z[1][4]*por))*Fluence + z[1][0] + z[1][3]*por
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    dimlessDeltaV, dimlessDerDeltaV = Modified_Formula_dim_errs(x, y)
    # if A, B, x, y are scalar then den, Rad can make deltaV an array
    deltaV = np.exp(A)*np.sqrt(Yield/den) * np.power(Rad,-2) * dimlessDeltaV
    derDeltaVdA = deltaV
    derDeltaVdB = -np.exp(A)*np.power(Yield/den,0.5) * np.power(Rad,-2) * dimlessDerDeltaV*np.exp(B)
    derivVector = np.array([derDeltaVdA, derDeltaVdA*Fluence, derDeltaVdA*Fluence*Fluence, \
                            derDeltaVdA*por, derDeltaVdA*por*Fluence, derDeltaVdA*por*Fluence*Fluence, \
                            derDeltaVdB, derDeltaVdB*Fluence, derDeltaVdB*Fluence*Fluence, \
                            derDeltaVdB*por, derDeltaVdB*por*Fluence, derDeltaVdB*por*Fluence*Fluence])
    if len(derivVector.shape) == 1:
        derivVector = derivVector[:, np.newaxis]
    derivVectorT = np.transpose(derivVector)
    term = np.matvec(covariance, derivVectorT)
    terms = np.zeros(derivVector.shape[1])
    for i in range(derivVector.shape[1]):
        terms = derivVector[:,i] * term[i,:]
    dvError = np.vecdot(derivVectorT, term)
    return [deltaV, dvError]

def Impulse_integrand(mu, x, yp):
    """
    integrand for the impulse integral.
    mu - value of the angle
    x  - D/R - (HOB or standoff (m))/(radius of asteroid (m))
    yp - Yield (kt)/(b HOB^2)
    """
    ssqr = 1.0 + (1.0 + x)**2 - 2.0*(1.0 + x)*mu
    Term = yp*x*x/(ssqr) - 1.0
    if Term < 0.0:
        Term = 0.0
    sqrt_Term = math.sqrt(Term)
    Result = mu*(-1.0 + (1.0 + x)*mu)*(sqrt_Term - math.atan(sqrt_Term))/ \
    (math.sqrt(ssqr) + 1e-20)
    return Result

def Impulse_dim(x, yp):
    """
    provide dimensionless delta v from Impulse formula for given parms
    x  - D/R - (HOB or standoff (m))/(radius of asteroid (m))
    yp - Yield (kt)/(b D^2)
    """
    # lower limit is max of tangent value and melt limit value
    mu_1 = max(1.0/(1.0 + x), 0.5*(1.0 + (1.0 + x)**2 - yp*x*x)/(1.0 + x))
    # do the integral
    if mu_1 < 1.0:
        Result = 2**1.5 /(x*math.sqrt(yp)) * \
        scipy.integrate.quad(Impulse_integrand, mu_1, 1.0, args=(x, yp))[0]
    else:
        Result = 0.0
    return Result

def ImpulseModel(d, Yield, Rad, den, z=AB_imp_2):
    """
    evaluate the impulse model
    d     is standoff in m
    Yield is yield in kt
    Rad   is radius in m
    den   is density in g/cm^3
    z = [a,b], defaults to 2kev fit
      constant coefficients were a = 12.0197,    b = -2.3033
      exponential of those are   a =  1.660e+05, b = 9.993e-02
    """
    Standoff = d
    # dimensionless HOB
    x = d/Rad
    Fluence = np.log10(Yield/(4*math.pi*Standoff*Standoff))
    Fluence = min(Fluence, 0.0) # This gives better behavior for small d
    # calculate fit coefficients from the fluence
    A = (z[0][2]*Fluence + z[0][1])*Fluence + z[0][0]
    B = (z[1][2]*Fluence + z[1][1])*Fluence + z[1][0]
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    deltaV = np.zeros(np.shape(np.asarray(x)))
    if (type(x) == np.float64) or (type(x) == float):
        deltaV = np.exp(A)*np.sqrt(Yield/den) * np.power(Rad,-2) * Impulse_dim(x, y)
    else:
        for i in range(len(np.asarray(deltaV))):
            deltaV = np.exp(A)*np.sqrt(Yield/den) * np.power(Rad,-2) * Impulse_dim(x[i], y[i])
    return deltaV
