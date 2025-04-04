# Copyright (c) 2023, Lawrence Livermore National Security, LLC and
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

# Fit with constant a and b
# a = math.log(5750.)
# b = math.log(3.16e-4)

# 1 keV black body fits to original formula
A_1keV = np.array([0.127528, 1.11471, 10.7089])
B_1keV = np.array([0.159871, 3.06632, -0.808491])

# 2 keV black body fits to original formula
A_2keV = np.array([0.0688574, 0.791745, 11.2799])
B_2keV = np.array([0.130942, 2.97197, -1.09399])

AB1 = np.array([A_1keV,B_1keV])
AB2 = np.array([A_2keV,B_2keV])

# 1 keV black body fits to modified formula
A_mod_1keV = np.array([ 0.056303, 0.996104, 11.1338])
B_mod_1keV = np.array([-0.356001, 2.20468,  -0.805492])

# 2 keV black body fits to modified formula
A_mod_2keV = np.array([ 0.0135686, 0.79254, 11.7366])
B_mod_2keV = np.array([-0.768708,  1.884,   -1.15929])

AB_mod_1 = np.array([A_mod_1keV,B_mod_1keV])
AB_mod_2 = np.array([A_mod_2keV,B_mod_2keV])

# 1 keV black body fits to impulse formula
A_imp_1keV = np.array([ 0.0381572, 0.929061, 11.6852])
B_imp_1keV = np.array([-0.28174, 2.00948, -0.822539])

# 2 keV black body fits to impulse formula
A_imp_2keV = np.array([ 0.00549028, 0.694854, 12.2464])
B_imp_2keV = np.array([-0.409627,   1.76872, -1.20587])

AB_imp_1 = np.array([A_imp_1keV,B_imp_1keV])
AB_imp_2 = np.array([A_imp_2keV,B_imp_2keV])

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
    Mass_melted = np.where( (yp <= 1.0) | (x == 0.0), 0.0, (x**2/(1.0 + x))* \
        np.where( mu_1 > mu_t, (yp - 1.0 - np.log(yp)), ((2.0/(x + 1e-20))*( 1.0 + np.log(yp)) - \
            (1.0 + 2.0/(x + 1e-20))*np.log(1.0 + 2.0/(x + 1e-20))) ) )
    # Edep is the dimensionless part of the energy dep. It is missing the Y/2 factor
    Edep = 0.5*np.where( yp <= 1.0, 0.0, np.where( mu_1 > mu_t, 1.0 - \
        np.sqrt( 1.0 - (x*x/yp) * ( (yp - 1.0)/( 2.0*(1.0 + x) ) )**2  *(4.0*(1.0 + x)/\
            (x*x*(yp - 1.0) + 1e-20) - 1.0)), 1.0 - np.sqrt( 1.0 - (1.0 + x)**(-2)) ) )
    deltaV = np.sqrt(2.0*Mass_melted*Edep)
    return deltaV

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
    A = (z[0][0]*Fluence + z[0][1])*Fluence + z[0][2]
    B = (z[1][0]*Fluence + z[1][1])*Fluence + z[1][2]
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    deltaV = np.exp(A)*np.power(Yield/den,0.5) * np.power(Rad,-2) * Original_Formula_dim(x, y)
    return deltaV

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
    A = (z[0][0]*Fluence + z[0][1])*Fluence + z[0][2]
    B = (z[1][0]*Fluence + z[1][1])*Fluence + z[1][2]
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    deltaV = np.exp(A)*Yield**0.5 * Rad**(-2) * Modified_Formula_dim(x, y)
    return deltaV

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
    A = (z[0][0]*Fluence + z[0][1])*Fluence + z[0][2]
    B = (z[1][0]*Fluence + z[1][1])*Fluence + z[1][2]
    # dimensionless fluence
    y = Yield/(np.exp(B)*den*Standoff*Standoff)
    deltaV = np.zeros(np.shape(np.asarray(x)))
    if (type(x) == np.float64) or (type(x) == float):
        deltaV = np.exp(A)*Yield**0.5 * Rad**(-2) * Impulse_dim(x, y)
    else:
        for i in range(len(np.asarray(deltaV))):
            deltaV = np.exp(A)*Yield**0.5 * Rad**(-2) * Impulse_dim(x[i], y[i])
    return deltaV
