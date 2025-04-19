# Copyright (c) 2025, Lawrence Livermore National Security, LLC
#
# LLNL-CODE-853603
#
# SPDX-License-Identifier: (BSD-3-Clause)
#------------------------------------------------------------------------------
# Analytic_Deflection_of_Asteroids_by_NEDs project
#------------------------------------------------------------------------------

from Deflection_formulae import *
import scipy.optimize
import numpy as np

den   =   2.0  # g/cc
Rad   = 200.0  # m
d     =  50.0  # m
Yield = 300.0  # kt
por   = 0.245

# test a range of heights of burst for SiO2, use default coefficients
d = np.linspace(10, 110, num=11)
dv, dvError = OriginalModel_Por_errs(d, Yield, Rad, den, por)
print("SiO2  Radius %f m, Yield %f kt, den %f g/cm^3, por = %f"%(Rad, Yield, den, por))
for i in range(len(d)):
    print("HOB %8.2f m gives DeltaV %f +- %f cm/s "%(d[i], dv[i], dvError[i]))
d = 50

# test a range of yields for Forsterite, need to specify coefficients
print()
Yield = np.linspace(100, 1000, num=10)
dv, dvError = OriginalModel_Por_errs(d, Yield, Rad, den, por, AB_Fo_1_2keV, AB_Fo_1_2keV_covariance)
print("Forsterite  Radius %f m, HOB %f m, den %f g/cm^3, por = %f, T = 2 keV "%(Rad, d, den, por))
for i in range(len(Yield)):
    print("Yield %8.2f kt gives DeltaV %f +- %f cm/s "%(Yield[i], dv[i], dvError[i]))

