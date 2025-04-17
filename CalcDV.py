# Copyright (c) 2023, Lawrence Livermore National Security, LLC
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

TestOriginal = True
TestModified = False
TestImpulse  = False
TestPorosity = True

den   =   2.0  # g/cc
Rad   = 200.0  # m
d     =  50.0  # m
Yield = 300.0  # kt
por   = 0.245

if TestOriginal:
    print("\nThe original model gives:")
    print("Radius %f m, Yield %f kt, den %f g/cm^3, T = 1 keV "%(Rad, Yield, den))
    for d in np.linspace(10, 110, num=11):
        DV = OriginalModel(d, Yield, Rad, den, z=AB1)
        print("HOB %8.2f m gives DeltaV %f cm/s "%(d, DV))
    
    print()
    d = 50.0
    print("Radius %f m, HOB %f m, den %f g/cm^3, T = 2 keV "%(Rad, d, den))
    for Y in np.linspace(100, 1000, num=10):
        DV = OriginalModel(d, Y, Rad, den, z=AB2)
        print("Yield %8.2f kt, gives DeltaV %f cm/s "%(Y, DV))

    if TestPorosity:
        print()
        d = np.linspace(10, 110, num=11)
        dv, dvError = OriginalModel_Por_errs(d, Yield, Rad, den, por, AB_Si_1keV, AB_Si_1keV_covariance)
        print("Radius %f m, Yield %f kt, den %f g/cm^3, por = %f, T = 1 keV "%(Rad, Yield, den, por))
        for i in range(len(d)):
            print("HOB %8.2f m gives DeltaV %f +- %f cm/s "%(d[i], dv[i], dvError[i]))
        d = 50

        print()
        Yield = np.linspace(100, 1000, num=10)
        dv, dvError = OriginalModel_Por_errs(d, Yield, Rad, den, por, AB_Si_2keV, AB_Si_2keV_covariance)
        print("Radius %f m, HOB %f m, den %f g/cm^3, por = %f, T = 2 keV "%(Rad, d, den, por))
        for i in range(len(Yield)):
            print("Yield %8.2f kt gives DeltaV %f +- %f cm/s "%(Yield[i], dv[i], dvError[i]))

        print()
        dv, dvError = OriginalModel_Por_errs(d, Yield, Rad, den, por)
        print("Radius %f m, HOB %f m, den %f g/cm^3, por = %f, T = 1.5 keV "%(Rad, d, den, por))
        for i in range(len(Yield)):
            print("Yield %8.2f kt gives DeltaV %f +- %f cm/s "%(Yield[i], dv[i], dvError[i]))
        Yield = 300.0


# Modified model example

if TestModified:
    print("\nThe modified model gives:")
    print("Radius %f m, Yield %f kt, den %f g/cm^3, T = 1 keV "%(Rad, Yield, den))
    for d in np.linspace(10, 110, num=11):
        DV = ModifiedModel(d, Yield, Rad, den, z=AB_mod_1)
        print("HOB %8.2f m gives DeltaV %f cm/s "%(d, DV))
    
    print()
    d = 50.0
    print("Radius %f m, HOB %f kt, den %f g/cm^3, T = 2 keV "%(Rad, d, den))
    for Y in np.linspace(100, 1000, num=10):
        DV = ModifiedModel(d, Y, Rad, den, z=AB_mod_2)
        print("Yield %8.2f kt, gives DeltaV %f cm/s "%(Y, DV))

    if TestPorosity:
        print()
        d = np.linspace(10, 110, num=11)
        dv, dvError = ModifiedModel_Por_errs(d, Yield, Rad, den, por, AB_Si_1keV, AB_Si_1keV_covariance)
        print("Radius %f m, Yield %f kt, den %f g/cm^3, por = %f, T = 1 keV "%(Rad, Yield, den, por))
        for i in range(len(d)):
            print("HOB %8.2f m gives DeltaV %f +- %f cm/s "%(d[i], dv[i], dvError[i]))
        d = 50

        print()
        Yield = np.linspace(100, 1000, num=10)
        dv, dvError = ModifiedModel_Por_errs(d, Yield, Rad, den, por, AB_Si_2keV, AB_Si_2keV_covariance)
        print("Radius %f m, HOB %f m, den %f g/cm^3, por = %f, T = 2 keV "%(Rad, d, den, por))
        for i in range(len(Yield)):
            print("Yield %8.2f kt gives DeltaV %f +- %f cm/s "%(Yield[i], dv[i], dvError[i]))
        Yield = 300.0

#  Impulse model example

if TestImpulse:
    print("\nThe impulse model gives:")
    print("Radius %f m, Yield %f kt, den %f g/cm^3, T = 1 keV "%(Rad, Yield, den))
    for d in np.linspace(10, 110, num=11):
        DV = ImpulseModel(d, Yield, Rad, den, z=AB_imp_1)
        print("HOB %8.2f m gives DeltaV %f cm/s "%(d, DV))
    
    print()
    d = 50.0
    print("Radius %f m, HOB %f kt, den %f g/cm^3, T = 2 keV "%(Rad, d, den))
    for Y in np.linspace(100, 1000, num=10):
        DV = ImpulseModel(d, Y, Rad, den, z=AB_imp_2)
        print("Yield %8.2f kt, gives DeltaV %f cm/s "%(Y, DV))
