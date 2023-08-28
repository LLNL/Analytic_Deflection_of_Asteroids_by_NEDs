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

den   =   2.0  # g/cc
Rad   = 200.0  # m
d     =  50.0  # m
Yield = 300.0  # kt

print("\nThe original model gives:")
print("Radius %f m, Yield %f kt, den %f g/cm^3, T = 1 keV "%(Rad, Yield, den))
for d in np.linspace(10, 110, num=11):
    DV = OriginalModel(d, Yield, Rad, den, z=AB1)
    print("HOB %8.2f m gives DeltaV %f cm/s "%(d, DV))

print()
d = 50.0
print("Radius %f m, HOB %f kt, den %f g/cm^3, T = 2 keV "%(Rad, d, den))
for Y in np.linspace(100, 1000, num=10):
    DV = OriginalModel(d, Y, Rad, den, z=AB2)
    print("Yield %8.2f kt, gives DeltaV %f cm/s "%(Y, DV))

# Modified model example

# print("\nThe modified model gives:")
# print("Radius %f m, Yield %f kt, den %f g/cm^3, T = 1 keV "%(Rad, Yield, den))
# for d in np.linspace(10, 110, num=11):
#     DV = ModifiedModel(d, Yield, Rad, den, z=AB_mod_1)
#     print("HOB %8.2f m gives DeltaV %f cm/s "%(d, DV))

# print()
# d = 50.0
# print("Radius %f m, HOB %f kt, den %f g/cm^3, T = 2 keV "%(Rad, d, den))
# for Y in np.linspace(100, 1000, num=10):
#     DV = ModifiedModel(d, Y, Rad, den, z=AB_mod_2)
#     print("Yield %8.2f kt, gives DeltaV %f cm/s "%(Y, DV))

#  Impulse model example

# print("\nThe impulse model gives:")
# print("Radius %f m, Yield %f kt, den %f g/cm^3, T = 1 keV "%(Rad, Yield, den))
# for d in np.linspace(10, 110, num=11):
#     DV = ImpulseModel(d, Yield, Rad, den, z=AB_imp_1)
#     print("HOB %8.2f m gives DeltaV %f cm/s "%(d, DV))

# print()
# d = 50.0
# print("Radius %f m, HOB %f kt, den %f g/cm^3, T = 2 keV "%(Rad, d, den))
# for Y in np.linspace(100, 1000, num=10):
#     DV = ImpulseModel(d, Y, Rad, den, z=AB_imp_2)
#     print("Yield %8.2f kt, gives DeltaV %f cm/s "%(Y, DV))
