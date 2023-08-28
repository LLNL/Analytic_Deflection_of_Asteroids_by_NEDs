# Analytic_NED_Asteroid_Deflection_Model

This package evaluates an analytic formula for the &Delta v imparted to an asteroid by a 1 or 2 keV black body source given the radius of the asteroid, the standoff distance, the Yield in x-rays, the density of the asteroid, and the fit coefficients for the 1 or 2 keV black body source (provided in the module). 
Three variations of the formula are given; the "original" form, the "corrected" form that accounts for the angle of incidence in calculating the melt depth, and a form based on an "impulse" model.
Each formula is fit to simulations of spherical asteroids of uniform composiiton illuminated by black body x-rays.

A copy of the report on the recent changes to the models from PDC 2023, [PDC2023_Managan](PDC2023_Managan.pdf), is included. 
I have also included a more detailed report from the proceedings of PDC 2021, [PDC_2021_Managan](Managan_PDC_2021.pdf). 
This report assumes that the fitting coefficients are constant.


## Getting Started

Clone the git repository to a convenient location. You can copy the Deflections_formulae.py module to your Python site-packages directory if you want it globally accessible.


### Prerequisites

The Python scripts load [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org). 

## Running the example

The example script, CalcDV.py, prints out the delta-V generated by a few examples.
```
python CalcDV.py
```
It uses the preferred formula, OriginalModel(d, Yield, Rad, den, z), where d is the standoff distance in m, Y is yield in kt, Rad is radius in m, den is density in g/cm^3, z is the set of coefficients to use, either AB1 or AB2 for 1 keV or 2 keV source respectively.

For completeness the other two formulae given in [PDC_2021_Managan](Managan_PDC_2021.pdf) are also included.
They are ModifiedModel(d, Yield, Rad, den, z) and ImpulseModel(d, Yield, Rad, den, z).


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Robert Managan** - *Initial work* - [Managan](https://people.llnl.gov/managan1)
* **Mary Burkey** - *Initial work* - [Burkey](https://people.llnl.gov/burkey1)

See also the list of [contributors](CONTRIBUTING.md) who participated in this project.

## License

This project is licensed under the BSD-3 License - see the [LICENSE.md](LICENSE.md) file for details

Unlimited Open Source - BSD 3-clause Distribution LLNL-CODE-853603

## SPDX usage

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)


