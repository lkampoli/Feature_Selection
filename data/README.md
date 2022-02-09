# Dataset description

STS = state to state
MT = multi-temperature

## STS transport coefficients
The KAPPA library can be donwloaded here:
```
git clone https://github.com/lkampoli/kappa
```

### Example: shear_viscosity.txt
Transport properties with state-to-state model for N2/N mixture
computed with KAPPA, mixture-sts-transport_properties.cpp

Pressure [Pa] Temperature [K] Molecular molar fractions [] Atomic molar fractions [] Shear viscosity [Pa-s]

## Multi-temperature (6T,5T,4T) code for CO2 mixture transport coefficients
```
git clone https://github.com/Project-CO2/CO2-6T_HARM.git
git clone https://github.com/Project-CO2/CO2-5T_HARM
git clone https://github.com/Project-CO2/Air5-4T_HARM
```

### Example: DB4T.dat
Transport properties for 4T model:
write(666,"(46f15.7)") press, T, TVCO2, TVO2, TVCO, x, & ! inputs
                       visc, bulk_visc, lvibr_co2, lvibr_O2, lvibr_CO, ltot, & ! output
                       THDIF(1), THDIF(2), THDIF(3), THDIF(4), THDIF(5), &
                       ((DIFF(i,j), i=1,5),j=1,5)
