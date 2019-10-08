"""
This code computes some key design parameters of a LOX-CH4 liquid rocket engine, namely:

- Oxidiser mass flow rate (kg/s)
- Fuel mass flow rate (kg/s)
- Oxidiser total mass (kg)
- Fuel total mass (kg)
- Nozzle throat and exit areas (m2)
- Number injector orifices
- Volume of combustion chamber (m3)

Provided that the following inputs are given:

- Nominal Thrust (N)
- Burn time (s)
- Combustion chamber pressure (Pascal)
- Oxidiser tank pressure (Pascal)
- Fuel tank pressure (Pascal)
- Heat capacity ratio of combustion products
- Molar weight of combustion products (kg/kmol)
- Adiabatic flame temperature (K)
- Oxidiser-fuel mass ratio
- Discharge coefficient of injector's orifices
- Diameter of injector's orifices (m)
- Combustion efficiency
- Nozzle expansion efficiency

Assumptions:

- Isentropic flow along the nozzle

- Temperature inside the combustion chamber equals the adiabatic flame temperature

- Combustion chamber is adiabatic

- Combustion products form a mixture which behaves like an ideal gas

OBS:

For determining the optimal oxidiser-fuel mass ratio, the highest theoretical Isp criterium was used.
The NASA-CEA software,through the Rocket CEA compiler, was used in the Google Colaboratory environment
to find the optimal O/F ratio in a given range, for a given pressure.

The heat capacity ratio, the adiabatic flame temperature and the molar mass of the combustion products
were also taken from the NASA-CEA code, from this optimal condition.

If one wants to change the combustion chamber pressure with respect to the default one (35 bar), it is
recommended to use yet again the Rocket CEA compiler. To do so, follow these steps:

1 - Open Google Colaboratory

https://colab.research.google.com

2 - Install Rocket CEA and libgfortran3

!pip install RocketCEA

!apt-get install libgfortran3

3 - Change the combustion chamber pressure in the following code

%%file chk_cea.py
from rocketcea.cea_obj import CEA_Obj
C = CEA_Obj( oxName='LOX', fuelName='CH4')
mr = 0;
highest_Isp = 0;
opt_mr = 0;
T_ad = 0;
M = 0;
opt_gamma = 0;

pc = 35; # combustion chamber pressure in bar

while mr < 10:
  IspVac, Cstar, Tc, MW, gamma = C.get_IvacCstrTc_ChmMwGam(Pc=pc*14.5038, MR=mr)
  if IspVac > highest_Isp:
    highest_Isp = IspVac
    opt_mr = mr
    T_ad = Tc
    M = MW
    opt_gamma = gamma
  mr += 0.1

print(highest_Isp) # Sec
print(opt_mr) # O/F ratio
print(T_ad*5/9) # Kelvin
print(M) # kg/kmol
print(opt_gamma) # Heat capacity ratio

4 - Run the file

!python chk_cea.py


"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import odeint
try:
    import CoolProp.CoolProp
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print('Unable to load CoolProp. CoolProp files will not be imported.')  
    
class Motor:
    
        def __init__ (self,
                      thrust = 1000,
                      burn_time = 10,
                      p_chamber = 35,
                      To = 3459,
                      Mol_Weight = 21.479,
                      k = 1.1275,
                      OFratio = 3.4,
                      n_cstar = 0.885,
                      n_cf = 0.95,
                      p_ox = 60,
                      p_fuel = 60,
                      cd_ox = 0.4,
                      cd_fuel = 0.4):
            
            """
            Thrust - Nominal Thrust,                                   N
            Burn time -                                                seconds
            p_chamber - Chamber pressure,                              bar
            To - Adiabatic flame temperature,                          K
            Mol_Weight - Molecular weight of exhaust products,         kg/mol
            k - Ratio of specific heats of exhaust products
            OFratio - Oxidiser/fuel mass ratio
            n_cstar - Combustion efficiency
            n_cf - Thrust coefficient efficiency
            p_ox - Pressure in the oxidiser tank,                      bar
            p_fuel - Pressure in the fuel tank,                        bar
            cd_ox - Discharge coefficient on oxidiser injector
            cd_fuel - Discharge coefficient on fuel injector
            
            """
            
    
            "INPUTS"

            # Environment
            self.g = 9.81    #"Gravitational Acceleration, m/s^2"
            self.Pa = 101325 #"Ambient Pressure, Pascal"
            self.Ta = 300    #"Ambient Temperature, Kelvin

            # Performance
            self.thrust = thrust;        #"Nominal Thrust, N"
            self.burn_time = burn_time;  #"Burn time, seconds"

            # Tanks
            self.p_ox = p_ox*10**5      #"Pressure on oxidiser tank, Pascal"
            self.p_fuel = p_fuel*10**5      #"Pressure on fuel tank, Pascal"

            # Injectors
            self.cd_ox = cd_ox          #"Discharge coefficient for LOX injector"
            self.cd_fuel = cd_fuel          #"Discharge coefficient for CH4 injector"

            # Combustion chamber
            self.p_chamber = p_chamber*10**5  #"Pressure on combustion chamber, Pascal"
            self.n_cstar = n_cstar       #"Combustion Efficiency"
            self.n_cf = n_cf           #"Thrust Coefficient Efficiency"
            self.k = k            #"Ratio of Specific Heats with respect to combustion products"
            self.M = Mol_Weight            #"Molar mass of combustion products, kg/kmol"
            self.To = To             #"Chamber temperature, Kelvin"
            self.mix_ratio = OFratio       #"Oxidiser/Fuel mass ratio"

            "COMPUTED PARAMETERS"

            self.rho_ox = PropsSI('D','T',90,'P',self.p_ox,'OXYGEN') # Oxidiser density, kg/m3

            self.rho_fuel = PropsSI('D','T',111,'P',self.p_fuel,'CH4') # Fuel density, kg/m3

            self.cstar = (8314*self.To*self.k/self.M)**0.5/self.k/((2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5 # Characteristic velocity, m/s

            self.cf = (((2*self.k**2)/(self.k-1))*(2/(self.k+1))**((self.k+1)/(self.k-1))*(1-(self.Pa/self.p_chamber)**((self.k-1)/self.k)))**0.5 # Thrust coefficient

            self.Isp = self.cstar*self.n_cstar*self.cf*self.n_cf/self.g # Specific impulse, seconds

            self.total_mass_flow = self.thrust/(self.Isp*self.g) # Total mass flow rate, kg/s

            self.ox_mass_flow = (self.mix_ratio/(self.mix_ratio+1))*self.total_mass_flow # Oxidiser mass flow rate, kg/s

            self.fuel_mass_flow = (1/(self.mix_ratio+1))*self.total_mass_flow # Fuel mass flow rate, kg/s

            self.ox_mass_total = self.ox_mass_flow*self.burn_time # Total oxidiser mass, kg

            self.fuel_mass_total = self.fuel_mass_flow*self.burn_time # Total fuel mass, kg

            self.throat_area = self.thrust/(self.p_chamber*self.cf*self.n_cf) # Throat area, m2

            self.throat_diameter = (4*self.throat_area/np.pi)**0.5 # Throat diameter, m

            self.exit_area = self.throat_area/( ((self.k+1)/2)**(1/(self.k-1)) * (self.Pa/self.p_chamber)**(1/self.k) * ( ((self.k+1)/(self.k-1))*(1-(self.Pa/self.p_chamber)**((self.k-1)/self.k) ) )**0.5 ) # Exit area, m2

            self.exit_diameter = (4*self.exit_area/np.pi)**0.5 # Exit diameter, m
            
            return None
        
        def report (self):
            
            print("Thrust (N): ",self.thrust)
            print()
            print("Burn time (seconds): ",self.burn_time)
            print()
            print("Chamber pressure (bar): ",self.p_chamber/10**5)
            print()
            print("Adiabatic chamber temperature (Kelvin): ",self.To)
            print()
            print("Molecular Weight of exhaust products (kg/kmol): ",self.M)
            print()
            print("Ratio of specific heats of exhaust products: ",self.k)
            print()
            print("Oxidiser/fuel mass ratio: ",self.mix_ratio)
            print()
            print("Combustion efficiency (%): ",self.n_cstar)
            print()
            print("Thrust coefficient efficiency (%): ",self.n_cf)
            print()
            print("Pressure on oxidiser tank (bar): ",self.p_ox/10**5)
            print()
            print("Pressure on fuel tank (bar): ",self.p_fuel/10**5)
            print()
            print("Characteristic velocity (m/s): ",self.cstar)
            print()
            print("Thrust coefficient: ",self.cf)
            print()
            print("Specific impulse (seconds): ",self.Isp)
            print()
            print("Total mass flow rate (kg/s): ",self.total_mass_flow)
            print()
            print("Oxidiser mass flow rate (kg/s): ",self.ox_mass_flow)
            print()
            print("Fuel mass flow rate (kg/s): ",self.fuel_mass_flow)
            print()
            print("Total oxidiser mass (kg): ",self.ox_mass_total)
            print()
            print("Total fuel mass (kg): ",self.fuel_mass_total)
            print()
            print("Nozzle throat diameter (mm): ",self.throat_diameter*10**3)
            print()
            print("Nozzle exit diameter (mm): ",self.exit_diameter*10**3)
            print()