# RocketCEA
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_oxidizer, add_new_propellant
from rocketcea.biprop_utils.rho_isp_plot_obj import RhoIspPlot
# CoolProp
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp
# Numpy
import numpy as np
# Matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl

class fluid:
    def __init__(self,
                 name,
                 coolprop_name,
                 formula,
                 fluid_type, 
                 storage_pressure=None, 
                 storage_temperature=None,
                 storage_density=None,
                 storage_enthalpy=None):
        """Class to hold and calculate thermodynamic properties of
        oxidizers and fuels. When a new object is created, a new
        card is also created to be used in rocketcea.
        
        Parameters
        ----------
        name: string
            Name used to specify the fluid in rocketcea.
        coolprop_name: string
            Name used to get thermodynamic data from coolprop
        formula: string
            Chemical formula used to specify fluid in rocketcea,
            if data from thermo.lib is desired. Example: formula
            for CH4 would be: 'C 1 H 4'. If none, no cea card
            will be created.
        fluid_type: string
            Either 'oxidizer' or 'fuel'.
        storage_pressure: float
            Storage pressure in Pa. If not give, it will automatically
            be calculated considering saturation conditions (phase change),
            using coolprop. Either storage_temperature or presure must be given!
        storage_temperature: float, optional
            Storage temperature in K. If not given, it will automatically
            be calculated considering saturation conditions (phase change),
            using coolprop. Either storage_temperature or presure must be given!
        storage_density: float, optional
            Storage density in kg/m³. If not given, it will automatically
            be calculated based on pressure and temperature given. If the
            fluid is in phase change regime, liquid density will be considered.
        storage_enthalpy: float, optional
            Storage enthalpy in J/kg. If not given, it will be calculated using coolprop
            based on pressure and temperature given, considering liquid properties
            if state is phase change. Coolprop and NASA CEA use different enthalpy
            references. Therefore, this entalhpy will not be used in the input card
            for rocketcea. Thermo.lib will be used to calculate the entalphy based on
            the fluid formula.
        """
        
        # Save input data
        self.name = name
        self.coolprop_name = coolprop_name
        self.formula = formula
        self.fluid_type = fluid_type
        self.storage_pressure = storage_pressure
        self.storage_temperature = storage_temperature
        self.storage_density = storage_density
        self.storage_enthalpy = storage_enthalpy
        self.quality = None
        
        # Calculate relevant thermodynamic properties
        if self.storage_pressure is None:
            self.quality = 0 # Consider only liquid phase
            self.calc_pressure()        
        elif self.storage_temperature is None:
            self.quality = 0 # Consider only liquid phase
            self.calc_temperature()
        
        if self.storage_density is None:
            self.calc_density()
        
        if self.storage_enthalpy is None:
            self.calc_enthalpy()
        
        # Create CEA card
        self.card = ""
        if self.formula != None:
            self.cea_card()
    
    def calc_pressure(self):
        # Calculate pressure based on saturation curve
        P = PropsSI("P",
                    "Q", 0,
                    "T", self.storage_temperature,
                    self.coolprop_name)
        self.storage_pressure = P
        return P
    
    def calc_temperature(self):
        # Calculate temperature based on saturation curve
        T = PropsSI("T",
                    "Q", 0,
                    "P", self.storage_pressure,
                    self.coolprop_name)
        self.storage_temperature = T
        return T
    
    def calc_density(self):
        # Calculate density using CoolProp
        if self.quality == 0:
            D = PropsSI("D",
                        "Q", 0,
                        "P", self.storage_pressure,
                        self.coolprop_name)
        else:
            D = PropsSI("D",
                        "T", self.storage_temperature,
                        "P", self.storage_pressure,
                        self.coolprop_name)
        # Save result
        self.storage_density = D
        return D
    
    def calc_enthalpy(self):
        # Calculate density using CoolProp
        if self.quality == 0:
            H = PropsSI("H",
                        "Q", 0,
                        "P", self.storage_pressure,
                        self.coolprop_name)
        else:            
            H = PropsSI("H",
                        "T", self.storage_temperature,
                        "P", self.storage_pressure,
                        self.coolprop_name)
        # Save result
        self.storage_enthalpy = H
        return H
    
    def cea_card(self):
        # Create cea card and rocketcea fuel/oxidizer
        self.card += self.name + " "
        self.card += self.formula + " "
        self.card += "wt%=100" + " "
        # self.card += "h,cal={:.3f}".format(self.storage_enthalpy/4.184) + " "
        self.card += "t(k)={:.3f}".format(self.storage_temperature) + " "
        self.card += "rho={:.3f}".format(self.storage_density/1000) + " " # convert density to g/cc
        if self.fluid_type == "oxidizer":
            self.card = "oxid " + self.card
            add_new_oxidizer(self.name, self.card)
        elif self.fluid_type == "fuel":
            self.card = "fuel " + self.card
            add_new_fuel(self.name, self.card)
        return self.card 
    
    def __repr__(self):
        return self.name

class Motor:
    def __init__ (self,
                    oxidizer,
                    fuel,
                    thrust = 1000,
                    burn_time = 10,
                    p_chamber = 35,
                    n_cstar = 0.885,
                    n_cf = 0.95,
                    cd_ox = 0.4,
                    cd_fuel = 0.4,
                    suboptimal=1):
        
        """
        Motor preliminary design class.
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
        
        Parameters
        ----------
        oxidizer: Fluid object
            Object from Fluid class with oxidizer properties.
            Example: LOX_PC = fluid(name='O2(L)', coolprop_name='oxygen', 
                                    formula='O 2', fluid_type='oxidizer',
                                    storage_temperature=90)
        fuel: Fluid object
            Object from Fluid class with fuel properties.
            Example: LCH4_PC = fluid(name='CH4(L)', coolprop_name='methane',
                                        formula='C 1 H 4', fluid_type='fuel',
                                        storage_temperature=112)
        thrust: float
            Nominal desired thrust (N).
        burn_time: float
            Nominal desired motor burn time (s).
        p_chamber: float
            Chamber pressure (bar).
        n_cstar: float
            Combustion efficiency. Ranges from 0 to 1 (1 is better).
        n_cf: float
            Thrust coefficient efficiency. Ranges from 0 to 1 (1 is better).
        cd_ox: float
            Discharge coefficient on oxidiser injector. (No units)
        cd_fuel: float
            Discharge coefficient on fuel injector. (No units)
        
        Calculated Parameters
        ---------------------
        self.To: float
            Adiabatic flame temperature (K).
        self.OFratio: float
            Oxidiser to fuel mass ratio for optimal Isp.
        self.k: float
            Ratio of specific heats of exhaust products in case 
            of optimal Isp.
        self.Mol_Weight: float
            Molecular weight of exhaust products (kg/kmol) in case
            of optimal Isp.            
        """
        

        #---------------Inputs----------------#
        # Environment
        self.g = 9.81    # Gravitational Acceleration, m/s^2
        self.Pa = 101325 # Ambient Pressure, Pascal
        self.Ta = 300    # Ambient Temperature, Kelvin

        # Oxidizer and fuel
        self.oxidizer = oxidizer
        self.rho_ox = oxidizer.storage_density # Oxidizer density, kg/m3
        self.fuel = fuel
        self.rho_fuel = fuel.storage_density   # Fuel density, kg/m3
        
        # Performance
        self.thrust = thrust        # Nominal Thrust, N
        self.burn_time = burn_time  # Burn time, seconds

        # Injectors
        self.cd_ox = cd_ox          # Discharge coefficient for LOX injector
        self.cd_fuel = cd_fuel      # Discharge coefficient for CH4 injector

        # Combustion chamber
        self.p_chamber = p_chamber*10**5             # Pressure on combustion chamber, Pascal
        self.p_chamber_psi = 14.5038*p_chamber  # Pressure on combustion chamber, Pascal
        self.n_cstar = n_cstar                       # Combustion Efficiency
        self.n_cf = n_cf                             # Thrust Coefficient Efficiency
        
        #---------------Calculated Inputs----------------#
        # Calculate input parameters from oxidiser and fuel combustion
        # Initialize CEA analysis
        cea_analysis = CEA_Obj(oxName=self.oxidizer.name, fuelName=self.fuel.name)
        
        # print(cea_analysis.get_full_cea_output(Pc=self.p_chamber_psi, MR=2))
        
        # Initialize Isp optimization sequence
        min_OFratio = 0.1     # Minimum Oxidizer to Fuel ratio to test
        max_OFratio = 10      # Maximum Oxidizer to Fuel ratio to test
        samples_OFratio = 100 # Number of Oxider to Fuel ratios to test
        
        # Initialize optimum values
        optimum_Isp = 0
        optimum_OFratio = 0
        optimum_To = 0
        optimum_M = 0
        optimum_k = 0

        for test_OFratio in np.linspace(min_OFratio, max_OFratio, samples_OFratio):
            # Get combustion results from cea analysis for given test_OFratio
            eps = cea_analysis.get_eps_at_PcOvPe(Pc=self.p_chamber_psi, MR=test_OFratio,PcOvPe=self.p_chamber/self.Pa)
            Isp, cstar, To, M, k = cea_analysis.get_IvacCstrTc_ChmMwGam(Pc=self.p_chamber_psi, MR=test_OFratio, eps=eps)
            # Check if Isp for this test_OFratio is maximum relative to previous
            if Isp > optimum_Isp:
                # Isp is biggest yet -> record data
                optimum_Isp = Isp
                optimum_OFratio = test_OFratio
                optimum_To = To
                optimum_M = M
                optimum_k = k
        
        # Optimization sequence done, store results
        self.To = optimum_To*5/9
        self.OFratio = optimum_OFratio
        self.k = optimum_k
        self.M = optimum_M
        
        # Suboptimal conditions
        if suboptimal != 1:
            self.OFratio *= suboptimal
            eps = cea_analysis.get_eps_at_PcOvPe(Pc=self.p_chamber_psi, MR=self.OFratio,PcOvPe=self.p_chamber/self.Pa)
            Isp, cstar, To, M, k = cea_analysis.get_IvacCstrTc_ChmMwGam(Pc=self.p_chamber_psi, MR=self.OFratio, eps=eps)
            self.To = To*5/9
            self.k = k
            self.M = M

        #---------------Computed Output Parameters----------------#
        self.propellant_storage_density = (self.OFratio + 1)/(self.OFratio/self.oxidizer.storage_density + 1/self.fuel.storage_density)
        
        self.cstar = (8314*self.To*self.k/self.M)**0.5/self.k/((2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5 # Characteristic velocity, m/s

        self.cf = (((2*self.k**2)/(self.k-1))*(2/(self.k+1))**((self.k+1)/(self.k-1))*(1-(self.Pa/self.p_chamber)**((self.k-1)/self.k)))**0.5 # Thrust coefficient

        self.Isp = self.cstar*self.n_cstar*self.cf*self.n_cf/self.g # Specific impulse, seconds

        self.Iv = self.Isp*self.g*self.propellant_storage_density
        
        self.total_mass_flow = self.thrust/(self.Isp*self.g) # Total mass flow rate, kg/s

        self.ox_mass_flow = (self.OFratio/(self.OFratio+1))*self.total_mass_flow # Oxidiser mass flow rate, kg/s

        self.fuel_mass_flow = (1/(self.OFratio+1))*self.total_mass_flow # Fuel mass flow rate, kg/s

        self.ox_mass_total = self.ox_mass_flow*self.burn_time # Total oxidiser mass, kg

        self.fuel_mass_total = self.fuel_mass_flow*self.burn_time # Total fuel mass, kg

        self.throat_area = self.thrust/(self.p_chamber*self.cf*self.n_cf) # Throat area, m2

        self.throat_diameter = (4*self.throat_area/np.pi)**0.5 # Throat diameter, m

        self.exit_area = self.throat_area/( ((self.k+1)/2)**(1/(self.k-1)) * (self.Pa/self.p_chamber)**(1/self.k) * ( ((self.k+1)/(self.k-1))*(1-(self.Pa/self.p_chamber)**((self.k-1)/self.k) ) )**0.5 ) # Exit area, m2

        self.exit_diameter = (4*self.exit_area/np.pi)**0.5 # Exit diameter, m
        
        return None
    

    def report (self):
        print("Thrust (N): {:.2f}".format(self.thrust))
        print()
        print("Burn time (seconds): {:.2f}".format(self.burn_time))
        print()
        print("Chamber pressure (bar): {:.1f}".format(self.p_chamber/10**5))
        print()
        print("Adiabatic chamber temperature (Kelvin): {:.1f}".format(self.To))
        print()
        print("Molecular Weight of exhaust products (kg/kmol): {:.2f}".format(self.M))
        print()
        print("Ratio of specific heats of exhaust products: {:.2f}".format(self.k))
        print()
        print("Oxidiser/fuel mass ratio: {:.2f}".format(self.OFratio))
        print()
        print("Combustion efficiency (%): {:.2f}".format(self.n_cstar))
        print()
        print("Thrust coefficient efficiency (%): {:.2f}".format(self.n_cf))
        print()
        print("Pressure on oxidiser tank (bar): {:.2f}".format(self.oxidizer.storage_pressure/10**5))
        print()
        print("Temperature on oxidiser tank (K): {:.2f}".format(self.oxidizer.storage_temperature))
        print()
        print("Pressure on fuel tank (bar): {:.2f}".format(self.fuel.storage_pressure/10**5))
        print()
        print("Temperature on fuel tank (K): {:.2f}".format(self.fuel.storage_temperature))
        print()
        print("Characteristic velocity (m/s): {:.2f}".format(self.cstar))
        print()
        print("Thrust coefficient: {:.2f}".format(self.cf))
        print()
        print("Specific impulse (seconds): {:.2f}".format(self.Isp))
        print()
        print("Volumetric Specific impulse (Ns/m³): {:.2f}".format(self.Iv))
        print()
        print("Total mass flow rate (kg/s): {:.3f}".format(self.total_mass_flow))
        print()
        print("Oxidiser mass flow rate (kg/s): {:.3f}".format(self.ox_mass_flow))
        print()
        print("Fuel mass flow rate (kg/s): {:.3f}".format(self.fuel_mass_flow))
        print()
        print("Total oxidiser mass (kg): {:.3f}".format(self.ox_mass_total))
        print()
        print("Total fuel mass (kg): {:.3f}".format(self.fuel_mass_total))
        print()
        print("Nozzle throat diameter (mm): {:.1f}".format(self.throat_diameter*10**3))
        print()
        print("Nozzle exit diameter (mm): {:.1f}".format(self.exit_diameter*10**3))
        print()