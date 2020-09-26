""" This code computes some key design parameters of a liquid rocket engine,
    namely:

    TODO: Update description
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
    - Temperature inside the combustion chamber equals the adiabatic flame
    temperature
    - Combustion chamber is adiabatic
    - Combustion products form a mixture which behaves like an ideal gas
"""

# RocketCEA
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel, add_new_oxidizer, add_new_propellant
from rocketcea import blends

# CoolProp
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp

# Numpy
import numpy as np

# Scipy
# from scipy.optimize import minimize

# Matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl

# PrettyTables
from prettytable import PrettyTable

# Sys
import sys
import os


class Fluid:
    def __init__(
        self,
        name,
        coolprop_name,
        formula,
        fluid_type,
        storage_pressure=None,
        storage_temperature=None,
        storage_density=None,
        storage_enthalpy=None,
    ):
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
            Storage enthalpy in J/kg. If not given, it will be calculated using
            coolprop based on pressure and temperature given, considering
            liquid properties if state is phase change. Coolprop and NASA CEA
            use different enthalpy references. Therefore, this entalhpy will
            not be used in the input card for rocketcea. Thermo.lib will be
            used to calculate the entalphy based on the fluid formula.
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
            self.quality = 0  # Consider only liquid phase
            self.calc_pressure()
        elif self.storage_temperature is None:
            self.quality = 0  # Consider only liquid phase
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
        P = PropsSI("P", "Q", 0, "T", self.storage_temperature, self.coolprop_name)
        self.storage_pressure = P
        return P

    def calc_temperature(self):
        # Calculate temperature based on saturation curve
        T = PropsSI("T", "Q", 0, "P", self.storage_pressure, self.coolprop_name)
        self.storage_temperature = T
        return T

    def calc_density(self):
        # Calculate density using CoolProp
        if self.quality == 0:
            D = PropsSI("D", "Q", 0, "P", self.storage_pressure, self.coolprop_name)
        else:
            D = PropsSI(
                "D",
                "T",
                self.storage_temperature,
                "P",
                self.storage_pressure,
                self.coolprop_name,
            )
        # Save result
        self.storage_density = D
        return D

    def calc_enthalpy(self):
        # Calculate density using CoolProp
        if self.quality == 0:
            H = PropsSI("H", "Q", 0, "P", self.storage_pressure, self.coolprop_name)
        else:
            H = PropsSI(
                "H",
                "T",
                self.storage_temperature,
                "P",
                self.storage_pressure,
                self.coolprop_name,
            )
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
        self.card += (
            "rho={:.3f}".format(self.storage_density / 1000) + " "
        )  # convert density to g/cc
        if self.fluid_type == "oxidizer":
            self.card = "oxid " + self.card
            add_new_oxidizer(self.name, self.card)
        elif self.fluid_type == "fuel":
            self.card = "fuel " + self.card
            add_new_fuel(self.name, self.card)
        return self.card

    def __repr__(self):
        return self.name


class FluidMixture:
    def __init__(self, fluid1, x1, fluid2, x2=-1):
        """Class to hold and calculate thermodynamic properties of fluid
        mixtures. When a new object is created, a new card is also created to
        be used in rocketcea.
        
        Parameters
        ----------
        fluid1 : Fluid
        x1 : float
            Weigth percentage (0 - 100).
        fluid2 : Fluid
        x2 : float, optional
            Weigth percentage (0 - 100). Defalut is 100 - x1.

        Returns
        -------
        None

        Note
        ----
        Both fluids should have the same storage pressure and temperature.
        TODO: check if above restriction is really necessary.

        Both fluids shoud be of the same type, either oxidizer or fuel.

        Does not support fluid blends wh
        """

        # Save input data
        self.fluid1 = fluid1
        self.fluid2 = fluid2
        self.x1 = x1
        self.x2 = x2 if x2 != -1 else (1 - x1)

        # Check if storage pressure and temperature and type is the same
        if self.fluid1.storage_pressure != self.fluid2.storage_pressure:
            raise ValueError(
                "Fluid pressures do not match. "
                + "Fluid 1: {:.2f} K | Fluid 2: {:.2f} K".format(
                    self.fluid1.storage_pressure, self.fluid2.storage_pressure
                )
            )
        if self.fluid1.storage_temperature != self.fluid2.storage_temperature:
            raise ValueError(
                "Fluid temperatures do not match. "
                + "Fluid 1: {:.2f} K | Fluid 2: {:.2f} K".format(
                    self.fluid1.storage_temperature, self.fluid2.storage_temperature
                )
            )
        if self.fluid1.fluid_type != self.fluid2.fluid_type:
            raise ValueError(
                "Fluid types are not the same! Must be two oxidizers or two fuels."
            )
        # Save storage temperature, pressure and type
        self.fluid_type = self.fluid1.fluid_type
        self.storage_pressure = self.fluid1.storage_pressure
        self.storage_temperature = self.fluid1.storage_temperature

        # Create a new fluid name
        self.coolprop_name = self.fluid1.coolprop_name + "&" + self.fluid2.coolprop_name

        # Create a new rocketcea fluid blend
        if self.fluid_type == "fuel":
            self.name = blends.newFuelBlend(
                fuelL=[self.fluid1.name, self.fluid2.name],
                fuelPcentL=[self.x1, self.x2],
            )
        else:
            self.name = blends.newOxBlend(
                oxL=[self.fluid1.name, self.fluid2.name],
                oxPcentL=[self.x1, self.x2],
            )
        
        # Create HEOS CoolProp object
        self.HEOS = CoolProp.AbstractState("HEOS", self.coolprop_name)
        self.HEOS.set_mass_fractions([self.x1 / 100, self.x2 / 100])
        self.HEOS.update(
            CoolProp.PT_INPUTS, self.storage_pressure, self.storage_temperature
        )

        self.storage_density = self.HEOS.rhomass()

    def __repr__(self):
        return self.name


class Motor:
    def __init__(
        self,
        oxidizer,
        fuel,
        thrust=1000,
        burn_time=10,
        p_chamber=35,
        n_cstar=0.885,
        n_cf=0.95,
        cd_ox=0.4,
        cd_fuel=0.4,
        phi=None,
    ):

        """
        Motor preliminary design class.
        
        Parameters
        ----------
        oxidizer: Fluid or FluidMixture object
            Object from Fluid class with oxidizer properties.
            Example: LOX_PC = Fluid(name='O2(L)', coolprop_name='oxygen', 
                                    formula='O 2', fluid_type='oxidizer',
                                    storage_temperature=90)
        fuel: Fluid or FluidMixture object
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
        phi: float, optional
            Equivalence ratio. If None, which is default, this will be
            calculated to optimize specific impulse.
            phi = (fuel-to-oxidizer ratio) / (fuel-to-oxidizer ratio)st.
        """
        # -------------------------------Inputs--------------------------------#
        # Environment
        self.g = 9.81  # Gravitational Acceleration, m/s^2
        self.Pa = 1  # Ambient Pressure, bar
        self.Ta = 300  # Ambient Temperature, Kelvin

        # Oxidizer and fuel
        self.oxidizer = oxidizer
        self.rho_ox = oxidizer.storage_density  # Oxidizer density, kg/m3
        self.fuel = fuel
        self.rho_fuel = fuel.storage_density  # Fuel density, kg/m3

        # Performance
        self.thrust = thrust  # Nominal Thrust, N
        self.burn_time = burn_time  # Burn time, seconds

        # Injectors
        self.cd_ox = cd_ox  # Discharge coefficient for LOX injector
        self.cd_fuel = cd_fuel  # Discharge coefficient for CH4 injector

        # Combustion chamber
        self.p_chamber = p_chamber  # Pressure on combustion chamber, bar
        self.n_cstar = n_cstar  # Combustion Efficiency
        self.n_cf = n_cf  # Thrust Coefficient Efficiency

        #-------------------Computed Combustion Parameters--------------------#
        if phi is None:
            OF_ratio, gamma, molecular_weight, flame_temperature = self.optimize_combustion_parameters_at_chamber()
        else:
            OF_ratio, gamma, molecular_weight, flame_temperature = self.calculate_combustion_parameters_at_chamber(phi)
        
        self.OF_ratio = OF_ratio
        self.k = gamma
        self.M = molecular_weight
        self.To = flame_temperature

        #---------------------Computed Output Parameters----------------------#
        self.calculate_output_parameters()

        return None

    def calculate_combustion_parameters_at_chamber(self, phi):
        """ Uses RocketCEA, which wraps NASA CEA, to calculate 4 parameters at
        the combustion chamber:

            OF_ratio : Oxidizer/fuel mass ratio.
            k : exhaust gasses cp/cv = gamma.
            M : exhaust gasses moleculaer weight.
            To : adiabatic flame temperature.

        Parameters
        ----------
        phi : float
            Equivalence ratio: phi = (fuel-to-oxidizer ratio) /
            (fuel-to-oxidizer ratio)st.

        Returns
        -------
        OF_ratio : float
            Oxidizer/fuel mass ratio based on given phi.
        gamma : float
            Exhaust gasses cp/cv.
        molecular_weight : float
            Exhaust gasses moleculaer weight.
        flame_temperature : float
            Adiabatic flame temperature.
        """
        cea_analysis = CEA_Obj(
            oxName=self.oxidizer.name,
            fuelName=self.fuel.name,
            isp_units="sec",
            cstar_units="m/sec",
            pressure_units="bar",
            temperature_units="K",
            sonic_velocity_units="m/sec",
            enthalpy_units="kJ/kg",
            density_units="kg/m^3",
            specific_heat_units="kJ/kg-K",
            viscosity_units="millipoise",
            thermal_cond_units="W/cm-degC",
        )

        # Get OF ratio based on equivalence ratio
        OF_ratio = cea_analysis.getMRforER(ERphi=phi)

        # Calculate outputs
        area_ratio = cea_analysis.get_eps_at_PcOvPe(
            Pc=self.p_chamber, MR=OF_ratio, PcOvPe=self.p_chamber / self.Pa
        )
        (
            Isp,
            cstar,
            flame_temperature,
            molecular_weight,
            gamma,
        ) = cea_analysis.get_IvacCstrTc_ChmMwGam(
            Pc=self.p_chamber, MR=OF_ratio, eps=area_ratio
        )

        return OF_ratio, gamma, molecular_weight, flame_temperature

    def optimize_combustion_parameters_at_chamber(self):
        """ Uses RocketCEA, which wraps NASA CEA, to calculate 4 parameters at
        the combustion chamber:

            OF_ratio : Oxidizer/fuel mass ratio.
            k : exhaust gasses cp/cv = gamma.
            M : exhaust gasses moleculaer weight.
            To : adiabatic flame temperature.

        OF_ratio is calculated by optimizing the specific impulse of the motor.

        Returns
        -------
        optimal_OF_ratio : float
            Optimized oxidizer/fuel mass ratio generating optimum specific
            impulse.
        gamma : float
            Exhaust gasses cp/cv.
        molecular_weight : float
            Exhaust gasses moleculaer weight.
        flame_temperature : float
            Adiabatic flame temperature.
        """
        # ---------------Calculated Inputs----------------#
        # Calculate input parameters from oxidiser and fuel combustion
        # Initialize CEA analysis
        cea_analysis = CEA_Obj(
            oxName=self.oxidizer.name,
            fuelName=self.fuel.name,
            # Units
            isp_units="sec",
            cstar_units="m/sec",
            pressure_units="bar",
            temperature_units="K",
            sonic_velocity_units="m/sec",
            enthalpy_units="kJ/kg",
            density_units="kg/m^3",
            specific_heat_units="kJ/kg-K",
            viscosity_units="millipoise",
            thermal_cond_units="W/cm-degC",
        )

        # Get stoichiometric equivalence ratio
        stoichiometric_OF_ratio = cea_analysis.getMRforER(ERphi=1.0)

        # Find optimal specific impulse
        def minus_specific_impulse_function(OF_ratio):
            area_ratio = cea_analysis.get_eps_at_PcOvPe(
                Pc=self.p_chamber, MR=OF_ratio, PcOvPe=self.p_chamber / self.Pa
            )
            Isp, cstar, To, M, k = cea_analysis.get_IvacCstrTc_ChmMwGam(
                Pc=self.p_chamber, MR=OF_ratio, eps=area_ratio
            )
            return -Isp

        res = minimize(
            fun=minus_specific_impulse_function,
            x0=stoichiometric_OF_ratio,
            method="Powell",
            bounds=[[stoichiometric_OF_ratio * 0.5, stoichiometric_OF_ratio * 2]],
        )

        # Optimization sequence complete, store results
        optimal_OF_ratio = res.x[0]
        optimal_area_ratio = cea_analysis.get_eps_at_PcOvPe(
            Pc=self.p_chamber, MR=optimal_OF_ratio, PcOvPe=self.p_chamber / self.Pa
        )
        (
            optimal_Isp,
            cstar,
            flame_temperature,
            molecular_weight,
            gamma,
        ) = cea_analysis.get_IvacCstrTc_ChmMwGam(
            Pc=self.p_chamber, MR=optimal_OF_ratio, eps=optimal_area_ratio
        )

        return optimal_OF_ratio, gamma, molecular_weight, flame_temperature

    def calculate_output_parameters(self):
        """Calculate output parameters such as:
            propellant_storage_density
            cstar
            cf
            Isp
            Iv
            total_mass_flow
            ox_mass_flow
            fuel_mass_flow
            ox_mass_total
            fuel_mass_total
            throat_area
            throat_diameter
            exit_area
            exit_diameter
        """
        self.propellant_storage_density = (self.OF_ratio + 1) / (
            self.OF_ratio / self.oxidizer.storage_density
            + 1 / self.fuel.storage_density
        )

        # Characteristic velocity, m/s
        self.cstar = (
            (8314 * self.To * self.k / self.M) ** 0.5
            / self.k
            / (((2 / (self.k + 1)) ** ((self.k + 1) / (self.k - 1))) ** 0.5)
        )

        # Thrust coefficient
        self.cf = (
            ((2 * self.k ** 2) / (self.k - 1))
            * (2 / (self.k + 1)) ** ((self.k + 1) / (self.k - 1))
            * (1 - (self.Pa / self.p_chamber) ** ((self.k - 1) / self.k))
        ) ** 0.5

        # Specific impulse, seconds
        self.Isp = self.cstar * self.n_cstar * self.cf * self.n_cf / self.g

        # Volumetric specific impulse, kg/(s*m^2)
        self.Iv = self.Isp * self.g * self.propellant_storage_density

        # Total mass flow rate, kg/s
        self.total_mass_flow = self.thrust / (self.Isp * self.g)

        # Oxidiser mass flow rate, kg/s
        self.ox_mass_flow = (self.OF_ratio / (self.OF_ratio + 1)) * self.total_mass_flow

        # Fuel mass flow rate, kg/s
        self.fuel_mass_flow = (1 / (self.OF_ratio + 1)) * self.total_mass_flow

        # Total oxidiser mass, kg
        self.ox_mass_total = self.ox_mass_flow * self.burn_time

        # Total fuel mass, kg
        self.fuel_mass_total = self.fuel_mass_flow * self.burn_time

        # Throat area, m2
        self.throat_area = self.thrust / (
            self.p_chamber * 10 ** 5 * self.cf * self.n_cf
        )

        # Throat diameter, m
        self.throat_diameter = (4 * self.throat_area / np.pi) ** 0.5

        # Exit area, m2
        self.exit_area = self.throat_area / (
            ((self.k + 1) / 2) ** (1 / (self.k - 1))
            * (self.Pa / self.p_chamber) ** (1 / self.k)
            * (
                ((self.k + 1) / (self.k - 1))
                * (1 - (self.Pa / self.p_chamber) ** ((self.k - 1) / self.k))
            )
            ** 0.5
        )

        # Exit diameter, m
        self.exit_diameter = (4 * self.exit_area / np.pi) ** 0.5

        return None

    def report(self):
        print("Thrust (N): {:.2f}".format(self.thrust))
        print()
        print("Burn time (seconds): {:.2f}".format(self.burn_time))
        print()
        print("Chamber pressure (bar): {:.1f}".format(self.p_chamber))
        print()
        print("Adiabatic chamber temperature (Kelvin): {:.1f}".format(self.To))
        print()
        print("Molecular Weight of exhaust products (kg/kmol): {:.2f}".format(self.M))
        print()
        print("Ratio of specific heats of exhaust products: {:.2f}".format(self.k))
        print()
        print("Oxidiser/fuel mass ratio: {:.2f}".format(self.OF_ratio))
        print()
        print("Combustion efficiency (%): {:.2f}".format(self.n_cstar))
        print()
        print("Thrust coefficient efficiency (%): {:.2f}".format(self.n_cf))
        print()
        print(
            "Pressure on oxidiser tank (bar): {:.2f}".format(
                self.oxidizer.storage_pressure / 10 ** 5
            )
        )
        print()
        print(
            "Temperature on oxidiser tank (K): {:.2f}".format(
                self.oxidizer.storage_temperature
            )
        )
        print()
        print(
            "Pressure on fuel tank (bar): {:.2f}".format(
                self.fuel.storage_pressure / 10 ** 5
            )
        )
        print()
        print(
            "Temperature on fuel tank (K): {:.2f}".format(self.fuel.storage_temperature)
        )
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
        print(
            "Nozzle throat diameter (mm): {:.1f}".format(self.throat_diameter * 10 ** 3)
        )
        print()
        print("Nozzle exit diameter (mm): {:.1f}".format(self.exit_diameter * 10 ** 3))
        print()

    def report_ptable(self):
        
        performance_tab = PrettyTable()
        geometric_tab = PrettyTable()
        injector_tab = PrettyTable()

        performance_tab.field_names = ["Performance parameters", "Value", "Units"]
        injector_tab.field_names = ["Injector parameters", "Value", "Units"]
        geometric_tab.field_names = ["Geometric parameters", "Value", "Units"]

        performance_tab.add_row(["Thrust",round(self.thrust,2), 'N'])
        performance_tab.add_row(["Burn time", round(self.burn_time,2), 'Seconds'])
        performance_tab.add_row(["Chamber pressure", round(self.p_chamber,2), 'bar'])
        performance_tab.add_row(["Adiabatic chamber temperature", round(self.To,2), 'K'])
        performance_tab.add_row(["Molecular Weight of exhaust products", round(self.M,2), 'kg/kmol'])
        performance_tab.add_row(["Ratio of specific heats of exhaust products", round(self.k,2), '-'])
        performance_tab.add_row(["Combustion efficiency", round(100*self.n_cstar,2), '%'])
        performance_tab.add_row(["Thrust coefficient efficiency", round(100*self.n_cf,2), '%'])
        injector_tab.add_row(["Pressure on oxidiser tank", round(self.oxidizer.storage_pressure/10**5,2), 'bar'])
        injector_tab.add_row(["Temperature on oxidiser tank", round(self.oxidizer.storage_temperature,2), 'K'])
        injector_tab.add_row(["Pressure on fuel tank", round(self.fuel.storage_pressure/10**5,2), 'bar'])
        injector_tab.add_row(["Temperature on fuel tank", round(self.fuel.storage_temperature,2), 'K'])
        performance_tab.add_row(["Characteristic velocity", round(self.cstar,2), 'm/s'])
        performance_tab.add_row(["Thrust coefficient", round(self.cf,2), '-'])
        performance_tab.add_row(["Specific impulse", round(self.Isp,2), 'Seconds'])
        performance_tab.add_row(["Volumetric Specific impulse", round(self.Iv,2), 'Ns/m³'])
        injector_tab.add_row(["Total mass flow rate", round(self.total_mass_flow,3), 'kg/s'])
        injector_tab.add_row(["Oxidiser mass flow rate", round(self.ox_mass_flow,3), 'kg/s'])
        injector_tab.add_row(["Fuel mass flow rate", round(self.fuel_mass_flow,3), 'kg/s'])
        injector_tab.add_row(["Total oxidiser mass", round(self.ox_mass_total,3), 'kg'])
        injector_tab.add_row(["Total fuel mass", round(self.fuel_mass_total,3), 'kg'])
        geometric_tab.add_row(["Nozzle throat diameter", round(self.throat_diameter*10**3,1), 'mm'])
        geometric_tab.add_row(["Nozzle exit diameter", round(self.exit_diameter*10**3,1), 'mm'])

        print("PERFORMANCE PARAMETERS")
        print()
        print(performance_tab)
        print()
        print("INJECTOR PARAMETERS")
        print()
        print(injector_tab)
        print()
        print("GEOMETRIC PARAMETERS")
        print()
        print(geometric_tab)

    def print_cea_output(self, subar=None, supar=None):
        """Prints NASA CEA output file."""
        cea_analysis = CEA_Obj(
            oxName=self.oxidizer.name,
            fuelName=self.fuel.name,
            isp_units="sec",
            cstar_units="m/sec",
            pressure_units="bar",
            temperature_units="K",
            sonic_velocity_units="m/sec",
            enthalpy_units="kJ/kg",
            density_units="kg/m^3",
            specific_heat_units="kJ/kg-K",
            viscosity_units="millipoise",
            thermal_cond_units="W/cm-degC",
        )

        area_ratio = cea_analysis.get_eps_at_PcOvPe(
            Pc=self.p_chamber, MR=self.OF_ratio, PcOvPe=self.p_chamber / self.Pa
        )

        cea_output = cea_analysis.cea_obj.get_full_cea_output(
            Pc=self.p_chamber,
            MR=self.OF_ratio,
            PcOvPe=self.p_chamber / self.Pa,
            subar=subar,
            eps=supar,
            show_transport=1,
            pc_units="bar",
            output="siunits",
            show_mass_frac=True,
            frozen=0,
            frozenAtThroat=1,
            #fac_CR=2,
        )
        print(cea_output)

    def value_cea_output(self, frozen, subar, supar):
        """Return a list containing values for a certain parameter of the NASA CEA txt output
           The parameter is computed at different sections of the combustion chamber/nozzle, according to the input file,
           which leads to different values

        T - Temperature, K
        gamma - Ratio of specific heats
        visc - Viscosity, milipoise
        cond - Conductivity, MILLIWATTS/(CM)(K)
        Pr - Prandtl Number

        frozen - Conductivity and Prandtl Number are available for frozen or equilibrium assumptions
                 True or False  
        """
        
        try:
            os.remove('cea_output.txt')
        except:
            pass

        orig_stdout = sys.stdout
        f = open('cea_output.txt', 'w')
        sys.stdout = f

        self.print_cea_output(subar, supar)

        sys.stdout = orig_stdout
        f.close()

        number = ''

        values = []

        conductivity_counter = 0
        end_prandtl = False
        end_cond = False

        temperature = []
        density = []
        gamma = []
        viscosity = []
        conductivity = []
        prandtl = []

        with open('cea_output.txt') as f:
            for line in f:
                if 'T, K' in line:
                    for i in range(len(line)):
                        try:
                            if line[i] == '.':
                                number = number + line[i]
                            else:    
                                isnumber = int(line[i])
                                number = number + line[i]
                        except:
                            if not number:
                                pass
                            else:
                                temperature.append(float(number))
                                number = ''
                if 'RHO, KG/CU M' in line:
                    for i in range(len(line)):
                        try:
                            if line[i] == '.':
                                number = number + line[i]
                            else:    
                                isnumber = int(line[i])
                                number = number + line[i]
                        except:
                            if not number:
                                pass
                            else:
                                density.append(float(number))
                                number = ''                          
                elif 'GAMMAs' in line:
                    for i in range(len(line)):
                        try:
                            if line[i] == '.':
                                number = number + line[i]
                            else:    
                                isnumber = int(line[i])
                                number = number + line[i]
                        except:
                            if not number:
                                pass
                            else:
                                gamma.append(float(number))
                                number = ''
                elif 'VISC,MILLIPOISE' in line:
                    for i in range(len(line)):
                        try:
                            if line[i] == '.':
                                number = number + line[i]
                            else:    
                                isnumber = int(line[i])
                                number = number + line[i]
                        except:
                            if not number:
                                pass
                            else:
                                viscosity.append(float(number))
                                number = ''
                elif 'CONDUCTIVITY' in line:
                    if conductivity_counter == 0:
                        conductivity_counter = 1
                        continue    
                    if frozen == True:
                        frozen = False
                        continue
                    elif end_cond == False:
                        for i in range(len(line)):
                            try:
                                if line[i] == '.':
                                    number = number + line[i]
                                else:    
                                    isnumber = int(line[i])
                                    number = number + line[i]
                            except:
                                if not number:
                                    pass
                                else:
                                    conductivity.append(float(number))
                                    number = ''
                        end_cond = True
                elif 'PRANDTL NUMBER' in line: 
                    if frozen == True:
                        frozen = False
                        continue
                    elif end_prandtl == False:
                        for i in range(len(line)):
                            try:
                                if line[i] == '.':
                                    number = number + line[i]
                                else:    
                                    isnumber = int(line[i])
                                    number = number + line[i]
                            except:
                                if not number:
                                    pass
                                else:
                                    prandtl.append(float(number))
                                    number = ''
                        end_prandtl = True

        #TODO - Reading density
        new_density = []
        for i in range(len(density)-1):
            if i % 2 == 0:
                new_density.append(density[i]*10**(-density[i+1]))
            else:
                continue

        return temperature, new_density, gamma, viscosity, conductivity, prandtl

    def calculate_transport_properties(self, radius_mesh):
        """
        radius_mesh - vector with radius
        Return [rho, visc, prandtl, conductivity]
        """
        # Calculate areas
        throat_index = np.argmin(radius_mesh)
        throat_radius = radius_mesh[throat_index]
        area_mesh = np.pi*np.array(radius_mesh)**2
        area_ratios_mesh = area_mesh/(np.pi*throat_radius**2)

        # Separate sub and supersonic area ratios
        subar = area_ratios_mesh[:throat_index]
        supar = area_ratios_mesh[throat_index:]



        # Get transport properties
        temperature, density, gamma, viscosity, conductivity, prandtl = self.value_cea_output(frozen=False, subar=subar, supar=supar)

        density = density[3:]
        viscosity = viscosity[3:]
        prandtl = prandtl[3:]
        conductivity = conductivity[3:]

        return density, viscosity, prandtl, conductivity         

# Good for debugging in VSCode
if __name__ == "__main__":
    # Oxidizer
    NOX = Fluid(
        name="N2O",
        coolprop_name="NitrousOxide",
        formula=None,
        fluid_type="oxidizer",
        storage_temperature=298.15,
    )

    # Fuels
    H2O = Fluid(
        name="H2O(L)",
        coolprop_name="water",
        formula="H 2 O 1",
        fluid_type="fuel",
        storage_pressure=35e5,
        storage_temperature=298.15,
    )

    LC2H5OH = Fluid(
        name="C2H5OH(L)",
        coolprop_name="ethanol",
        formula="C 2 H 6 O 1",
        fluid_type="fuel",
        storage_pressure=35e5,
        storage_temperature=298.15,
    )

    H2O_30_C2H50H_70 = FluidMixture(fluid1=LC2H5OH, x1=70, fluid2=H2O, x2=30)

    NOELLE = Motor(
        NOX,
        LC2H5OH,
        thrust=1500,
        burn_time=10,
        p_chamber=35,
        n_cstar=1,
        n_cf=1,
        cd_ox=0.6,
        cd_fuel=0.182,
        phi=1,
    )
    
    density, viscosity, prandtl, conductivity = NOELLE.calculate_transport_properties(np.array([1.5, 1.3, 1, 1.4, 1.6])**0.5/np.pi)
    print()