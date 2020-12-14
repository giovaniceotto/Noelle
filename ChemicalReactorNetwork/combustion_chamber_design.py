import numpy as np
# np.set_printoptions(precision=2)
import scipy.integrate
import scipy.signal
from rocketpy import Function
import CoolProp.CoolProp as CoolProp
from CoolProp.CoolProp import PropsSI
import cantera as ct
print('Runnning Cantera version: ' + ct.__version__)


# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = [10.0, 6.0]
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
font = {'weight' : 'bold',
         'size'   : 17}
mpl.rc('font', **font)
plt.style.use(['science', 'grid'])

def create_solution_mechanism():
    # Defining Reaction Mechanism
    # Mechanism II - Marinov + Mevel
    marinov_species = ct.Species.listFromFile('marinov_ethanol_mechanism.cti')
    marinov_reactions = ct.Reaction.listFromFile('marinov_ethanol_mechanism.cti')

    mevel_species = ct.Species.listFromFile('mevel_ethanol_mechanism.cti')
    mevel_reactions = ct.Reaction.listFromFile('mevel_ethanol_mechanism.cti')

    new_species = []
    new_reactions = []

    # Filter species
    for specie in mevel_species:
        # Include all nitrogen compounds except for N2
        if 'N' in specie.composition and specie.composition != {'N':2}: new_species.append(specie)

    new_species_names = {specie.name for specie in new_species}
    # print('N based species: {0}'.format(', '.join(name for name in new_species_names)))

    marinov_mevel_species = marinov_species + new_species
    marinov_mevel_species_names = {specie.name.upper() for specie in marinov_mevel_species}

    # Filter reactions, keeping only those that only involve the selected species
    # print('\nReactions:')
    for R in mevel_reactions:
        if any(reactant in new_species_names for reactant in R.reactants) or any(product in new_species_names for product in R.products):
            # for reactant in R.reactants:
                # if reactant not in marinov_mevel_species_names:
                    # print('Missing reactant:', reactant, 'when analyzing reaction', R.equation)
            # for product in R.products:
                # if product not in marinov_mevel_species_names:
                    # print('Missing product:', product, 'when analyzing reaction', R.equation)
            if all(reactant in marinov_mevel_species_names for reactant in R.reactants):
                if all(product in marinov_mevel_species_names for product in R.products):
                    new_reactions.append(R)
                    # print('Accepted reaction:', R.equation)
    # print('\n')

    marinov_mevel_species = marinov_species + new_species
    marinov_mevel_reactions = marinov_reactions + new_reactions

    marinov_mevel_gas = ct.Solution(
        thermo='IdealGas',
        kinetics='GasKinetics',
        species=marinov_mevel_species,
        reactions=marinov_mevel_reactions
    )

    marinov_mevel_gas = ct.Solution('sandiego2016_plus_N_CK.cti')
    print('Number of species:', marinov_mevel_gas.n_species)
    print('Number of reactions:', marinov_mevel_gas.n_reactions)

    return marinov_mevel_gas


# Droplet Fed Variable Area Plug Flow Reactor Model
class NoelleReactor(object):
    def __init__(self, gas, area, liquid, N_dot, q_dot_prime=0):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.Tmin = 1
        self.A = area
        self.dA_dx = Function(lambda x: area.differentiate(x))
        self.N_dot = N_dot
        
        # Liquid information - always at boiling temperature
        self.droplets_exist = True
        self.liquid = liquid
        self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 0)
        ## Density
        self.rho_l = self.liquid.rhomass()
        ## Boiling temperature
        self.T_l = self.liquid.T()
        ## Heat of vaporization
        self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 1)
        h_v = self.liquid.hmass()
        self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 0)
        h_l = self.liquid.hmass()
        self.h_fg = h_v - h_l

        # Heat Loss due to Regenerative Cooling
        self.q_dot_prime = 0.0

        self.last_x = -100

    def state_derivate_with_droplets(self, x, u):
        """ ODE function u'= f(x, u).
        
        Parameters
        ----------
        x : float
            Axial position in meters.
        u : np.array
            State variable. Variables are:
            u[0] = D^2 (droplet diameter (SMD) squared)
            u[1] = ml (liquid phase flow rate)
            u[2] = mg (gas phase flow rate)
            u[3] = v_d (droplet velocity)
            u[4] = rho_g (gas phase density)
            u[5] = T (gas phase temperature)
            u[6:6+N] = Y_i (mass fraction of the i-th species, out of N species)       
        """
        # Get variables from state variable
        self.droplets_exist = False
        D2, ml, mg, v_d, rho_g, T = u[:6]
        if D2 <= 0 or rho_g <= 0 or T <=0:
            return 0*u
        D = (D2**0.5)*1e-6
        Y = u[6:]
        A = self.A(x)
        dA_dx = self.dA_dx(x)
        rho_g = max(0.5, rho_g)
        v_g = mg/(rho_g*A)
        rho_l = self.rho_l

        # Update gas state
        self.gas.set_unnormalized_mass_fractions(Y)
        self.gas.TP = T, rho_g*ct.gas_constant*T/(self.gas.mean_molecular_weight)
        if self.gas.P > 1e7:
            self.gas.TP = T, 2e6

        # Get cp, MW, omega, etc
        R_u = ct.gas_constant
        cp = self.gas.cp
        omega_i = self.gas.net_production_rates
        MW_i = self.gas.molecular_weights
        MW_mix = self.gas.mean_molecular_weight
        h_i = self.gas.partial_molar_enthalpies
        mu_g = self.gas.viscosity

        # Compute dD^2/dx
        T_bar = 0.5*T + 0.5*self.T_l
        try:
            # Update states
            self.gas.TP = T_bar, self.gas.P
            self.liquid.update(CoolProp.PT_INPUTS, self.gas.P, T)
            # Calculate K
            k_v   = self.liquid.conductivity()
            k_inf = self.gas.thermal_conductivity # PropsSI('conductivity','T', T, 'P', P, 'Air')
            kg    = 0.4*k_v + 0.6*k_inf
            c_pv  = self.liquid.cpmass()
            K     = 8*kg/(rho_l*c_pv) * np.log(1 + c_pv*(T - self.T_l)/self.h_fg)
            # Roll back states
            self.gas.TP = T, self.gas.P
            self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 0)
        except ValueError as E:
            # print(E)
            # print('ERROR! State Variable:', u)
            # print('Using K = 7.25e-7 to continue.')
            K = 7.25e-07
        dD2_dx = -K/v_d * 1e12

        # Compute dml/dx and dmg/dx
        dml_dx = np.pi/4 * self.N_dot * rho_l * D * dD2_dx * 1e-12
        dmg_dx = -dml_dx

        # Compute dv_d/dx
        v_rel = v_d - v_g
        Re = rho_g*abs(v_d - v_g)*D/mu_g
        Cd = 24/Re + 6/(1+np.sqrt(Re)) + 0.4
        dv_d_dx = -(3*Cd*rho_g*v_rel**2)/(4*rho_l*v_d*D)*v_rel/abs(v_rel)

        # Check Mach Number
        # M2 = v_g**2 / (self.gas.cp/self.gas.cv * R_u/MW_mix * T)
        # s = 0.0001
        # dA_dx *= (1 - np.exp(-((M2-1)/s)**2))

        # Compute drho_g/dx
        # TODO: verify if droplets affect this equation
        drho_g_dx = (
            (1 - R_u/(cp*MW_mix)) * (rho_g**2) * (v_g**2) * (dA_dx/A) +
            ((rho_g*R_u)/(v_g*cp*MW_mix)) * sum(omega_i*(h_i - cp*T*MW_mix))
        )/(
            self.gas.P*(1+ (v_g**2)/(cp*T)) - (rho_g*v_g**2)
        )

        # Compute dT/dx
        # TODO: remove heat due to cooling and recirculation
        self.liquid.update(CoolProp.PT_INPUTS, self.gas.P, T)
        h_g = self.liquid.hmass()
        self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 0)
        h_l = self.liquid.hmass()
        dT_dx = (
            ((v_g**2)/(rho_g*cp)) * drho_g_dx +
            ((v_g**2)/cp) * (dA_dx/A) -
            (1/(v_g*rho_g*cp))*sum(h_i*omega_i) + 
            (h_g - h_l)*dml_dx/(mg*cp)
        )      
        # drho_g_dx2 = rho_g * ( M2 / (1 - M2) * (1/A * dA_dx) )        
        # dT_dx2 =  ( 1 + M2 / (1 - M2)) * ( (1/A * dA_dx) * M2 * T * (self.gas.cp/self.gas.cv - 1) )
        
        # Compute dY_dx
        dY_dx = omega_i * MW_i / (rho_g*v_g)  
        # Add droplet vaporization term to ethanol mass fraction
        dY_dx[37] += dmg_dx/mg

        return np.hstack(([dD2_dx, dml_dx, dmg_dx, dv_d_dx, drho_g_dx, dT_dx], dY_dx)) 

    def state_derivate_vaporization_controlled_combustion(self, x, u):
        """ ODE function u'= f(x, u).
        
        Parameters
        ----------
        x : float
            Axial position in meters.
        u : np.array
            State variable. Variables are:
            u[0] = D^2 (droplet diameter (SMD) squared)
            u[1] = ml (liquid phase flow rate)
            u[2] = mg (gas phase flow rate)
            u[3] = v_d (droplet velocity)
            u[4] = rho_g (gas phase density)
            u[5] = T (gas phase temperature)
            u[6] = Phi (gas equivalence ratio)
            u[7:7+N] = Y_i (mass fraction of the i-th species, out of N species)       
        """
        # Get variables from state variable
        self.droplets_exist = True
        D2, ml, mg, v_d, rho_g, T, phi = u[:7]
        if D2 <= 0 or rho_g <= 0 or T <=0:
            return 0*u
        D = (D2**0.5)*1e-6
        Y = u[7:]
        A = self.A(x)
        dA_dx = self.dA_dx(x)
        rho_l = self.rho_l
        # Update gas state
        self.gas.set_equivalence_ratio(phi, fuel='C2H5OH', oxidizer='N2O')
        self.gas.TP = T, P # rho_g*ct.gas_constant*T/(self.gas.mean_molecular_weight)
        self.gas.equilibrate('TP')
        rho_g = self.gas.density
        v_g = mg/(rho_g*A)

        # Get cp, MW, omega, etc
        R_u = ct.gas_constant
        cp = self.gas.cp
        omega_i = self.gas.net_production_rates
        MW_i = self.gas.molecular_weights
        MW_mix = self.gas.mean_molecular_weight
        h_i = self.gas.partial_molar_enthalpies
        mu_g = self.gas.viscosity

        # Compute dD^2/dx
        T_bar = 0.5*T + 0.5*self.T_l
        try:
            # Update states
            self.gas.TP = T_bar, self.gas.P
            self.liquid.update(CoolProp.PT_INPUTS, self.gas.P, T)
            # Calculate K
            k_v   = self.liquid.conductivity()
            k_inf = self.gas.thermal_conductivity # PropsSI('conductivity','T', T, 'P', P, 'Air')
            kg    = 0.4*k_v + 0.6*k_inf
            c_pv  = self.liquid.cpmass()
            K     = 8*kg/(rho_l*c_pv) * np.log(1 + c_pv*(T - self.T_l)/self.h_fg)
            # Roll back states
            self.gas.TP = T, self.gas.P
            self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 0)
        except ValueError as E:
            # print(E)
            # print('ERROR! State Variable:', u)
            # print('Using K = 7.25e-7 to continue.')
            K = 7.25e-07
        dD2_dx = -K/v_d * 1e12

        # Compute dml/dx and dmg/dx
        dml_dx = np.pi/4 * self.N_dot * rho_l * D * dD2_dx * 1e-12
        dmg_dx = -dml_dx

        # Compute dv_d/dx
        v_rel = v_d - v_g
        Re = rho_g*abs(v_d - v_g)*D/mu_g
        Cd = 24/Re + 6/(1+np.sqrt(Re)) + 0.4
        dv_d_dx = -(3*Cd*rho_g*v_rel**2)/(4*rho_l*v_d*D)*v_rel/abs(v_rel)

        # Compute dphi_dx
        FOst = 0.18445603193
        dphi_dx = 1/(FOst) * dmg_dx /mg_0 
        
        # Compute dT_dx
        # h_g = self.enthalpy(T, P, phi)
        self.liquid.update(CoolProp.PT_INPUTS, self.gas.P, T)
        h_g = self.liquid.hmass()
        self.liquid.update(CoolProp.PQ_INPUTS, self.gas.P, 0)
        h_l = self.liquid.hmass()
        dh_dphi = self.enthalpy_partial_phi(T, P, phi)
        dT_dx = ((h_g - h_l)*dml_dx/mg - dh_dphi*dphi_dx + self.q_dot_prime/mg)/cp
        return np.hstack(([dD2_dx, dml_dx, dmg_dx, dv_d_dx, 0, dT_dx, dphi_dx], 0*Y)) 

    def state_derivate_reacting_nozzle(self, x, u):
        """ ODE function u'= f(x, u).
        
        Parameters
        ----------
        x : float
            Axial position in meters.
        u : np.array
            State variable. Variables are:
            u[0] = D^2 (droplet diameter (SMD) squared)
            u[1] = ml (liquid phase flow rate)
            u[2] = mg (gas phase flow rate)
            u[3] = v_d (droplet velocity)
            u[4] = rho_g (gas phase density)
            u[5] = T (gas phase temperature)
            u[6] = 0
            u[7:7+N] = Y_i (mass fraction of the i-th species, out of N species)       
        """
        # Get variables from state variable
        self.droplets_exist = False
        D2, ml, mg, v_d, rho_g, T, phi = u[:7]
        if D2 <= 0 or rho_g <= 0 or T <=0:
            return 0*u
        Y = u[7:]
        A = self.A(x)
        dA_dx = self.dA_dx(x)
        v_g = mg/(rho_g*A)
        rho_l = self.rho_l

        # Update gas state
        self.gas.set_unnormalized_mass_fractions(Y)
        self.gas.TP = T, rho_g*ct.gas_constant*T/(self.gas.mean_molecular_weight)
        # Get cp, MW, omega, etc
        R_u = ct.gas_constant
        cp = self.gas.cp
        omega_i = self.gas.net_production_rates
        MW_i = self.gas.molecular_weights
        MW_mix = self.gas.mean_molecular_weight
        h_i = self.gas.partial_molar_enthalpies
        mu_g = self.gas.viscosity
        
        dD2_dx = 0
        dml_dx = 0
        dmg_dx = 0
        dv_d_dx = 0

        # Check Mach Number
        # M2 = v_g**2 / (self.gas.cp/self.gas.cv * R_u/MW_mix * T)
        # s = 0.0001
        # dA_dx *= (1 - np.exp(-((M2-1)/s)**2))

        # Compute drho_g/dx
        drho_g_dx = (
            (1 - R_u/(cp*MW_mix)) * (rho_g**2) * (v_g**2) * (dA_dx/A) +
            ((rho_g*R_u)/(v_g*cp*MW_mix)) * sum(omega_i*(h_i - cp*T*MW_mix))
        )/(
            self.gas.P*(1+ (v_g**2)/(cp*T)) - (rho_g*v_g**2)
        )

        # Compute dT/dx
        dT_dx = (
            ((v_g**2)/(rho_g*cp)) * drho_g_dx +
            ((v_g**2)/cp) * (dA_dx/A) -
            (1/(v_g*rho_g*cp))*sum(h_i*omega_i) +
            self.q_dot_prime/cp/mg
        )      
        # drho_g_dx2 = rho_g * ( M2 / (1 - M2) * (1/A * dA_dx) )        
        # dT_dx2 =  ( 1 + M2 / (1 - M2)) * ( (1/A * dA_dx) * M2 * T * (self.gas.cp/self.gas.cv - 1) )
        
        # Compute dY_dx
        dY_dx = omega_i * MW_i / (rho_g*v_g)  

        return np.hstack(([dD2_dx, dml_dx, dmg_dx, dv_d_dx, drho_g_dx, dT_dx, 0], dY_dx)) 

    def state_derivate_equilibrium_nozzle(self, x, u):
        """ ODE function u'= f(x, u).
        
        Parameters
        ----------
        x : float
            Axial position in meters.
        u : np.array
            State variable. Variables are:
            u[0] = D^2 (droplet diameter (SMD) squared)
            u[1] = ml (liquid phase flow rate)
            u[2] = mg (gas phase flow rate)
            u[3] = v_d (droplet velocity)
            u[4] = rho_g (gas phase density)
            u[5] = T (gas phase temperature)
            u[6:6+N] = Y_i (mass fraction of the i-th species, out of N species)       
        """
        # Get variables from state variable
        self.droplets_exist = False
        D2, ml, mg, v_d, rho_g, T = u[:6]
        Y = u[6:]
        A = self.A(x)
        dA_dx = self.dA_dx(x)
        v_g = mg/(rho_g*A)
        rho_l = self.rho_l

        if rho_g < 0:  
            print('x:', x, 'r:', rho_g)
            rho_g = abs(rho_g)
    
        if T < 0: print(x, T)
        T = max(self.Tmin, T)
        
        # Update gas state
        # self.gas.set_unnormalized_mass_fractions(Y)
        diff = []
        for i in range(3):
            self.gas.TP = T, rho_g*ct.gas_constant*T/(self.gas.mean_molecular_weight)
            self.gas.equilibrate('TP')
            diff += [(rho_g - self.gas.density)]
        
        # Get cp, MW, omega, etc
        R_u = ct.gas_constant
        cp = self.gas.cp
        omega_i = self.gas.net_production_rates
        MW_i = self.gas.molecular_weights
        MW_mix = self.gas.mean_molecular_weight
        h_i = self.gas.partial_molar_enthalpies
        mu_g = self.gas.viscosity # PropsSI('viscosity','T', T, 'P', P, 'Air') # Pa*s
        
        dD2_dx = 0
        dml_dx = 0
        dmg_dx = 0
        dv_d_dx = 0
        dY_dx = 0*Y

        # Check Mach Number
        M2 = v_g**2 / (self.gas.cp/self.gas.cv * R_u/MW_mix * T)
        s = 0.0001
        # dA_dx *= (1 - np.exp(-((M2-1)/s)**2))
        if abs(M2 - 1) < 1e-6:
            dA_dx = abs(dA_dx)
            # v_g += 10.0 * (M2 - 1)*abs(M2 - 1)
            # print(M2)
            # M2 = v_g**2 / (self.gas.cp/self.gas.cv * R_u/MW_mix * T)
            print(M2)


        # Compute drho_g/dx
        drho_g_dx = (
            (1 - R_u/(cp*MW_mix)) * (rho_g**2) * (v_g**2) * (dA_dx/A)
        )/(
            self.gas.P*(1+ (v_g**2)/(cp*T)) - (rho_g*v_g**2)
        )

        
        drho_g_dx2 = rho_g * ( M2 / (1 - M2) * (1/A * dA_dx) )
        

        if 100*abs((drho_g_dx2 - drho_g_dx)/drho_g_dx2) > 1.0:
            print('x:', x)
            print('Delta rho:', drho_g_dx2 - drho_g_dx)

        # Compute dT/dx
        dT_dx = (
            ((v_g**2)/(rho_g*cp)) * drho_g_dx +
            ((v_g**2)/cp) * (dA_dx/A)
        )      

        dT_dx2 =  ( 1 + M2 / (1 - M2)) * ( (1/A * dA_dx) * M2 * T * (self.gas.cp/self.gas.cv - 1) )
        if 100*abs((dT_dx2 - dT_dx)/dT_dx2) > 1.0:
            print('x:', x)
            print('Delta T:', 100*abs((dT_dx2 - dT_dx)/dT_dx2))
        
        return np.hstack(([dD2_dx, dml_dx, dmg_dx, dv_d_dx, drho_g_dx, dT_dx], dY_dx))      

    def enthalpy(self, T, P, phi):
        # T: temperature in K
        # P: pressure in Pa
        # phi: equivalence ratio
        
        # gas.enthalpy_mass: J/kg
        # Set initial state
        self.gas.TP = T, P
        self.gas.set_equivalence_ratio(phi, fuel='C2H5OH', oxidizer='N2O')
        
        # Calculate equilibrium under constant temperature and pressure
        self.gas.equilibrate('TP')
        
        return self.gas.enthalpy_mass

    def enthalpy_partial_T(self, T, P, phi, dT=1):
        return (self.enthalpy(T+dT, P, phi) - self.enthalpy(T, P, phi))/(dT)

    def enthalpy_partial_phi(self, T, P, phi, dphi=1e-8):
        return (self.enthalpy(T, P, phi+dphi) - self.enthalpy(T, P, phi))/(dphi)


# Setting Up Gas, Reactor and Initial Conditions

## State
T_0 = 2922.58 # K
P = 15e5 # Pa
T_0 = 2000.0 # K
# P = 10e5 # Pa

## Gas
gas = create_solution_mechanism()
gas.TPY = T_0, P, 'N2O: 1.0'
# gas.TP = T_0, P
# gas.set_equivalence_ratio(1.0, fuel='C2H5OH', oxidizer='N2O')
# gas.equilibrate('TP')

## Liquid
liquid = CoolProp.AbstractState("HEOS", "Ethanol") # &Water")
# liquid.set_mass_fractions([0.92, 0.08])
liquid.update(CoolProp.PQ_INPUTS, gas.P, 0)
liquid_density = liquid.rhomass()

## Intial conditions
D_0 = 40.002*1e-6 # micro m
D2_0 = (D_0*1e6)**2
ml_0 = 0.314 # kg/s
mg_0 = 1.103 # kg/s
# mg_0 = 1.417 # kg/s
rho_g_0 = gas.density # kg/m3
v_d_0 = 93.75 # m/s
phi_0 = 0.0

## Geometry
# radius = 1.005*0.9395177184726075*Function('nozzle_geometry2.csv', interpolation='linear')
radius = Function('nozzle_geometry.csv', interpolation='linear')
radius.source[:, 1] = scipy.signal.savgol_filter(radius.source[:, 1], 21, 3)
radius.source = radius.source[::3, :]
radius.setInterpolation('spline')
# radius = Function(0.053)
# radius = Function([(0.0, 0.053), (0.1, 0.053), (0.15, 0.0)], interpolation='linear')
area = np.pi*radius**2

## Droplet flow rate
N_dot = 6*ml_0/(liquid_density*np.pi*D_0**3)

## Reactor
q_dot_prime = -0*87.8e3 / 0.0838
reactor = NoelleReactor(gas, area, liquid, N_dot, q_dot_prime)
# reactor.A()
# reactor.dA_dx.plot(-0.275, 0.6)

# Analytical Model - Spalding
k = gas.thermal_conductivity
cp_g = gas.cp
Pr = gas.cp_mass * gas.viscosity / gas.thermal_conductivity
B = 5.35
G_t = (mg_0 + ml_0)/area(0.0)
S = 9*Pr/(2*np.log(1+B))
X0 = rho_g_0 * v_d_0 / G_t
xsi_star = (X0 + 3*S/10)/(S +2)
x_star = xsi_star * G_t * (D_0/2)**2 / (rho_g_0 * np.log(1+B) * k/cp_g/liquid_density)
print(1000*x_star)
# Numerical Integration

## Vaporization-Controlled Combustion
print('Simulating Vaporization-Controlled Combustion')
x_init = -0.275
x_max = 0.060
# x_init = 0.0
# x_max = 0.283
initial_state = np.hstack(([D2_0, ml_0, mg_0, v_d_0, rho_g_0, T_0, phi_0], gas.Y))

def fully_evaporated_event(x, u):
    return min(u[0], u[1])
fully_evaporated_event.terminal = True

def choke_event(x, u):
    D2, ml, mg, v_d, rho_g, T, phi = u[:7]
    Y = u[7:]
    gas.set_equivalence_ratio(phi, fuel='C2H5OH', oxidizer='N2O')
    A = area(x)
    v_g = mg/(rho_g*A)
    gas.TP = T, rho_g*ct.gas_constant*T/(gas.mean_molecular_weight)
    M2 = v_g**2 / (gas.cp/gas.cv * ct.gas_constant/gas.mean_molecular_weight * T)
    return M2 - 1
choke_event.terminal = True

sol_vaporization_controlled_combustion = scipy.integrate.solve_ivp(
    fun=reactor.state_derivate_vaporization_controlled_combustion,
    t_span=(x_init, x_max),
    y0=initial_state,
    method='BDF',
    t_eval=None,
    dense_output=True,
    events=[choke_event, fully_evaporated_event],
    max_step=0.001
)
print(sol_vaporization_controlled_combustion.status)

### Process solution to compute mass fractions
states = ct.SolutionArray(gas)
for i in range(sol_vaporization_controlled_combustion.y.shape[1]):
    u = sol_vaporization_controlled_combustion.y[:, i]
    D2, ml, mg, v_d, rho_g, T, phi = u[:7]
    gas.set_equivalence_ratio(phi, fuel='C2H5OH', oxidizer='N2O')
    gas.TP = T, P
    gas.equilibrate('TP')
    states.append(gas.state)
    sol_vaporization_controlled_combustion.y[4, i] = gas.density
    sol_vaporization_controlled_combustion.y[7:, i] = gas.Y

## Reacting Nozzle
print('Simulating Reacting Nozzle - Converging')
# x_init = -0.050
x_max = -0.006625
x_init = sol_vaporization_controlled_combustion.t[-1]
# x_max = 0.283
initial_state = sol_vaporization_controlled_combustion.y[:, -1]

def choke_event(x, u):
    D2, ml, mg, v_d, rho_g, T, phi = u[:7]
    Y = u[7:]
    gas.set_unnormalized_mass_fractions(Y)
    A = area(x)
    v_g = mg/(rho_g*A)
    gas.TP = T, rho_g*ct.gas_constant*T/(gas.mean_molecular_weight)
    M2 = v_g**2 / (gas.cp/gas.cv * ct.gas_constant/gas.mean_molecular_weight * T)
    return M2 - 1
choke_event.terminal = True

sol_reacting_nozzle_converging = scipy.integrate.solve_ivp(
    fun=reactor.state_derivate_reacting_nozzle,
    t_span=(x_init, x_max),
    y0=initial_state,
    method='BDF',
    t_eval=None,
    dense_output=True,
    events=choke_event,
    max_step=0.001
)
print(sol_reacting_nozzle_converging.status)

print('Simulating Reacting Nozzle - Diverging')
# x_init = -0.050
x_max = 0.060
x_init = 0.0011452 # sol_reacting_nozzle_converging.t[-1] + 0.00556
# x_max = 0.283
initial_state = sol_reacting_nozzle_converging.y[:, -1]

sol_reacting_nozzle_diverging = scipy.integrate.solve_ivp(
    fun=reactor.state_derivate_reacting_nozzle,
    t_span=(x_init, x_max),
    y0=initial_state,
    method='LSODA',
    t_eval=None,
    dense_output=True,
    events=choke_event,
    max_step=0.001
)
print(sol_reacting_nozzle_diverging.status)

solution_y = np.hstack([sol_vaporization_controlled_combustion.y, sol_reacting_nozzle_converging.y, sol_reacting_nozzle_diverging.y])
solution_t = np.hstack([1000*sol_vaporization_controlled_combustion.t, 1000*sol_reacting_nozzle_converging.t, 1000*(sol_reacting_nozzle_diverging.t-0.0077702)])

# solution_y = np.hstack([sol_vaporization_controlled_combustion.y, sol_reacting_nozzle_converging.y])
# solution_t = np.hstack([1000*sol_vaporization_controlled_combustion.t, 1000*sol_reacting_nozzle_converging.t])

# Plot
# Hard variables
states = ct.SolutionArray(gas)
pressure = []
sound_speed = []

for u in solution_y.T:
    D2, ml, mg, v_d, rho_g, T, phi = u[:7]
    Y = u[7:]
    gas.set_unnormalized_mass_fractions(Y)
    gas.TP = T, rho_g*ct.gas_constant*T/(gas.mean_molecular_weight)
    # gas.equilibrate('TP')
    states.append(gas.state)
    pressure += [gas.P]
    sound_speed += [(gas.cp/gas.cv * gas.P/gas.density)**0.5]

sound_speed = np.array(sound_speed)
pressure = np.array(pressure)

# Easy ones
diameter_ratio = np.sqrt(solution_y[0])/(D_0*1e6)
droplet_velocity_ratio = solution_y[3]/solution_y[3][0]
ethanol_mass_fraction = solution_y[37+7]
equivalence_ratio = solution_y[6]

temperature_ratio = solution_y[5]/(1.29*3187.5)
gas_density = solution_y[4]/solution_y[4, 0]
gas_velocity = solution_y[2]/(solution_y[4]*area(solution_t/1000))
gas_mach = gas_velocity/sound_speed

## Ethanol Droplet Plots
# plt.figure(figsize=(10,6))

# plt.plot(solution_t, diameter_ratio, label='Droplet diameter $D/D_0$', linewidth=2)
# plt.plot(solution_t, droplet_velocity_ratio, label='Droplet velocity ratio', linewidth=2)
# plt.plot(solution_t, gas_velocity/solution_y[3][0], label='Gas velocity ratio', linewidth=2)
# plt.plot(solution_t, ethanol_mass_fraction, label=r'Ethanol mass fraction', linewidth=2)

# plt.xlabel('Chamber $x$-coordinate (mm)')
# plt.ylabel('Non-dimensional parameters')
# plt.legend()
# plt.show()

## Nozzle Flow Plots
# plt.figure(figsize=(12,4))
# plt.plot(solution_t, radius(solution_t/1000)/min(radius(solution_t/1000)), linewidth=5, c='k')
# plt.ylim(0, 3.2)
# plt.xlabel('Coordenada Axial $x$ (mm)')
# plt.ylabel('Valores Adimensionais')
# plt.savefig('CRN.svg')
# plt.show()


plt.figure(figsize=(12,4))
# plt.plot(solution_t, radius(solution_t/1000)/min(radius(solution_t/1000)), linewidth=5, c='k')
plt.ylim(0, 2.6)
plt.xlim(0, solution_t[-1]-solution_t[0])
plt.plot(solution_t-solution_t[0], diameter_ratio, label='Diâmetro de Gotículas $D/SMD_0$', linewidth=2)
plt.xlabel('Coordenada Axial $x$ (mm)')
plt.ylabel('Valores Adimensionais')
# plt.legend()
plt.savefig('CRN_diameter.svg')
plt.show()

# plt.figure(figsize=(12,4))
# plt.plot(solution_t, radius(solution_t/1000)/min(radius(solution_t/1000)), linewidth=5, c='k')
# plt.plot(solution_t, diameter_ratio, label='Diâmetro de Gotículas $D/SMD_0$', linewidth=2)
# plt.plot(solution_t, equivalence_ratio, label='Razão de Equivalência $\Phi$', linewidth=2)
# # plt.plot(solution_t, temperature_ratio, label='Temperature ratio $T/T_{ad}$', linewidth=2)
# # plt.plot(solution_t, gas_density, label=r'Gas Density $\rho/\rho_0$', linewidth=2)
# # plt.plot(solution_t, gas_mach, label=r'Gas Mach Number', linewidth=2)
# # plt.plot(solution_t, pressure/15e5, label=r'Pressure Ratio', linewidth=2)
# plt.xlabel('Coordenada Axial $x$ (mm)')
# plt.ylabel('Valores Adimensionais')
# # plt.legend()
# plt.savefig('CRN_diameter_equivratio.svg')
# plt.show()

plt.figure(figsize=(12,4))
plt.ylim(0, 2.6)
plt.xlim(0, solution_t[-1]-solution_t[0])
plt.plot(solution_t-solution_t[0], diameter_ratio, label='Diâmetro de Gotículas $D/SMD_0$', linewidth=2)
# plt.plot(solution_t, equivalence_ratio, label='Razão de Equivalência $\Phi$', linewidth=2)
plt.plot(solution_t-solution_t[0], temperature_ratio, label='Temperatura $T/T_{ad}$', linewidth=2)
plt.xlabel('Coordenada Axial $x$ (mm)')
plt.ylabel('Valores Adimensionais')
# plt.legend()
plt.savefig('CRN_diameter_temp.svg')
plt.show()

plt.figure(figsize=(12,4))
plt.ylim(0, 2.6)
plt.xlim(0, solution_t[-1]-solution_t[0])
plt.plot(solution_t-solution_t[0], diameter_ratio, label='Diâmetro de Gotículas $D/SMD_0$', linewidth=2)
plt.plot(solution_t-solution_t[0], equivalence_ratio, label='Razão de Equivalência $\Phi$', linewidth=2)
# plt.plot(solution_t, temperature_ratio, label='Temperatura $T/T_{ad}$', linewidth=2)
plt.plot(solution_t-solution_t[0], gas_mach, label='Número de Mach', linewidth=2)
plt.xlabel('Coordenada Axial $x$ (mm)')
plt.ylabel('Valores Adimensionais')
# plt.legend()
plt.savefig('CRN_diameter_temp_mach.svg')
plt.show()

plt.figure(figsize=(12,4))
plt.ylim(0, 2.6)
plt.xlim(0, solution_t[-1]-solution_t[0])
plt.plot(solution_t-solution_t[0], diameter_ratio, label='Diâmetro de Gotículas $D/SMD_0$', linewidth=2)
# plt.plot(solution_t, equivalence_ratio, label='Razão de Equivalência $\Phi$', linewidth=2)
plt.plot(solution_t-solution_t[0], temperature_ratio, label='Temperatura $T/T_{ad}$', linewidth=2)
plt.plot(solution_t-solution_t[0], gas_mach, label='Número de Mach', linewidth=2)
plt.xlabel('Coordenada Axial $x$ (mm)')
plt.ylabel('Valores Adimensionais')
plt.legend()
plt.savefig('CRN_legend.svg')
plt.show()

## Combustion Flow
plt.figure(figsize=(10,6))

plt.plot(solution_t, area(solution_t/1000)/min(solution_t/1000), label='Area ratio $A/A_{*}$', linewidth=5, c='k')
plt.plot(solution_t, states('CO2').Y, label=r'$Y_{CO_2}$', linewidth=2)
# plt.plot(solution_t, states('N2O').Y, label=r'$Y_{N_2O}$', linewidth=2)
# plt.plot(solution_t, states('C2H5OH').Y, label=r'$Y_{C_2H_5OH}$', linewidth=2)
plt.plot(solution_t, states('H2O').Y, label=r'$Y_{H_2O}$', linewidth=2)
plt.plot(solution_t, states('O2').Y, label=r'$Y_{O_2}$', linewidth=2)
# plt.plot(solution_t, temperature_ratio, label='Temperature ratio $T/T_{ad}$', linewidth=2)
# plt.plot(solution_t, pressure/15e5, label=r'Pressure Ratio', linewidth=2)
# plt.plot(1000*area.source[:, 0], area.source[:, 1]/max(area.source[:, 1]), label='Area ratio $A/A_{*}$', linewidth=5, c='k')

plt.xlabel('Chamber $x$-coordinate (mm)')
plt.ylabel('Non-dimensional parameters')
plt.legend()
# plt.show()

# reactor.gas.set_unnormalized_mass_fractions(sol_with_droplets.y[6:, -1])
# reactor.gas()



