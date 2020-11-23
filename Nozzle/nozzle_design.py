#-----------------------------------------------------------------
# NOELLE Project - Nozzle design
# Author: Luis Henrique da Silva Dias
#-----------------------------------------------------------------

# Importing libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve
from scipy.integrate import simps
from pyswarm import pso

from noelle import *

# Defining mathematical parameters and operations
sqrt = math.sqrt
pi = math.pi
ln = math.log
tan = math.tan
cos = math.cos
sin = math.sin
tanh = math.tanh

class Nozzle:
    """Keeps all nozzle information

    Attributes:

    """
    def __init__(
        self,
        inletPressure,
        inletTemperature,
        exitPressure,
        thrust,
        gas,
        motor=False,
        n=100
        ):

        # Storing input parameters as attributes of the Nozzle object
        self.inletPressure = inletPressure
        self.inletTemperature = inletTemperature
        self.exitPressure = exitPressure
        self.thrust = thrust
        self.motor = motor
        self.discretization = n 

        # Extracting gamma and R of the internal flow
        self.gasName = gas

        if gas == 'Air':
            self.gamma = 1.4 # dimensionless
            self.R = 287 # J/kg.K

        elif gas == "CombustionProducts":
            self.gamma = self.motor.k # dimensionless
            self.R = 8314.462/self.motor.M # J/kg.K

        # Bell-shaped Nozzle parameters, taken from Sutton
        self.thetaN = 19 # deg
        self.thetaE = 9 # deg

        # Combustion Chamber dimensions
        self.chamberDiameter = 0.05
        self.chamberLength = 0.05

        # Initializing other attributes that will be used further
        self.throatArea = None
        self.throatDiameter = None
        self.exitArea = None
        self.exitDiameter = None
        self.exitTemperature = None
        self.exitMach = None
        self.exitVelocity = None
        self.massFlow = None
        self.ISP = None

        self.epsilon = None
        self.xGeometry = []
        self.yGeometry = []

        self.MachFunction = []
        self.temperatureFunction = []
        self.pressureFunction = []

        self.channelHeight = None
        self.channelWidth = None
        self.channelLength = None
        self.numberOfChannels = None
        self.coolantMassFlow = None
        self.coolantInletTemperature = None
        self.coolantPressure = None
        self.coolant = None
        self.coolantWaterFraction = None
        self.wallConductivity = None
        self.wallThickness = None

        self.gasH = []
        self.coolantH = None
        self.coolantVariableH = []
        self.coolantFrictionFactor = None
        self.coolantVariableFrictionFactor = []

        self.finEfficiencyFunction = []
        self.finThicknessFunction = []

        # Running evaluateInternalFlow method, that will calculate the main design parameters
        self.evaluateInternalFlow()

        return None

    def evaluateInternalFlow(self):
        # Physical properties and constants
        gravity = 9.81 # m/s²
        gamma = self.gamma
        R = self.R

        # Design parameters
        thrust = self.thrust
        inletPressure = self.inletPressure
        inletTemperature = self.inletTemperature
        exitPressure = self.exitPressure

        # Calculations: Compressible flow inside the nozzle

        exitMach = sqrt((2/(gamma - 1))*((inletPressure/exitPressure)**((gamma-1)/gamma) - 1))

        exitTemperature = inletTemperature/(1 + ((gamma - 1)/2)*exitMach**2)

        exitVelocity = exitMach*sqrt(gamma*R*exitTemperature)

        massFlowRate = (thrust)/exitVelocity

        throatArea = sqrt((gamma/R)*(2/(gamma + 1))**((gamma + 1)/(gamma - 1)))
        throatArea = throatArea*inletPressure/(sqrt(inletTemperature))
        throatArea = massFlowRate/throatArea

        exitArea = throatArea*(1/exitMach)*(1 + ((gamma - 1)/2)*exitMach**2)**((gamma + 1)/(2*(gamma - 1)))
        exitArea = exitArea*(1/((gamma+1)/2))**((gamma + 1)/(2*(gamma - 1)))

        throatDiameter = sqrt(4*throatArea/pi)

        exitDiameter = sqrt(4*exitArea/pi)

        specificImpulse = exitVelocity/gravity

        # Storing important parameters
        self.throatArea = throatArea
        self.throatDiameter = throatDiameter
        self.exitArea = exitArea
        self.exitDiameter = exitDiameter
        self.exitTemperature = exitTemperature
        self.exitMach = exitMach
        self.exitVelocity = exitVelocity
        self.massFlow = massFlowRate
        self.ISP = specificImpulse

        return None

    def evaluateGeometry(self):
        """Evaluates the bell-nozzle geometry

            n: number of points in the discretization
        """
        alpha = 15*pi/180 # rad

        exitDiameter = self.exitDiameter
        throatDiameter = self.throatDiameter
        n = self.discretization

        length = (exitDiameter - throatDiameter)/(2*tan(alpha))

        self.epsilon = (exitDiameter/throatDiameter)**2

        thetaN = self.thetaN
        thetaE = self.thetaE

        def circle1(Rt):
            theta_vector = (pi/180)*np.linspace(-135, -90, round(0.2*n))
            y = []
            x = []
            for theta in theta_vector:
                x_i = 1.5*Rt*cos(theta)
                y_i = 1.5*Rt*sin(theta) + 1.5*Rt + Rt
                y.append(y_i)
                x.append(x_i)
            return x, y

        def circle2(Rt, thetaN):
            theta_vector = (pi/180)*np.linspace(-90, (thetaN - 90), round(0.1*n))
            y = []
            x = []
            for theta in theta_vector:
                x_i = 0.382*Rt*cos(theta)
                y_i = 0.382*Rt*sin(theta) + 0.382*Rt + Rt
                y.append(y_i)
                x.append(x_i)
            return x, y

        def parabola(Rt, Re, Ln, thetaN, thetaE):
            t_vector = np.linspace(0, 1, int(n - round(0.2*n) - round(0.1*n)))
            x = []
            y = []
            
            x_circle2, y_circle2 = circle2(Rt, thetaN)

            Nx = x_circle2[-1]
            Ny = y_circle2[-1]
            
            Ex = Ln
            Ey = Re
            
            m1 = tan(thetaN*pi/180)
            m2 = tan(thetaE*pi/180)
            C1 = Ny - m1*Nx
            C2 = Ey - m2*Ex
            Qx = (C2 - C1)/(m1 - m2)
            Qy = (m1*C2 - m2*C1)/(m1 - m2)
            
            for t in t_vector:
                x_i = Nx*((1 - t)**2) + Qx*2*(1 - t)*t + Ex*(t**2) 
                y_i = Ny*((1 - t)**2) + Qy*2*(1 - t)*t + Ey*(t**2) 
                x.append(x_i)
                y.append(y_i)
                
            return x, y

        x_vector1, y_vector1 = circle1(throatDiameter/2)
        x_vector2, y_vector2 = circle2(throatDiameter/2, thetaN)
        x_vector3, y_vector3 = parabola(throatDiameter/2, exitDiameter/2, length, thetaN, thetaE)

        x_vector0 = [x_vector1[0] - self.chamberLength, x_vector1[0]]
        y_vector0 = [y_vector1[0], y_vector1[0]]

        x_bell_contour = x_vector0 + x_vector1[1:] + x_vector2[1:] + x_vector3[1:]
        y_bell_contour = y_vector0 + y_vector1[1:] + y_vector2[1:] + y_vector3[1:]

        x_bell_contour = np.array(x_bell_contour)
        y_bell_contour = np.array(y_bell_contour)

        self.xGeometry = x_bell_contour
        self.yGeometry = y_bell_contour

        return None

    def evaluateTemperatureFunction(self):
        gamma = self.gamma
        throatArea = self.throatArea
        inletTemperature = self.inletTemperature
        inletPressure = self.inletPressure

        if len(self.xGeometry) == 0 or len(self.yGeometry) == 0:
            self.evaluateGeometry()

        xGeometry = self.xGeometry
        yGeometry = self.yGeometry

        def mach_number(R, supersonic=True):
            def funct(M):
                At = throatArea
                f = At*(1/M)*((1 + ((gamma - 1)/2)*M**2)*(1/((gamma+1)/2)))**((gamma + 1)/(2*(gamma - 1))) - A
                return f

            A = pi*(R**2)

            if supersonic == True:
                x0 = fsolve(funct, 1.5)
            if supersonic == False:
                x0 = fsolve(funct, 0.5)
            return float(x0)

        mach_contour = []
        for i in range(len(yGeometry)):
            R = yGeometry[i]
            if xGeometry[i] <= 0:
                mach_contour.append(mach_number(R, supersonic=False))
            else:
                mach_contour.append(mach_number(R, supersonic=True))

        temperature_profile = []
        pressure_profile = []
        for M in mach_contour:
            temperature_profile.append(inletTemperature/(1 + ((gamma - 1)/2)*M**2))
            pressure_profile.append(inletPressure*((1 + 0.5*(gamma - 1)*(M**2))**(-gamma/(gamma - 1))))
        
        self.MachFunction = np.array(mach_contour)
        self.temperatureFunction = np.array(temperature_profile) 
        self.pressureFunction = np.array(pressure_profile)

        return None

    def getCoolantProperties(self):
        x = self.waterFraction
        T = self.coolantInletTemperature
        P = self.coolantPressure

        if self.coolantType == 'Ethanol':
            self.coolant_rho = PropsSI('D', 'T', T, 'P', P, self.coolantType)
            self.coolant_viscosity = PropsSI('viscosity', 'T', T, 'P', P, self.coolantType)
            self.coolant_k = PropsSI('conductivity', 'T', T, 'P', P, self.coolantType)
            self.coolant_Pr = PropsSI('Prandtl', 'T', T, 'P', P, self.coolantType)
            self.coolant_Cp = PropsSI('Cpmass', 'T', T, 'P', P, self.coolantType)

            # Ethanol boiling temperature at chamber pressure
            self.ethanolBoilingT = PropsSI('T', 'P', self.inletPressure, 'Q', 0, self.coolantType)

        elif self.coolantType == 'Ethanol+Water':
            HEOS = CP.AbstractState('HEOS', 'Ethanol&Water')
            HEOS.set_mass_fractions([1 - x, x])
            HEOS.update(CP.PT_INPUTS, P, T)
    
            self.coolant_Pr = HEOS.Prandtl()
            self.coolant_k = HEOS.conductivity()
            self.coolant_rho = HEOS.rhomass()
            self.coolant_viscosity = HEOS.viscosity()
            self.coolant_Cp = HEOS.cpmass()

            # Ethanol boiling temperature at chamber pressure
            #HEOS_2 = CP.AbstractState('HEOS', 'Ethanol&Water')
            #HEOS_2.set_mass_fractions([1 - x, x])
            #HEOS_2.update(CP.PQ_INPUTS, self.inletPressure, 0)

            #self.ethanolBoilingT = HEOS_2.T()
            self.ethanolBoilingT = PropsSI('T', 'P', self.inletPressure, 'Q', 0, 'Ethanol')

        else: 
            self.coolant_rho = None
            self.coolant_viscosity = None
            self.coolant_k = None
            self.coolant_Pr = None
            self.coolant_Cp = None
        
        return self.coolant_rho, self.coolant_viscosity, self.coolant_k, self.coolant_Pr
    
    def getVariableCoolantProperties(self, T):
        x = self.waterFraction
        P = self.coolantPressure

        if self.coolantType == 'Ethanol':
            self.coolant_rho = PropsSI('D', 'T', T, 'P', P, self.coolantType)
            self.coolant_viscosity = PropsSI('viscosity', 'T', T, 'P', P, self.coolantType)
            self.coolant_k = PropsSI('conductivity', 'T', T, 'P', P, self.coolantType)
            self.coolant_Pr = PropsSI('Prandtl', 'T', T, 'P', P, self.coolantType)
            self.coolant_Cp = PropsSI('Cpmass', 'T', T, 'P', P, self.coolantType)

            # Ethanol boiling temperature at chamber pressure
            self.ethanolBoilingT = PropsSI('T', 'P', self.inletPressure, 'Q', 0.5, self.coolantType)

        elif self.coolantType == 'Ethanol+Water':
            HEOS = CP.AbstractState('HEOS', 'Ethanol&Water')
            HEOS.set_mass_fractions([1 - x, x])
            HEOS.update(CP.PT_INPUTS, P, T)
    
            self.coolant_Pr = HEOS.Prandtl()
            self.coolant_k = HEOS.conductivity()
            self.coolant_rho = HEOS.rhomass()
            self.coolant_viscosity = HEOS.viscosity()
            self.coolant_Cp = HEOS.cpmass()

            # Ethanol boiling temperature at chamber pressure
            #HEOS_2 = CP.AbstractState('HEOS', 'Ethanol&Water')
            #HEOS_2.set_mass_fractions([1 - x, x])
            #HEOS_2.update(CP.PQ_INPUTS, self.inletPressure, 0.5)

            #self.ethanolBoilingT = HEOS_2.T()
            self.ethanolBoilingT = PropsSI('T', 'P', self.inletPressure, 'Q', 0, 'Ethanol')

        else: 
            self.coolant_rho = None
            self.coolant_viscosity = None
            self.coolant_k = None
            self.coolant_Pr = None
            self.coolant_Cp = None
        
        return self.coolant_rho, self.coolant_viscosity, self.coolant_k, self.coolant_Pr, self.coolant_Cp

    def getGasProperties(self):
        temperatureFunction = self.temperatureFunction
        pressureFunction = self.pressureFunction
        yGeometry = self.yGeometry

        if self.gasName == 'CombustionProducts':
            rho_vector, viscosity_vector, Pr_vector, k_vector = self.motor.calculate_transport_properties(yGeometry)
            viscosity_vector = np.array(viscosity_vector)/10000
            k_vector = np.array(k_vector)/10

        elif self.gasName == 'Air':
            rho_vector = []
            viscosity_vector = []
            Pr_vector = []
            k_vector = []
            for i in range(len(temperatureFunction)):
                T_i = temperatureFunction[i]
                P_i = pressureFunction[i]
                
                rho_vector.append(PropsSI('D', 'T', T_i, 'P', P_i, 'Air'))
                viscosity_vector.append(PropsSI('viscosity', 'T', T_i, 'P', P_i, 'Air'))
                Pr_vector.append(PropsSI('Prandtl', 'T', T_i, 'P', P_i, 'Air'))
                k_vector.append(PropsSI('conductivity', 'T', T_i, 'P', P_i, 'Air'))
        else:
            rho_vector = None
            viscosity_vector = None
            Pr_vector = None
            k_vector = None

        self.rho_vector = rho_vector
        self.viscosity_vector = viscosity_vector
        self.Pr_vector = Pr_vector
        self.k_vector = k_vector

        return None
    
    def addCooling(
        self,
        channelHeight,
        channelWidth,
        numberOfChannels,
        coolantType,
        coolantWaterFraction,
        k,
        wallThickness,
        coolantExcess=1,
        constantCoolantH=False,
        reverseDirection=True,
        coolantMassFlow=False,
        coolantInletTemperature=False,
        coolantPressure=False
        ):

        self.channelHeight = channelHeight
        self.channelWidth = channelWidth
        self.numberOfChannels = numberOfChannels

        if self.motor == False:
            self.coolantMassFlow = coolantExcess*coolantMassFlow
            self.coolantInletTemperature = coolantInletTemperature
            self.coolantPressure = coolantPressure
            
        else:
            self.coolantMassFlow = coolantExcess*self.motor.fuel_mass_flow
            self.coolantInletTemperature = self.motor.fuel.storage_temperature
            self.coolantPressure = self.motor.fuel.storage_pressure

        self.coolantType = coolantType
        self.waterFraction = coolantWaterFraction

        self.wallConductivity = k
        self.wallThickness = wallThickness

        self.constantCoolantH = constantCoolantH
        self.reverseDirection = reverseDirection

        return None

    def getCoolantH(self):
        channelWidth = self.channelWidth
        channelHeight = self.channelHeight
        numberOfChannels = self.numberOfChannels
        massFlow = self.coolantMassFlow

        rho, viscosity, k, Pr = self.getCoolantProperties()

        channelArea = channelWidth*channelHeight
        channelDiameter = 4*channelArea/(2*(channelHeight + channelWidth))
        channelMassFlowRate = massFlow/numberOfChannels
        channelFlowRate = channelMassFlowRate/rho
        channelVelocity = channelFlowRate/channelArea

        reynolds = rho*channelVelocity*channelDiameter/viscosity
        self.coolantReynolds = reynolds

        if reynolds >= 10000:
            f = (0.79*np.log(reynolds) - 1.64)**(-2)
            Nu = 0.023*reynolds**(4/5)*Pr**(0.4)
        elif reynolds <= 2300: 
            Nu = 4.36
            f = 64/(reynolds)
        elif reynolds < 10000 and reynolds > 3000:
            # Gnielinski correlation
            f = (0.79*np.log(reynolds) - 1.64)**(-2)
            Nu = (Pr*(f/8)*(reynolds - 1000))/(1 + 12.7*((f/8)**(0.5))*(Pr**(2/3) -1))
        else:
            # Interpolating
            f = (0.79*np.log(3000) - 1.64)**(-2)
            Nu = 4.36 + ((reynolds - 2300)/(3000 - 2300))*((Pr*(f/8)*(3000 - 1000))/(1 + 12.7*((f/8)**(0.5))*(Pr**(2/3) -1)) - 4.36)

            f = 64/2300 + ((reynolds - 2300)/(3000 - 2300))*((0.79*np.log(3000) - 1.64)**(-2) - 64/2300)

        h = Nu*k/channelDiameter

        L = self.xGeometry[-1] - self.xGeometry[0]
        deltaP = 0.5*rho*f*(channelVelocity**2)*(L/channelDiameter)

        self.coolantH = h
        self.coolantFrictionFactor = f
        self.coolantDeltaP = deltaP
        self.coolantExitPressure = self.coolantPressure - deltaP

        return h

    def getVariableCoolantH(self, T):
        channelWidth = self.channelWidth
        channelHeight = self.channelHeight
        numberOfChannels = self.numberOfChannels
        massFlow = self.coolantMassFlow

        rho, viscosity, k, Pr, Cp = self.getVariableCoolantProperties(T)

        channelArea = channelWidth*channelHeight
        channelDiameter = 4*channelArea/(2*(channelHeight + channelWidth))
        channelMassFlowRate = massFlow/numberOfChannels
        channelFlowRate = channelMassFlowRate/rho
        channelVelocity = channelFlowRate/channelArea

        reynolds = rho*channelVelocity*channelDiameter/viscosity
        self.coolantReynolds = reynolds

        if reynolds >= 10000:
            f = (0.79*np.log(reynolds) - 1.64)**(-2)
            Nu = 0.023*reynolds**(4/5)*Pr**(0.4)
        elif reynolds <= 2300: 
            f = 64/(reynolds)
            Nu = 4.36
        elif reynolds < 10000 and reynolds > 3000:
            # Gnielinski correlation
            f = (0.79*np.log(reynolds) - 1.64)**(-2)
            Nu = (Pr*(f/8)*(reynolds - 1000))/(1 + 12.7*((f/8)**(0.5))*(Pr**(2/3) -1))
        else:
            # Interpolating
            f = (0.79*np.log(3000) - 1.64)**(-2)
            Nu = 4.36 + ((reynolds - 2300)/(3000 - 2300))*((Pr*(f/8)*(3000 - 1000))/(1 + 12.7*((f/8)**(0.5))*(Pr**(2/3) -1)) - 4.36)

            f = 64/2300 + ((reynolds - 2300)/(3000 - 2300))*((0.79*np.log(3000) - 1.64)**(-2) - 64/2300)

        h = Nu*k/channelDiameter

        self.coolantVariableFrictionFactor.append(f)
        self.coolantVariableH.append(h)

        return h, Cp, f, rho, channelVelocity, channelDiameter
    
    def getHotH(self):
        if len(self.xGeometry) == 0 or len(self.yGeometry) == 0:
            self.evaluateGeometry()

        yGeometry = self.yGeometry

        if len(self.temperatureFunction) == 0 or len(self.MachFunction) == 0:
            self.evaluateTemperatureFunction()

        massFlowRate = self.massFlow

        self.getGasProperties()

        rho_vector = self.rho_vector
        viscosity_vector = self.viscosity_vector
        Pr_vector = self.Pr_vector
        k_vector = self.k_vector
        
        h_profile = []

        for i in range(len(yGeometry)):
            D_i = 2*yGeometry[i]
            A_i = 0.25*pi*D_i**2

            rho = rho_vector[i]
            viscosity = viscosity_vector[i]
            Pr = Pr_vector[i]
            k = k_vector[i]

            flowRate = massFlowRate/rho
            velocity = flowRate/A_i
            reynolds = rho*velocity*D_i/viscosity

            Nu = 0.023*reynolds**(4/5)*Pr**(0.3)
            
            h = Nu*k/D_i

            h_profile.append(h)

        self.gasH = np.array(h_profile)
        
        return None

    def getBartzHotH(self):
        if len(self.xGeometry) == 0 or len(self.yGeometry) == 0:
            self.evaluateGeometry()

        yGeometry = self.yGeometry

        if len(self.temperatureFunction) == 0 or len(self.MachFunction) == 0:
            self.evaluateTemperatureFunction()

        massFlowRate = self.massFlow

        self.getGasProperties()

        rho_vector = self.rho_vector
        viscosity_vector = self.viscosity_vector
        Pr_vector = self.Pr_vector
        k_vector = self.k_vector
        
        t_wall_profile = self.wallTemperatureFunction
        t_profile = self.temperatureFunction
        M_profile = self.MachFunction
        
        T0 = t_profile[0]
        gamma = self.gamma
        w = 0.6
        viscosity = viscosity_vector[0]
        Pr = Pr_vector[0]
        k = k_vector[0]
        Cp = Pr*k/viscosity

        h_profile = []
        
        for i in range(len(yGeometry)):
            rho = rho_vector[i]
            Tw_i = t_wall_profile[i]
            M_i = M_profile[i]
            D_i = 2*yGeometry[i]

            A_i = 0.25*pi*D_i**2
            flowRate = massFlowRate/rho
            velocity = flowRate/A_i

            sigma_i = 1/(((0.5*(Tw_i/T0)*(1 + (M_i**2)*(gamma-1)/2) + 0.5)**(0.8-w/5))*((1 + (M_i**2)*(gamma-1)/2)**(w/5)))

            h = (0.026/(D_i**0.2))* ((viscosity**0.2)*Cp/(Pr**0.6)) * ((rho*velocity)**0.8) * sigma_i

            h_profile.append(h)

        self.gasH2 = np.array(h_profile)
        
        return None

    def fin_efficiency(self, finThickness):
        k = self.wallConductivity
        h = self.coolantH
        channelHeight = self.channelHeight

        t = finThickness
        L_fin = channelHeight

        efficiency = sqrt(2*k/(h*t))*tanh(sqrt(2*h/(k*t))*L_fin)

        return efficiency

    def wallTemperature(self, finModel=True):
        if len(self.xGeometry) == 0 or len(self.yGeometry) == 0:
            self.evaluateGeometry()
        xGeometry = self.xGeometry
        yGeometry = self.yGeometry

        if len(self.temperatureFunction) == 0:
            self.evaluateTemperatureFunction()
        temperature_profile = self.temperatureFunction

        if self.constantCoolantH == True:
            self.getCoolantH()
        self.getHotH()

        channelHeight = self.channelHeight
        channelWidth = self.channelWidth
        numberOfChannels = self.numberOfChannels
        numberOfChannels = round(numberOfChannels)
        numberOfFins = numberOfChannels
        hGas = self.gasH
        k = self.wallConductivity
        t = self.wallThickness

        wall_temperature_profile = []
        wall_min_temperature_profile = []
        channel_temperature_profile = [self.coolantInletTemperature]
        channel_pressure_profile = [self.coolantPressure]
        fin_efficiency_vector = []
        fin_thickness_vector = []
        channel_h_profile = []

        if self.reverseDirection == True:
            xGeometry = np.flip(xGeometry)
            yGeometry = np.flip(yGeometry)
            temperature_profile = np.flip(temperature_profile)
            hGas = np.flip(hGas)

        for i in range(len(xGeometry)):
            if finModel == True:
                D_i = 2*yGeometry[i]
                finThickness = (pi*D_i - numberOfChannels*channelWidth)/numberOfFins
                fin_thickness_vector.append(finThickness)

                if finThickness < 0.5e-3:
                    print('Error: Fin Thickness is smaller than 0.5 mm')
                    return None

            T_inf1 = temperature_profile[i]
            T_inf2 = channel_temperature_profile[i]
            h1 = hGas[i]

            if self.constantCoolantH == False:
                h2, coolant_Cp, ff, rho, channelVelocity, channelDiameter = self.getVariableCoolantH(T_inf2)
            else:
                h2 = self.coolantH

            if finModel == True:
                t_fin = finThickness
                L_fin = channelHeight
        
                finEfficiency = sqrt(2*k/(h2*t_fin))*tanh(sqrt(2*h2/(k*t_fin))*L_fin)
                fin_efficiency_vector.append(finEfficiency)
            
                x = (pi*D_i)/(finEfficiency*numberOfFins*finThickness + numberOfChannels*channelWidth)
                Tw1 = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + x*h1/h2)
                Tw2 = T_inf2 + (T_inf1 - T_inf2)/(1 + t*h2/k + h2/(x*h1))

            else:
                Tw1 = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + h1/h2)
                Tw2 = T_inf2 + (T_inf1 - T_inf2)/(1 + t*h2/k + h2/h1)

            if i < (len(xGeometry) - 1):
                if self.constantCoolantH == False:
                    dTdx = h1*(T_inf1 - Tw1)*pi*2*yGeometry[i]/(self.coolantMassFlow*coolant_Cp)
                    dT = dTdx*abs(xGeometry[i] - xGeometry[i+1])
                    channel_temperature_profile.append(dT + T_inf2)

                    dPdx = - 0.5*ff*(rho*channelVelocity**2)/channelDiameter
                    dP = dPdx*abs(xGeometry[i] - xGeometry[i+1])
                    channel_pressure_profile.append(dP + channel_pressure_profile[-1])
                else:
                    channel_temperature_profile.append(T_inf2)
                    channel_pressure_profile.append(channel_pressure_profile[-1])

            wall_temperature_profile.append(Tw1)
            wall_min_temperature_profile.append(Tw2)
            channel_h_profile.append(h2)

        if self.reverseDirection == True:
            wall_temperature_profile = np.flip(np.array(wall_temperature_profile))
            wall_min_temperature_profile = np.flip(np.array(wall_min_temperature_profile))            
            fin_efficiency_vector = np.flip(np.array(fin_efficiency_vector))
            fin_thickness_vector = np.flip(np.array(fin_thickness_vector))
            channel_temperature_profile = np.flip(np.array(channel_temperature_profile))
            channel_pressure_profile = np.flip(np.array(channel_pressure_profile))
            channel_h_profile = np.flip(np.array(channel_h_profile))
        else: 
            wall_temperature_profile = np.array(wall_temperature_profile)
            wall_min_temperature_profile = np.array(wall_min_temperature_profile)            
            fin_efficiency_vector = np.array(fin_efficiency_vector)
            fin_thickness_vector = np.array(fin_thickness_vector)
            channel_temperature_profile = np.array(channel_temperature_profile)
            channel_pressure_profile = np.array(channel_pressure_profile)
            channel_h_profile = np.array(channel_h_profile)

        self.wallTemperatureFunction = wall_temperature_profile
        self.wallMinTemperatureFunction = wall_min_temperature_profile
        self.finEfficiencyFunction = fin_efficiency_vector
        self.finThicknessFunction = fin_thickness_vector
        self.channelTemperatureFunction = channel_temperature_profile
        self.channelPressureFunction = channel_pressure_profile
        self.channelHFunction = channel_h_profile

        # Calculating total heat transfer rate and Ethanol delta T
        y = self.gasH*(np.pi*2*self.yGeometry)*(self.temperatureFunction - self.wallTemperatureFunction)
        Q_total = simps(y, self.xGeometry)

        Tm = (channel_temperature_profile[0] + channel_temperature_profile[-1])/2
        h2, coolant_Cp, _, _, _, _ = self.getVariableCoolantH(Tm)
        deltaT_Ethanol = Q_total/(self.coolant_Cp*self.coolantMassFlow)
        ethanol_exit_temperature = deltaT_Ethanol + self.coolantInletTemperature

        self.totalQ = Q_total
        self.ethanolDeltaT = deltaT_Ethanol
        self.ethanolExitT = ethanol_exit_temperature
        self.Qderivative = y

        self.coolantExitPressure = channel_pressure_profile[-1]
        self.coolantDeltaP = channel_pressure_profile[-1] - channel_pressure_profile[0]

        return None

    def max_wall_temperature(self, n_points):
        # n_points: number of points to be evaluated between inlet and throat
        channelWidth = self.channelWidth
        numberOfChannels = self.numberOfChannels
        numberOfChannels = round(numberOfChannels)
        coolantTemperature = self.coolantInletTemperature
        numberOfFins = numberOfChannels

        def reducedGeometry(n):
            throatDiameter = self.throatDiameter

            def circle1(Rt):
                theta_vector = (pi/180)*np.linspace(-135, -90, n)
                y = []
                x = []
                for theta in theta_vector:
                    x_i = 1.5*Rt*cos(theta)
                    y_i = 1.5*Rt*sin(theta) + 1.5*Rt + Rt
                    y.append(y_i)
                    x.append(x_i)
                return x, y

            x_vector1, y_vector1 = circle1(throatDiameter/2)

            x_bell_contour = x_vector1
            y_bell_contour = y_vector1

            x_bell_contour = np.array(x_bell_contour)
            y_bell_contour = np.array(y_bell_contour)

            self.xGeometry = x_bell_contour
            self.yGeometry = y_bell_contour

            return None

        reducedGeometry(n_points)
        self.evaluateTemperatureFunction()
        self.getCoolantH()
        self.getHotH()

        yGeometry = self.yGeometry
        temperature_profile = self.temperatureFunction
        hCoolant = self.coolantH
        hGas = self.gasH
        k = self.wallConductivity
        t = self.wallThickness

        wall_temperature_profile = []

        for i in range(n_points):
            D_i = 2*yGeometry[i]
            
            finThickness = (pi*D_i - numberOfChannels*channelWidth)/numberOfFins
            minFinThickness = (pi*self.throatDiameter - numberOfChannels*channelWidth)/numberOfFins

            if minFinThickness < 0.5e-3:
                return 3000
            
            T_inf1 = temperature_profile[i]
            T_inf2 = coolantTemperature
            h1 = hGas[i]
            h2 = hCoolant
            
            finEfficiency = self.fin_efficiency(finThickness)
            
            x = (pi*D_i)/(finEfficiency*numberOfFins*finThickness + numberOfChannels*channelWidth)
            Tw1 = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + x*h1/h2)
            wall_temperature_profile.append(Tw1)
            
        return(max(wall_temperature_profile))

    def geometryPlot(self):
        self.evaluateGeometry()
        x_vector = self.xGeometry
        y_vector = self.yGeometry
        n = self.discretization

        i1 = round(0.2*n)
        i2 = i1 + round(0.1*n)

        x_vector0 = x_vector[0: 2]
        y_vector0 = y_vector[0: 2]

        x_vector1 = x_vector[1: i1]
        y_vector1 = y_vector[1: i1]

        x_vector2 = x_vector[(i1-1): i2]
        y_vector2 = y_vector[(i1-1): i2]

        x_vector3 = x_vector[(i2-1):]
        y_vector3 = y_vector[(i2-1):]

        x_axis = np.array([x_vector[-1], x_vector[0]])
        y_axis = np.array([0, 0])

        x_outlet = np.array([x_vector[-1], x_vector[-1]])
        y_outlet = np.array([y_vector[-1], 0])

        x_inlet = np.array([x_vector[0], x_vector[0]])
        y_inlet = np.array([0, y_vector[0]])

        # Contour 
        plt.figure(1, dpi=150)
        plt.plot(x_vector0*1000, y_vector0*1000, 'k')
        plt.plot(x_vector1*1000, y_vector1*1000, 'b')
        plt.plot(x_vector2*1000, y_vector2*1000, 'r')
        plt.plot(x_vector3*1000, y_vector3*1000, 'g')
        plt.plot(x_axis*1000, y_axis*1000, 'k--')
        plt.plot(x_inlet*1000, y_inlet*1000, 'k')
        plt.plot(x_outlet*1000, y_outlet*1000, 'k')
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title("Bell nozzle contour")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('py_nozzle_geometry.png', dpi=300)
        plt.show()

        return None
    
    def exportGeometry(self, plot=False):
        n = self.discretization

        if len(self.xGeometry) == 0 or len(self.yGeometry) == 0:
            self.evaluateGeometry()
        x_vector = self.xGeometry
        y_vector = self.yGeometry

        x_axis = [x_vector[-1], x_vector[0]]
        y_axis = [0, 0]

        x_outlet = [x_vector[-1], x_vector[-1]]
        y_outlet = [y_vector[-1], 0]

        x_inlet = [x_vector[0], x_vector[0]]
        y_inlet = [0, y_vector[0]]

        x_vector_inflation1 = np.array(x_vector)
        y_vector_inflation1 = np.array(y_vector) - 1

        i1 = round(n*60/550)
        x_division1 = [x_vector[i1], x_vector[i1]]
        y_division1 = [y_vector[i1], 0]

        i2 = round(n*200/550)
        x_division2 = [x_vector[i2], x_vector[i2]]
        y_division2 = [y_vector[i2], 0]

        i3 = round(n*350/550)
        x_division3 = [x_vector[i3], x_vector[i3]]
        y_division3 = [y_vector[i3], 0]

        i4 = round(n*450/550)
        x_division4 = [x_vector[i4], x_vector[i4]]
        y_division4 = [y_vector[i4], 0]

        x_mesh_points = x_division1 + x_division2 + x_division3 + x_division4
        y_mesh_points = y_division1 + y_division2 + y_division3 + y_division4

        import csv

        with open('bell_nozzle.txt', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=" ")
            for i in range(len(x_vector)):
                writer.writerow([1, i+1, format(x_vector[i], '.8f'), format(y_vector[i], '.8f'), format(0, '.8f')])
                
        with open('axis.txt', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=" ")
            for i in range(len(x_axis)):
                writer.writerow([1, i+1, format(x_axis[i], '.8f'), format(y_axis[i], '.8f'), format(0, '.8f')])
                
        with open('offset_bell.txt', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=" ")
            for i in range(len(x_vector_inflation1)):
                writer.writerow([1, i+1, format(x_vector_inflation1[i], '.8f'), format(y_vector_inflation1[i], '.8f'), format(0, '.8f')])
                
        with open('mesh_points.txt', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=" ")
            for i in range(len(x_mesh_points)):
                writer.writerow([1, i+1, format(x_mesh_points[i], '.8f'), format(y_mesh_points[i], '.8f'), format(0, '.8f')])

        # Domain 
        plt.figure(1, dpi=150)
        plt.plot(x_vector, y_vector, 'b')
        plt.plot(x_axis, y_axis, 'b--')
        plt.plot(x_inlet, y_inlet, 'b')
        plt.plot(x_outlet, y_outlet, 'b')
        plt.plot(x_vector_inflation1, y_vector_inflation1, 'r')
        plt.plot(x_division1, y_division1, 'g')
        plt.plot(x_division2, y_division2, 'g')
        plt.plot(x_division3, y_division3, 'g')
        plt.plot(x_division4, y_division4, 'g')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title("Simulation Domain")
        plt.xlim(-10, 35)
        plt.ylim(-5, 20)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        return None
    
    def getPlots(self):
        self.evaluateGeometry()
        self.evaluateTemperatureFunction()
        self.wallTemperature()

        plt.figure(dpi=150)
        plt.plot(self.xGeometry*1000, self.MachFunction, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel("M(x)")
        plt.grid(True)
        plt.savefig('py_mach_profile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(self.xGeometry*1000, self.temperatureFunction, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$T_{\infty, hg}$(x) [K]")
        plt.grid(True)
        plt.savefig('py_temp_profile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(self.xGeometry*1000, self.pressureFunction/1e5, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel("P(x) [bar]")
        plt.grid(True)
        plt.savefig('py_pres_profile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.gasH, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$h_{hg}$ [W/m²K]")
        plt.grid(True)
        plt.savefig('py_h_hg.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.finEfficiencyFunction, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$ \epsilon_{fin} $ []")
        plt.grid(True)
        plt.savefig('py_fin_efficiency.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), 1000*np.array(self.finThicknessFunction), 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$ t_{fin} $ [mm]")
        plt.grid(True)
        plt.savefig('py_fin_thickness.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.wallTemperatureFunction, 'r', label="Inner Temperature")
        plt.plot(1000*np.array(self.xGeometry), self.wallMinTemperatureFunction, 'b', label="Outer Temperature")        
        plt.xlabel("x [mm]")
        plt.ylabel(r"$T_{w}$ [K]")
        plt.grid(True)
        plt.legend()
        plt.savefig('py_wall_temp_profile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.Qderivative/1000, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$ \frac{dQ}{dx} $ [kW/m]")
        plt.ylim(0, max(self.Qderivative)/1000)
        plt.grid(True)
        plt.savefig('py_Q_derivative.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.channelHFunction, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$h_{c}$ [W/m²K]")
        plt.grid(True)
        plt.savefig('py_h_c.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.channelTemperatureFunction, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$T_{\inf, c}$ [K]")
        plt.grid(True)
        plt.savefig('py_cool_temp.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(self.xGeometry), self.channelPressureFunction/1e5, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$P_{coolant}$ [bar]")
        plt.grid(True)
        plt.savefig('py_cool_pres.png', dpi=300)
        plt.show()

        return None

    def allInfo(self):
        self.wallTemperature()

        print("Input nozzle data: \n")
        print("Inlet Pressure: ", self.inletPressure/1000, "kPa")
        print("Inlet Temperature: ", self.inletTemperature, "K")
        print("Thrust: ", self.thrust, "N")
        print("Discretization: ", self.discretization, "\n \n")

        print("Input cooling data: \n")
        print("Channel Height: ", self.channelHeight*1000, "mm")
        print("Channel Width: ", self.channelWidth*1000, "mm")
        print("Number of channels: ", self.numberOfChannels)
        print("Coolant mass flow: ", self.coolantMassFlow, "kg/s")
        print("Coolant Temperature: ", self.coolantInletTemperature, "K")
        print("Coolant Pressure: ", self.coolantPressure/1000, "kPa")
        print("Coolant Type: ", self.coolantType)
        print("Coolant water fraction: ", self.waterFraction*100, "%")
        print("Wall conductivity: ", self.wallConductivity, "W/mK")
        print("Wall thickness: ", self.wallThickness*1000, "mm \n \n")

        print("Nozzle design parameters: \n")
        print("Throat diameter: ", self.throatDiameter*1000, "mm")
        print("Exit diameter: ", self.exitDiameter*1000, "mm")
        print("Epsilon: ", self.epsilon)
        print("Exit temperature: ", self.exitTemperature, "K")
        print("Exit Mach: ", self.exitMach)
        print("Exit velocity: ", self.exitVelocity, "m/s")
        print("Nozzle mass flow rate: ", self.massFlow, "kg/s")
        print("Specific Impulse (ISP): ", self.ISP, "s \n \n")

        print("Coolant data: \n")
        print("Coolant rho: ", self.coolant_rho)
        print("Coolant viscosity: ", self.coolant_viscosity)
        print("Coolant conductivity (k): ", self.coolant_k)
        print("Coolant Prandtl number (Pr): ", self.coolant_Pr)
        print("Reynolds number: ", self.coolantReynolds)
        print("Coolant convective heat transfer coefficient: ", self.coolantH, "W/m²K ")
        print("Total heat transfer rate (Q): ", self.totalQ/1000, "kW ")
        print("Ethanol delta T: ", self.ethanolDeltaT, "K")
        print("Ethanol exit temperature: ", self.ethanolExitT, "K")
        print("Ethanol boiling temperature (at chamber pressure): ", self.ethanolBoilingT, "K \n \n")

        self.geometryPlot()

        self.getPlots()
    
def objective_function(parameters):
    inletPressure = parameters[0]
    channelHeight = parameters[1]
    channelWidth = parameters[2]
    numberOfChannels = parameters[3]
    coolantWaterFraction = parameters[4]
    phi = parameters[5]
    coolantExcess = parameters[6]
    thrust = parameters[7]

    x2 = round(coolantWaterFraction*100, 2)
    x1 = 100 - x2
    p_chamber = inletPressure/(10**5)
    storage_pressure = inletPressure + 25e5

    exitPressure = 100000
    gas = 'CombustionProducts'
    k = 401
    wallThickness = 2e-3
    coolantType = 'Ethanol+Water'

    # Oxidizer
    NOX =  Fluid(
        name='N2O', 
        coolprop_name='NitrousOxide', 
        formula=None, 
        fluid_type='oxidizer', 
        storage_temperature=298.15)

    # Fuels
    H2O = Fluid(
        name='H2O(L)', 
        coolprop_name='water', 
        formula='H 2 O 1', 
        fluid_type='fuel', 
        storage_pressure=storage_pressure, 
        storage_temperature=298.15)

    LC2H5OH = Fluid(
        name='C2H5OH(L)', 
        coolprop_name='ethanol', 
        formula='C 2 H 6 O 1', 
        fluid_type='fuel', 
        storage_pressure=storage_pressure, 
        storage_temperature=298.15)

    H2O_C2H50H = FluidMixture(fluid1=LC2H5OH, x1=x1, fluid2=H2O, x2=x2)

    NOELLE = Motor(
        NOX,
        H2O_C2H50H,
        thrust = thrust,
        burn_time = 10,
        p_chamber = p_chamber,
        n_cstar = 1,
        n_cf = 1,
        cd_ox = 0.6,
        cd_fuel = 0.182,
        phi = phi
        )

    inletTemperature = NOELLE.To

    NOELLE_Nozzle = Nozzle(inletPressure,
        inletTemperature,
        exitPressure,
        thrust,
        gas,
        motor=NOELLE
        )

    NOELLE_Nozzle.addCooling(
        channelHeight,
        channelWidth,
        numberOfChannels,
        coolantType,
        coolantWaterFraction,
        k,
        wallThickness,
        coolantExcess=coolantExcess,
        )

    n_points = 3

    try:
        max_wall_temp = NOELLE_Nozzle.max_wall_temperature(n_points)
    except:
        print("Error while calling max_wall_temperature() method \n")
        print("Input parameters are:")
        print(parameters)
        max_wall_temp = 3000

    # Objective is to maximize ISP
    #if max_wall_temp > 750:
    #    objFunction = - NOELLE.Isp + (max_wall_temp - 750)*(300/2000)
    #else:
    #    objFunction = - NOELLE.Isp

    objFunction = max_wall_temp

    return objFunction

def optimize(lb, ub, j):
    """This function runs the Particle Swarm Optimization algorithm from the pyswarm library
    to minimize the required input torque. The algorithm is run for i times and the best solution
    among them is saved. It is subjected to lower bounds (lb) and upper bounds (ub). 
    It returns the optimum found."""

    xopt = []
    fopt = 0

    for i in range(j):
        xopt_i, fopt_i = pso(objective_function, lb, ub, swarmsize=500, maxiter=50, minstep=1e-6, minfunc=1e-5, debug=True)
        if fopt_i < fopt:
            fopt = fopt_i
            xopt = xopt_i
        i += 1

    return xopt, fopt

if __name__ == "__main__":
    # Optimization lower and upper bounds
    lb = np.array([15e5, 0.5e-3, 0.5e-3, 10,    0, 0.6])
    ub = np.array([45e5,   4e-3,   4e-3, 60, 0.05, 1.5])

    # Running the optimization
    x_opt, f_opt = optimize(lb, ub, 2)
    T_min = f_opt