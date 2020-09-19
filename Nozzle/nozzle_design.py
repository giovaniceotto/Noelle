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
from pyswarm import pso

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
        gas
        ):

        self.inletPressure = inletPressure
        self.inletTemperature = inletTemperature
        self.exitPressure = exitPressure
        self.thrust = thrust
        self.gas = Nozzle.Fluid(gas)

        self.thetaN = 29 # deg
        self.thetaE = 9 # deg

        # Other attributes
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
        self.xGeometry = None
        self.yGeometry = None

        self.MachFunction = None
        self.temperatureFunction = None

        self.channelHeight = None
        self.channelWidth = None
        self.channelLength = None
        self.numberOfChannels = None
        self.coolantMassFlow = None
        self.coolantInletTemperature = None
        self.coolantPressure = None
        self.coolant = None
        self.coolant.waterFraction = None
        self.wallConductivity = None
        self.wallThickness = None

        self.coolant.h = None

        self.gas.h = None
        self.PressureFunction = None

        # Running methods
        self.evaluateInternalFlow()


        return None

    class Fluid:
        def __init__(
            self,
            fluidName
        ):

            self.name = fluidName

            if fluidName == 'Air':
                self.gamma = 1.4 # dimensionless
                self.R = 287 # J/kg.K
            #elif fluidName == "CombustionProducts":

            #elif fluidName == 'Ethanol':

            #elif fluidName == "Ethanol+Water":

            #else:

            return None

        def getRho(self, P, T):
            self.rho = PropsSI('D', 'T', T, 'P', P, self.name)
            return self.rho

        def getViscosity(self, P, T):
            self.viscosity = PropsSI('viscosity', 'T', T, 'P', P, self.name)
            return self.viscosity

        def getK(self, P, T):
            self.k = PropsSI('conductivity', 'T', T, 'P', P, self.name)
            return self.k

        def getPr(self, P, T):
            self.Pr = PropsSI('Prandtl', 'T', T, 'P', P, self.name)
            return self.Pr

    def evaluateInternalFlow(self):
        # Physical properties and constants
        gravity = 9.81 # m/s²
        gamma = self.gas.gamma
        R = self.gas.R

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

    def evaluateGeometry(self, n=500):
        """Evaluates the bell-nozzle geometry

            n: number of points in the discretization
        """
        alpha = 15*pi/180 # rad

        exitDiameter = self.exitDiameter
        throatDiameter = self.throatDiameter

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

        x_bell_contour = x_vector1 + x_vector2[1:] + x_vector3[1:]
        y_bell_contour = y_vector1 + y_vector2[1:] + y_vector3[1:]

        x_bell_contour = np.array(x_bell_contour)
        y_bell_contour = np.array(y_bell_contour)

        self.xGeometry = x_bell_contour
        self.yGeometry = y_bell_contour

        return None

    def evaluateTemperatureFunction(self, n=500):
        gamma = self.gas.gamma
        throatArea = self.throatArea
        inletTemperature = self.inletTemperature

        if self.xGeometry == None or self.yGeometry == None:
            self.evaluateGeometry(n)

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
            return x0

        mach_contour = []

        for i in range(len(yGeometry)):
            R = yGeometry[i]
            if xGeometry[i] <= 0:
                mach_contour.append(mach_number(R, supersonic=False))
            else:
                mach_contour.append(mach_number(R, supersonic=True))

        temperature_profile = []
        for M in mach_contour:
            temperature_profile.append(inletTemperature/(1 + ((gamma - 1)/2)*M**2))

        self.MachFunction = mach_contour
        self.temperatureFunction = temperature_profile 

        return None

    def addCooling(
        self,
        channelHeight,
        channelWidth,
        channelLength,
        numberOfChannels,
        coolantMassFlow,
        coolantInletTemperature,
        coolantPressure,
        coolantType,
        coolantWaterFraction,
        k,
        wallThickness
        ):

        self.channelHeight = channelHeight
        self.channelWidth = channelWidth
        self.channelLength = channelLength
        self.numberOfChannels = numberOfChannels
        self.coolantMassFlow = coolantMassFlow
        self.coolantInletTemperature = coolantInletTemperature
        self.coolantPressure = coolantPressure

        self.coolant = Nozzle.Fluid(coolantType)
        self.coolant.waterFraction = coolantWaterFraction

        self.wallConductivity = k
        self.wallThickness = wallThickness

        self.coolantH()
        self.hotH()

        return None

    def coolantH(self):
        channelWidth = self.channelWidth
        channelHeight = self.channelHeight
        numberOfChannels = self.numberOfChannels
        massFlow = self.coolantMassFlow
        inletTemperature = self.coolantInletTemperature
        coolantPressure = self.coolantPressure

        x = 1 - self.coolant.waterFraction

        HEOS = CP.AbstractState('HEOS', 'Ethanol&Water')
        HEOS.set_mass_fractions([x, 1 - x])

        HEOS.update(CP.PT_INPUTS, coolantPressure, inletTemperature)
        
        Pr = HEOS.Prandtl()
        k = HEOS.conductivity()
        rho = HEOS.rhomass()
        viscosity = HEOS.viscosity()
            
        #rho = self.coolant.getRho(coolantPressure, inletTemperature)
        #viscosity = self.coolant.getViscosity(coolantPressure, inletTemperature)
        #k = self.coolant.getK(coolantPressure, inletTemperature)
        #Pr = self.coolant.getPr(coolantPressure, inletTemperature)

        channelArea = channelWidth*channelHeight
        channelDiameter = 4*channelArea/(2*(channelHeight + channelWidth))
        channelMassFlowRate = massFlow/numberOfChannels
        channelFlowRate = channelMassFlowRate/rho
        channelVelocity = channelFlowRate/channelArea

        reynolds = rho*channelVelocity*channelDiameter/viscosity

        if reynolds >= 10000:
            Nu = 0.023*reynolds**(4/5)*Pr**(0.4)
        else: 
            Nu = 4.36
        h = Nu*k/channelDiameter

        self.coolant.h = h

        return h
    
    def hotH(self, n=500):
        if self.xGeometry == None or self.yGeometry == None:
            self.evaluateGeometry(n)

        xGeometry = self.xGeometry
        yGeometry = self.yGeometry

        if self.temperatureFunction == None or self.MachFunction == None:
            self.evaluateTemperatureFunction(n)

        mach_contour = self.MachFunction
        temperature_profile = self.temperatureFunction
        inletPressure = self.inletPressure
        gamma = self.gas.gamma
        massFlowRate = self.massFlow

        pressure_profile = []
        h_profile = []

        for i in range(len(xGeometry)):
            M_i = mach_contour[i]
            T_i = temperature_profile[i]
            P_i = inletPressure*((1 + 0.5*(gamma - 1)*(M_i**2))**(-gamma/(gamma - 1)))
            pressure_profile.append(P_i)

            D_i = 2*yGeometry[i]
            A_i = 0.25*pi*D_i**2
            
            rho = PropsSI('D', 'T', T_i, 'P', P_i, 'Air')
            viscosity = PropsSI('viscosity', 'T', T_i, 'P', P_i, 'Air')
            Pr = PropsSI('Prandtl', 'T', T_i, 'P', P_i, 'Air')
            k = PropsSI('conductivity', 'T', T_i, 'P', P_i, 'Air')

            flowRate = massFlowRate/rho
            velocity = flowRate/A_i
            reynolds = rho*velocity*D_i/viscosity

            Nu = 0.023*reynolds**(4/5)*Pr**(0.3)
            
            h = Nu*k/D_i

            h_profile.append(h)

        self.gas.h = h_profile
        self.PressureFunction = pressure_profile

        return None

    def fin_efficiency(self, finThickness, T_b):
        k = self.wallConductivity
        h = self.coolant.h
        channelHeight = self.channelHeight
        channelLength = self.channelLength
        inletTemperature = self.coolantInletTemperature

        L_fin = channelHeight
        P = 2*channelLength + 2*finThickness
        Ac = channelLength*finThickness
        theta_b = T_b - inletTemperature
        M_f = sqrt(k*h*P*Ac)*theta_b
        m_f = sqrt(h*P/(k*Ac))
        q_fin = M_f*tanh(m_f*L_fin)
        
        efficiency = q_fin/(h*Ac*theta_b)

        return efficiency

    def wallTemperature(self, finModel=True, n=500):

        if self.xGeometry == None or self.yGeometry == None:
            self.evaluateGeometry(n)
        xGeometry = self.xGeometry
        yGeometry = self.yGeometry

        if self.temperatureFunction == None:
            self.evaluateTemperatureFunction(n)
        temperature_profile = self.temperatureFunction

        channelWidth = self.channelWidth
        numberOfChannels = self.numberOfChannels
        numberOfChannels = round(numberOfChannels)
        numberOfFins = numberOfChannels
        coolantTemperature = self.coolantInletTemperature
        hCoolant = self.coolant.h
        hGas = self.gas.h
        k = self.wallConductivity
        t = self.wallThickness

        wall_temperature_profile = []

        for i in range(len(xGeometry)):
            if finModel == True:
                D_i = 2*yGeometry[i]
                finThickness = (pi*D_i - numberOfChannels*channelWidth)/numberOfFins

                if finThickness < 0:
                    return None

            T_inf1 = temperature_profile[i]
            T_inf2 = coolantTemperature
            h1 = hGas[i]
            h2 = hCoolant

            Tw1 = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + h1/h2)

            if finModel == True:
                T_b = Tw1

                finEfficiency = self.fin_efficiency(finThickness, T_b)
            
                x = (pi*D_i)/(finEfficiency*numberOfFins*finThickness + numberOfChannels*channelWidth)
                Tw1 = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + x*h1/h2)

            wall_temperature_profile.append(Tw1)

        self.wallTemperatureFunction = wall_temperature_profile
        
        return None

    def max_wall_temperature(self, i_list, n=100):
        # i_list: list of indexes for which the wall temperature shall be evaluated
        channelWidth = self.channelWidth
        numberOfChannels = self.numberOfChannels
        numberOfChannels = round(numberOfChannels)
        coolantTemperature = self.coolantInletTemperature
        numberOfFins = numberOfChannels

        self.coolantH()
        self.hotH()
        self.evaluateTemperatureFunction()

        yGeometry = self.yGeometry
        temperature_profile = self.temperatureFunction
        hCoolant = self.coolant.h
        hGas = self.gas.h
        k = self.wallConductivity
        t = self.wallThickness

        wall_temperature_profile = []

        for i in i_list:
            D_i = 2*yGeometry[i]
            
            finThickness = (pi*D_i - numberOfChannels*channelWidth)/numberOfFins

            if finThickness < 0:
                return 3000
            
            T_inf1 = temperature_profile[i]
            T_inf2 = coolantTemperature
            h1 = hGas[i]
            h2 = hCoolant
            
            T_b = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + h1/h2)

            finEfficiency = self.fin_efficiency(finThickness, T_b)
            
            x = (pi*D_i)/(finEfficiency*numberOfFins*finThickness + numberOfChannels*channelWidth)
            Tw1 = T_inf1 - (T_inf1 - T_inf2)/(1 + t*h1/k + x*h1/h2)
            wall_temperature_profile.append(Tw1)
            
        return(max(wall_temperature_profile))

    def geometryPlot(self, n=550):
        self.evaluateGeometry(n)
        x_vector = self.xGeometry
        y_vector = self.yGeometry

        i1 = round(0.2*n)
        i2 = i1 + round(0.1*n)

        x_vector1 = x_vector[0: i1]
        y_vector1 = y_vector[0: i1]

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
        plt.plot(x_vector1*1000, y_vector1*1000, 'b')
        plt.plot(x_vector2*1000, y_vector2*1000, 'r')
        plt.plot(x_vector3*1000, y_vector3*1000, 'g')
        plt.plot(x_axis*1000, y_axis*1000, 'k--')
        plt.plot(x_inlet*1000, y_inlet*1000, 'k')
        plt.plot(x_outlet*1000, y_outlet*1000, 'k')
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title("Bell nozzle contour")
        plt.xlim(-10, 35)
        plt.ylim(-5, 20)
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.savefig('nozzleContour.png', dpi=300)
        plt.show()

        return None
        
    def exportGeometry(self, n=550, plot=False):
        self.evaluateGeometry(n)
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
            
    def getPlots(self, n=550):
        self.evaluateGeometry(n)
        self.evaluateTemperatureFunction()
        self.wallTemperature()

        x_bell_contour = self.xGeometry
        mach_contour = self.MachFunction
        temperature_profile = self.temperatureFunction
        pressure_profile = self.PressureFunction
        wall_temperature_profile = self.wallTemperatureFunction
        h1_profile = self.gas.h

        plt.figure(dpi=150)
        plt.plot(x_bell_contour*1000, mach_contour, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel("M(x)")
        plt.grid(True)
        #plt.savefig('machProfile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(x_bell_contour*1000, temperature_profile, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel("T(x) [K]")
        plt.grid(True)
        #plt.savefig('tempProfile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(x_bell_contour*1000, pressure_profile, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel("P(x) [Pa]")
        plt.grid(True)
        #plt.savefig('tempProfile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(x_bell_contour), wall_temperature_profile, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$T_{w, max}$ [K]")
        plt.grid(True)
        #plt.savefig('wall_temp_profile.png', dpi=300)
        plt.show()

        plt.figure(dpi=150)
        plt.plot(1000*np.array(x_bell_contour), h1_profile, 'b')
        plt.xlabel("x [mm]")
        plt.ylabel(r"$h_{1}$ [W/m²K]")
        plt.grid(True)
        #plt.savefig('h1_contour.png', dpi=300)
        plt.show()

        return None

def objective_function(parameters, n=50):
    inletPressure = parameters[0]
    inletTemperature = parameters[1]
    channelHeight = parameters[2]
    channelWidth = parameters[3]
    numberOfChannels = parameters[4]
    coolantMassFlow = parameters[5]
    coolantInletTemperature = parameters[6]
    coolantWaterFraction = parameters[7]
     
    exitPressure = 100000
    thrust = 1000
    gas = 'Air'
    channelLength = 40e-3
    k = 401
    wallThickness = 2e-3
    coolantPressure = inletPressure
    coolantType = 'Ethanol'

    NOELLE = Nozzle(inletPressure,
        inletTemperature,
        exitPressure,
        thrust,
        gas)

    NOELLE.addCooling(
        channelHeight,
        channelWidth,
        channelLength,
        numberOfChannels,
        coolantMassFlow,
        coolantInletTemperature,
        coolantPressure,
        coolantType,
        coolantWaterFraction,
        k,
        wallThickness
        )

    NOELLE.evaluateGeometry(n)

    i_list = [0, round(0.1*n), round(0.2*n)]
    objFunction = NOELLE.max_wall_temperature(i_list)

    return objFunction

def optimize(lb, ub, j):
    """This function runs the Particle Swarm Optimization algorithm from the pyswarm library
    to minimize the required input torque. The algorithm is run for i times and the best solution
    among them is saved. It is subjected to lower bounds (lb) and upper bounds (ub). 
    It returns the optimum found."""

    xopt = []
    fopt = 0

    for i in range(j):
        xopt_i, fopt_i = pso(objective_function, lb, ub, swarmsize=500, maxiter=100, minstep=1e-6, minfunc=1e-5, debug=True)
        if fopt_i < fopt:
            fopt = fopt_i
            xopt = xopt_i
        i += 1

    return xopt, fopt

