from dataclasses import dataclass

class Motor:
    """Docstring.
    """
    def __init__(
        self,
        thrust,
        injector,
        combustion_chamber,
        nozzle,
        environment
    ):
        # Save inputs
        self.thrust = thrust
        self.combustion_chamber = combustion_chamber
        self.nozzle = nozzle
        self.injector = injector
        self.environment = environment
        
        # Configure sub-systems
        self.combustion_chamber.set_motor(self)
        self.nozzle.set_motor(self)
        self.injector.set_motor(self)
        self.environment.set_motor(self)

        ...

    def simulate():
        ...

class CombustionChamber:
    """Docstring.
    """
    def __init__(
        self,
        wall_material,
        radius,
        equivalence_ratio,
        combustion_efficiency=1,
    ):
        ...
    
    def set_motor(self, motor):
        self.motor = motor

class Nozzle:
    """Docstring.
    """
    
    def __init__(
        self,
        expansion_efficiency=1,
    ):
        ...
    
    def set_motor(self, Motor):
        """
        """
        
        self.Motor = Motor
        self.thrust = Motor.thrust
        ...

    def add_cooling():
        ...

    def evaluate_internal_flow():
        ...

    def evaluate_geometry():
        ...
        
    def evaluate_gas_temperature():
        ...
    
    def evaluate_coolant_properties():
        ...

    def evaluate_gas_properties():
        ...

    def evaluate_coolant_h():
        ...

    def evaluate_gas_h():
        ...

    def evaluate_fin_efficiency():
        ...
    
    def evaluate_wall_temperature():
        ...

    def evaluate_max_wall_temperature():
        ...

    def plot_geometry():
        ...

    def plot_all_info():
        ...

    def get_all_info():
        ...
        

class Fluid:
    """Docstring.
    """
    def __init__():
        ...


class Injector:
    """
    This class 
    """
    def __init__(
        self,
        Cd,
        SMD,
        diameter_orifice,
        fluid,
        inj_type
    ):
        """Initialize injector instance
        
        Parameters
        ----------
        Cd: float
            Discharge coefficient (adimensional)
        SMD: float
            Sauter Mean Diameter (10^-6 m)
        diameter_orifice:
            Orifice diameter (m)
            fluid            - Fluid class, refering to oxidiser or fuel
            inj_type         - Injector type: 'biphasic' or 'liquid'

        OUTPUTS:
            None
                
        """

        self.Cd = Cd
        self.SMD = SMD
        self.diameter_orifice = diameter_orifice
        self.fluid = fluid
        self.inj_type = inj_type

        # Compute number of orifice

        ...
    ...

@dataclass
class Enviroment:
    """ Class to hold environment variables.

    Parameters
    ----------
    gravity : float
        Gravitational acceleration in m/s^2.
    ambient_pressure : float
        Ambient pressure in Pa.
    ambient_temperature : float
        Ambient temperature in K.

    Example
    -------
    Env = Enviroment(9.8, 1e5, 300)
    print(Env.gravity)               # 9.8
    """
    gravity: float
    ambient_pressure: float
    ambient_temperature: float

class Optimization:
    """Docstring.
    """
    def __init__():
        ...
    

class OptimizationVariables:
    """Docstring.
    """
    def __init__():
        ...
    

class RestrictionFunction:
    """Docstring.
    """
    def __init__():
        ...
    

