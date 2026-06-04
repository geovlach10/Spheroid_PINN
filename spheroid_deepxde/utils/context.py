from dataclasses import dataclass, field
from typing import Callable

@dataclass
class PhaseSpecification:
    '''tau: sec, Cext: nM, P: μm2/sec.'''
    name: str
    tau: float
    P: float
    Cext: float
    time_window: tuple[float, float] = (0, 1)

@dataclass
class Antibody:
    D: float 
    Koff: float
    Kd: float 
    Kint: float 
    Rt: float 
    phi: Callable

@dataclass
class CharacteristicUnits:
    C0: float
    R: float   

@dataclass
class DimLessModel:
    antibody :Antibody
    units :CharacteristicUnits
    specification :PhaseSpecification
    
    D_star: float = field(init=False)
    Kon_star: float = field(init=False)
    Koff_star: float = field(init=False)
    Kint_star: float = field(init=False)
    Rt_star: float = field(init=False)
    Cext_star: float = field(init=False)
    P_star: float = field(init=False)

    def __post_init__(self):
        ab, un, sp = self.antibody, self.units, self.specification

        Kon = ab.Koff / ab.Kd

        self.D_star = (sp.tau / un.R**2) * ab.D
        self.Kon_star = un.C0 * sp.tau * Kon
        self.Koff_star = sp.tau * ab.Koff
        self.Kint_star = sp.tau * ab.Kint
        self.Rt_star = ab.Rt / un.C0
        self.Cext_star = sp.Cext/ un.C0
        self.P_star = un.R / ab.D * sp.P

phi_liposomes = lambda r: 0.83 * r**5.21 + 0.17 
phi_antibody = lambda r: 0.44 * r**3.2 + 0.56
