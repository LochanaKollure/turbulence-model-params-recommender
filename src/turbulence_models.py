from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ParameterType(str, Enum):
    DIMENSIONLESS = "dimensionless"
    LENGTH = "length"
    TIME = "time"
    VELOCITY = "velocity"
    KINEMATIC_VISCOSITY = "kinematic_viscosity"
    ENERGY = "energy"
    DISSIPATION = "dissipation"

class TurbulenceParameter(BaseModel):
    name: str
    description: str
    parameter_type: ParameterType
    default_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    units: Optional[str] = None
    typical_range: Optional[str] = None

class TurbulenceModel(BaseModel):
    name: str
    full_name: str
    description: str
    category: str
    parameters: List[TurbulenceParameter]
    applications: List[str]
    limitations: List[str]

# Define popular turbulence models
TURBULENCE_MODELS = {
    "k_epsilon": TurbulenceModel(
        name="k_epsilon",
        full_name="k-ε (k-epsilon)",
        description="Two-equation eddy-viscosity model solving transport equations for turbulent kinetic energy (k) and dissipation rate (ε)",
        category="RANS",
        parameters=[
            TurbulenceParameter(
                name="Cmu",
                description="Turbulent viscosity constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.09,
                min_value=0.05,
                max_value=0.15,
                units="dimensionless",
                typical_range="0.08-0.12"
            ),
            TurbulenceParameter(
                name="C1epsilon",
                description="Dissipation equation constant C1",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.44,
                min_value=1.2,
                max_value=1.6,
                units="dimensionless",
                typical_range="1.35-1.50"
            ),
            TurbulenceParameter(
                name="C2epsilon",
                description="Dissipation equation constant C2",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.92,
                min_value=1.8,
                max_value=2.1,
                units="dimensionless",
                typical_range="1.85-2.0"
            ),
            TurbulenceParameter(
                name="sigma_k",
                description="Prandtl number for turbulent kinetic energy",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.0,
                min_value=0.5,
                max_value=2.0,
                units="dimensionless",
                typical_range="0.8-1.3"
            ),
            TurbulenceParameter(
                name="sigma_epsilon",
                description="Prandtl number for dissipation rate",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.3,
                min_value=1.0,
                max_value=1.8,
                units="dimensionless",
                typical_range="1.2-1.4"
            )
        ],
        applications=[
            "Free shear flows",
            "Wall-bounded flows with mild pressure gradients",
            "Industrial flow applications",
            "Environmental flows"
        ],
        limitations=[
            "Poor performance in adverse pressure gradients",
            "Overestimates spreading rate of round jets",
            "Struggles with strong streamline curvature",
            "Not suitable for transitional flows"
        ]
    ),
    
    "k_omega_sst": TurbulenceModel(
        name="k_omega_sst",
        full_name="k-ω SST (Shear Stress Transport)",
        description="Hybrid model combining k-ω near walls with k-ε in free stream, includes cross-diffusion term and stress limiter",
        category="RANS",
        parameters=[
            TurbulenceParameter(
                name="beta_star",
                description="Closure coefficient for turbulent kinetic energy destruction",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.09,
                min_value=0.05,
                max_value=0.15,
                units="dimensionless",
                typical_range="0.08-0.12"
            ),
            TurbulenceParameter(
                name="alpha1",
                description="Closure coefficient for ω equation in inner region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.553,
                min_value=0.4,
                max_value=0.7,
                units="dimensionless",
                typical_range="0.5-0.6"
            ),
            TurbulenceParameter(
                name="beta1",
                description="Closure coefficient for ω destruction in inner region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.075,
                min_value=0.05,
                max_value=0.1,
                units="dimensionless",
                typical_range="0.07-0.08"
            ),
            TurbulenceParameter(
                name="alpha2",
                description="Closure coefficient for ω equation in outer region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.44,
                min_value=0.3,
                max_value=0.6,
                units="dimensionless",
                typical_range="0.4-0.5"
            ),
            TurbulenceParameter(
                name="beta2",
                description="Closure coefficient for ω destruction in outer region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.0828,
                min_value=0.06,
                max_value=0.12,
                units="dimensionless",
                typical_range="0.07-0.09"
            ),
            TurbulenceParameter(
                name="sigma_k1",
                description="Prandtl number for k equation in inner region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.85,
                min_value=0.5,
                max_value=1.2,
                units="dimensionless",
                typical_range="0.8-0.9"
            ),
            TurbulenceParameter(
                name="sigma_omega1",
                description="Prandtl number for ω equation in inner region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.5,
                min_value=0.3,
                max_value=0.8,
                units="dimensionless",
                typical_range="0.4-0.6"
            ),
            TurbulenceParameter(
                name="sigma_k2",
                description="Prandtl number for k equation in outer region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.0,
                min_value=0.7,
                max_value=1.5,
                units="dimensionless",
                typical_range="0.9-1.2"
            ),
            TurbulenceParameter(
                name="sigma_omega2",
                description="Prandtl number for ω equation in outer region",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.856,
                min_value=0.6,
                max_value=1.2,
                units="dimensionless",
                typical_range="0.8-0.9"
            ),
            TurbulenceParameter(
                name="a1",
                description="Stress limiter constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.31,
                min_value=0.2,
                max_value=0.5,
                units="dimensionless",
                typical_range="0.25-0.35"
            )
        ],
        applications=[
            "Adverse pressure gradient flows",
            "Separated flows",
            "Airfoil and wing aerodynamics",
            "Heat transfer applications",
            "Turbomachinery flows"
        ],
        limitations=[
            "Higher computational cost than standard k-ε",
            "Sensitive to freestream values in external flows",
            "May predict early transition in some cases"
        ]
    ),
    
    "spalart_allmaras": TurbulenceModel(
        name="spalart_allmaras",
        full_name="Spalart-Allmaras",
        description="One-equation model solving transport equation for modified turbulent viscosity, designed for aerodynamic flows",
        category="RANS",
        parameters=[
            TurbulenceParameter(
                name="Cb1",
                description="Production constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.1355,
                min_value=0.1,
                max_value=0.2,
                units="dimensionless",
                typical_range="0.13-0.14"
            ),
            TurbulenceParameter(
                name="Cb2",
                description="Diffusion constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.622,
                min_value=0.5,
                max_value=0.8,
                units="dimensionless",
                typical_range="0.6-0.65"
            ),
            TurbulenceParameter(
                name="Cv1",
                description="Viscosity constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=7.1,
                min_value=6.0,
                max_value=8.0,
                units="dimensionless",
                typical_range="7.0-7.2"
            ),
            TurbulenceParameter(
                name="Cw1",
                description="Wall destruction constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=3.239,
                min_value=3.0,
                max_value=3.5,
                units="dimensionless",
                typical_range="3.2-3.3"
            ),
            TurbulenceParameter(
                name="Cw2",
                description="Wall destruction constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.3,
                min_value=0.2,
                max_value=0.4,
                units="dimensionless",
                typical_range="0.25-0.35"
            ),
            TurbulenceParameter(
                name="Cw3",
                description="Wall destruction constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=2.0,
                min_value=1.5,
                max_value=2.5,
                units="dimensionless",
                typical_range="1.9-2.1"
            ),
            TurbulenceParameter(
                name="sigma",
                description="Prandtl number for turbulent viscosity",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.667,
                min_value=0.5,
                max_value=1.0,
                units="dimensionless",
                typical_range="0.6-0.7"
            )
        ],
        applications=[
            "Aerodynamic flows around airfoils and wings",
            "External aerodynamics",
            "Flows with mild separation",
            "Aerospace applications"
        ],
        limitations=[
            "Primarily calibrated for aerodynamic flows",
            "Less suitable for free shear flows",
            "Limited performance in complex geometries",
            "Not ideal for heat transfer predictions"
        ]
    ),
    
    "reynolds_stress": TurbulenceModel(
        name="reynolds_stress",
        full_name="Reynolds Stress Model (RSM)",
        description="Seven-equation model solving transport equations for all Reynolds stress components and dissipation rate",
        category="RANS",
        parameters=[
            TurbulenceParameter(
                name="Cmu",
                description="Turbulent viscosity constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.09,
                min_value=0.05,
                max_value=0.15,
                units="dimensionless",
                typical_range="0.08-0.12"
            ),
            TurbulenceParameter(
                name="C1epsilon",
                description="Dissipation equation constant C1",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.44,
                min_value=1.2,
                max_value=1.6,
                units="dimensionless",
                typical_range="1.4-1.5"
            ),
            TurbulenceParameter(
                name="C2epsilon",
                description="Dissipation equation constant C2",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.92,
                min_value=1.8,
                max_value=2.1,
                units="dimensionless",
                typical_range="1.9-2.0"
            ),
            TurbulenceParameter(
                name="C1",
                description="Slow pressure-strain constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.8,
                min_value=1.5,
                max_value=2.2,
                units="dimensionless",
                typical_range="1.7-1.9"
            ),
            TurbulenceParameter(
                name="C2",
                description="Rapid pressure-strain constant",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=0.6,
                min_value=0.4,
                max_value=0.8,
                units="dimensionless",
                typical_range="0.55-0.65"
            ),
            TurbulenceParameter(
                name="sigma_epsilon",
                description="Prandtl number for dissipation rate",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.3,
                min_value=1.0,
                max_value=1.8,
                units="dimensionless",
                typical_range="1.2-1.4"
            ),
            TurbulenceParameter(
                name="sigma_k",
                description="Prandtl number for turbulent kinetic energy",
                parameter_type=ParameterType.DIMENSIONLESS,
                default_value=1.0,
                min_value=0.5,
                max_value=2.0,
                units="dimensionless",
                typical_range="0.8-1.3"
            )
        ],
        applications=[
            "Complex flows with strong streamline curvature",
            "Swirling flows",
            "Flows with strong anisotropy",
            "Secondary flow prediction",
            "Buoyant flows"
        ],
        limitations=[
            "High computational cost",
            "Complex implementation",
            "Convergence difficulties",
            "Still relies on gradient-diffusion assumption for some terms"
        ]
    )
}

def get_model_names() -> List[str]:
    """Get list of available turbulence model names."""
    return list(TURBULENCE_MODELS.keys())

def get_model(model_name: str) -> Optional[TurbulenceModel]:
    """Get turbulence model by name."""
    return TURBULENCE_MODELS.get(model_name)

def get_model_parameters(model_name: str) -> List[TurbulenceParameter]:
    """Get parameters for a specific turbulence model."""
    model = get_model(model_name)
    return model.parameters if model else []