from typing import Dict, Any, List, Optional
import openai
import json
from src.config import settings
from src.turbulence_models import get_model, TurbulenceParameter
import logging

logger = logging.getLogger(__name__)

class ParameterGenerator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o"
    
    def create_system_prompt(self, turbulence_model: str) -> str:
        """Create specialized system prompt for turbulence parameter recommendation."""
        
        model_info = get_model(turbulence_model)
        if not model_info:
            return self._get_generic_system_prompt()
        
        # Build detailed parameter descriptions
        param_descriptions = []
        for param in model_info.parameters:
            param_desc = f"""
- **{param.name}**: {param.description}
  - Type: {param.parameter_type.value}
  - Default: {param.default_value if param.default_value else 'varies'}
  - Range: {param.typical_range if param.typical_range else 'model-dependent'}
  - Units: {param.units if param.units else 'see type'}"""
            param_descriptions.append(param_desc)
        
        param_section = "\n".join(param_descriptions)
        
        system_prompt = f"""You are a world-class expert in Computational Fluid Dynamics (CFD) and turbulence modeling, specializing in parameter selection and optimization for the {model_info.full_name} turbulence model.

## Your Expertise
- Deep understanding of turbulence physics and mathematical modeling
- Extensive experience with {model_info.full_name} model implementation and calibration
- Knowledge of parameter sensitivity and optimal ranges for different flow conditions
- Understanding of how flow physics, geometry, and boundary conditions affect parameter selection

## Model Overview: {model_info.full_name}
**Description**: {model_info.description}
**Category**: {model_info.category}

**Applications**:
{chr(10).join([f'- {app}' for app in model_info.applications])}

**Known Limitations**:
{chr(10).join([f'- {lim}' for lim in model_info.limitations])}

## Parameters to Recommend:
{param_section}

## Your Task
Based on the provided research context and user description, recommend optimal parameter values for the {model_info.full_name} model. Consider:

1. **Flow Physics**: Match parameters to the dominant physical phenomena
2. **Geometry Effects**: Account for streamline curvature, separation, reattachment
3. **Boundary Conditions**: Consider wall treatment, inlet conditions, pressure gradients
4. **Application Domain**: Optimize for the specific engineering application
5. **Literature Best Practices**: Reference validated parameter sets from research

## Response Format
Provide your recommendations as a valid JSON object with this exact structure:
```json
{{
  "model_name": "{turbulence_model}",
  "parameters": {{
    "parameter_name": {{
      "value": <recommended_value>,
      "confidence": <confidence_0_to_1>,
      "rationale": "<explanation_for_choice>"
    }}
  }},
  "overall_confidence": <overall_confidence_0_to_1>,
  "key_considerations": [
    "<important_consideration_1>",
    "<important_consideration_2>"
  ],
  "sensitivity_warnings": [
    "<parameter_sensitivity_warning_if_any>"
  ],
  "validation_recommendations": [
    "<suggested_validation_approach>"
  ]
}}
```

## Guidelines
- **Be Specific**: Provide exact numerical values, not ranges
- **Justify Choices**: Explain your reasoning based on physics and literature
- **Consider Context**: Tailor recommendations to the user's specific application
- **Assess Confidence**: Honestly evaluate your confidence in each recommendation
- **Identify Sensitivities**: Highlight parameters that may need case-specific tuning
- **Suggest Validation**: Recommend how to verify the parameter choices

Remember: Your recommendations will directly impact CFD simulation accuracy and computational efficiency. Provide expert-level guidance that balances theoretical understanding with practical implementation considerations."""

        return system_prompt
    
    def _get_generic_system_prompt(self) -> str:
        """Generic system prompt when model-specific information is unavailable."""
        return """You are an expert CFD turbulence modeling specialist. Based on the provided research context and user requirements, recommend optimal parameter values for the specified turbulence model. 

Provide detailed justifications for your parameter choices, considering flow physics, boundary conditions, and application requirements. Format your response as a valid JSON object with parameter values, confidence levels, and detailed rationales."""
    
    def create_user_prompt(self, 
                          turbulence_model: str,
                          user_description: str,
                          context_text: str,
                          focus_area: str = "") -> str:
        """Create user prompt with context and requirements."""
        
        prompt_parts = []
        
        # Add context from retrieved documents
        if context_text.strip():
            prompt_parts.append(f"## Research Context\n{context_text}\n")
        
        # Add user requirements
        prompt_parts.append(f"## Task Requirements")
        prompt_parts.append(f"**Turbulence Model**: {turbulence_model}")
        
        if user_description.strip():
            prompt_parts.append(f"**Application Description**: {user_description}")
        
        if focus_area.strip():
            prompt_parts.append(f"**Focus Area**: {focus_area}")
        
        prompt_parts.append(f"\n## Request")
        prompt_parts.append(f"Please recommend optimal parameter values for the {turbulence_model} turbulence model based on the research context and application requirements above.")
        
        return "\n".join(prompt_parts)
    
    def generate_parameters(self, 
                          turbulence_model: str,
                          user_description: str,
                          context_text: str,
                          focus_area: str = "",
                          temperature: float = 0.1) -> Dict[str, Any]:
        """Generate parameter recommendations using GPT-4o."""
        
        try:
            # Create prompts
            system_prompt = self.create_system_prompt(turbulence_model)
            user_prompt = self.create_user_prompt(
                turbulence_model, user_description, context_text, focus_area
            )
            
            logger.info(f"Generating parameters for {turbulence_model}")
            
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            try:
                result = json.loads(response_text)
                
                # Validate and enhance response
                result = self._validate_and_enhance_response(result, turbulence_model)
                
                logger.info("Successfully generated parameter recommendations")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                return self._create_error_response("Invalid JSON response from LLM")
                
        except Exception as e:
            logger.error(f"Failed to generate parameters: {str(e)}")
            return self._create_error_response(str(e))
    
    def _validate_and_enhance_response(self, response: Dict[str, Any], turbulence_model: str) -> Dict[str, Any]:
        """Validate and enhance the LLM response."""
        
        # Ensure required fields exist
        if 'parameters' not in response:
            response['parameters'] = {}
        
        if 'overall_confidence' not in response:
            response['overall_confidence'] = 0.5
        
        if 'key_considerations' not in response:
            response['key_considerations'] = []
        
        if 'sensitivity_warnings' not in response:
            response['sensitivity_warnings'] = []
        
        if 'validation_recommendations' not in response:
            response['validation_recommendations'] = []
        
        # Add metadata
        response['model_name'] = turbulence_model
        response['generation_timestamp'] = None  # Could add timestamp here
        
        # Validate parameter values against model schema
        model_info = get_model(turbulence_model)
        if model_info:
            validated_params = {}
            for param in model_info.parameters:
                if param.name in response['parameters']:
                    param_data = response['parameters'][param.name]
                    
                    # Validate value range if specified
                    if 'value' in param_data:
                        value = param_data['value']
                        
                        # Check bounds
                        if param.min_value is not None and value < param.min_value:
                            param_data['validation_warning'] = f"Value {value} below minimum {param.min_value}"
                        
                        if param.max_value is not None and value > param.max_value:
                            param_data['validation_warning'] = f"Value {value} above maximum {param.max_value}"
                    
                    validated_params[param.name] = param_data
            
            response['parameters'] = validated_params
        
        return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': True,
            'error_message': error_message,
            'parameters': {},
            'overall_confidence': 0.0,
            'key_considerations': [f"Error occurred: {error_message}"],
            'sensitivity_warnings': [],
            'validation_recommendations': ["Please check system configuration and try again"]
        }
    
    def format_parameters_for_display(self, response: Dict[str, Any]) -> str:
        """Format parameter response for user-friendly display."""
        
        if response.get('error'):
            return f"Error: {response.get('error_message', 'Unknown error')}"
        
        lines = []
        lines.append(f"# Parameter Recommendations: {response.get('model_name', 'Unknown Model')}")
        lines.append(f"**Overall Confidence**: {response.get('overall_confidence', 0):.2f}")
        lines.append("")
        
        # Parameters
        lines.append("## Recommended Parameters")
        for param_name, param_data in response.get('parameters', {}).items():
            lines.append(f"### {param_name}")
            lines.append(f"- **Value**: {param_data.get('value', 'N/A')}")
            lines.append(f"- **Confidence**: {param_data.get('confidence', 0):.2f}")
            lines.append(f"- **Rationale**: {param_data.get('rationale', 'No rationale provided')}")
            
            if 'validation_warning' in param_data:
                lines.append(f"- ⚠️ **Warning**: {param_data['validation_warning']}")
            
            lines.append("")
        
        # Key considerations
        if response.get('key_considerations'):
            lines.append("## Key Considerations")
            for consideration in response['key_considerations']:
                lines.append(f"- {consideration}")
            lines.append("")
        
        # Sensitivity warnings
        if response.get('sensitivity_warnings'):
            lines.append("## Sensitivity Warnings")
            for warning in response['sensitivity_warnings']:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")
        
        # Validation recommendations
        if response.get('validation_recommendations'):
            lines.append("## Validation Recommendations")
            for recommendation in response['validation_recommendations']:
                lines.append(f"- {recommendation}")
        
        return "\n".join(lines)