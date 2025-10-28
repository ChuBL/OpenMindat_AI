from typing import Dict, Any
import json
import asyncio
from pydantic import BaseModel, Field
from typing import Optional
from rule_validator import RuleValidator
from model_validator import ModelValidator


class MindatQueryDict(BaseModel):
    ima: Optional[bool] = Field(description="Only IMA-approved names, should be True by default")
    hardness_min: Optional[float] = Field(description="Mohs hardness from")
    hardness_max: Optional[float] = Field(description="Mohs hardness to")
    crystal_system: Optional[list[str]] = Field(description=" Crystal system (csystem): multiple choice (OR), Items Enum: 'Amorphous','Hexagonal','Icosahedral','Isometric','Monoclinic','Orthorhombic','Tetragonal','Triclinic','Trigonal'")
    el_inc: Optional[str] = Field(description="Chemical elements must include, e.g., 'Fe,Cu'")
    el_exc: Optional[str] = Field(description="Chemical elements must exclude, e.g., 'Fe,Cu'")
    # expand: Optional[str] = Field(description="Expand the search scope, 'description','type_localities','locality','relations','minstats', leave blank if necessary")


# ============================================================================
# ValidationPipeline - Orchestrates all validation layers
# ============================================================================

class ValidationPipeline:
    """Orchestrates all validation layers"""
    
    def __init__(self, llm=None):
        """
        Initialize validation pipeline.
        
        Args:
            llm: Language model for LLM-based validation
        """
        # Automatically extract valid fields from MindatQueryDict
        valid_fields = set(MindatQueryDict.model_fields.keys())
        
        # Initialize validators
        self.rule_validator = RuleValidator(valid_fields)
        self.model_validator = ModelValidator(llm)
    
    async def validate(
        self, 
        params: dict, 
        original_query: str = "",
        endpoint: str = '/v1/geomaterials/'
    ) -> Dict[str, Any]:
        """
        Run all validation layers.
        
        Args:
            params: Parameters to validate
            original_query: Original user query for semantic validation
            endpoint: API endpoint path (default: '/v1/geomaterials/')
        
        Returns:
            {
                "status": "valid" | "invalid" | "uncertain",
                "issues": {  # Only if status is not "valid"
                    "rule_name" or "param_name": "error message" or {...}
                },
                "corrected_params": dict  # Corrected parameters (only present when rule validation passes)
            }
        """
        # Layer 1: Rule validators (fast, synchronous)
        rule_result = self.rule_validator.run_validation(params)
        
        if rule_result["status"] != "valid":
            # Rule validation failed - return immediately
            return rule_result
        
        # Rule validation passed - get corrected params
        corrected_params = rule_result.get("corrected_params", params)
        
        # Layer 2: Model validators (slow, asynchronous)
        if self.model_validator and original_query:
            model_result = await self.model_validator.run_validation(
                corrected_params,  # Use corrected params for model validation
                original_query,
                endpoint
            )
            
            # Add corrected_params to model result if not present
            if "corrected_params" not in model_result:
                model_result["corrected_params"] = corrected_params
            
            return model_result
        
        # No model validation - return rule result with corrected params
        return rule_result


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    pass    

    # from langchain_openai import AzureChatOpenAI
    # from dotenv import load_dotenv
    # import os

    # load_dotenv()
    # llm = AzureChatOpenAI(
    #                 deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    #                 api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #                 azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    #                 api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #                 temperature=0.3,
    #             )
 
    # pipeline = ValidationPipeline(llm)
    
    # # Test case 1: Valid params with element case correction
    # params1 = {
    #     "ima": True,
    #     "hardness_min": 3.0,
    #     "hardness_max": 5.0,
    #     "crystal_system": ["Hexagonal"],
    #     "el_inc": "fe,cu",  # lowercase - should be auto-corrected
    #     "el_exc": "s"        # lowercase - should be auto-corrected
    # }
    
    # # Test case 2: Valid hardness range
    # params2 = {
    #     "ima": True,
    #     "hardness_min": 5.0,
    #     "hardness_max": 7.0,  
    #     "el_inc": "Fe"
    # }
    
    # # Test case 3: Invalid element
    # params3 = {
    #     "ima": True,
    #     "el_inc": "Fe,Xx"  # Xx is invalid
    # }
    
    # async def test():
    #     print("=== Test 1: Invalid params with case correction ===")
    #     result1 = await pipeline.validate(params1, "Find IMA minerals with hardness 3-5")
    #     print(json.dumps(result1, indent=2))
        
    #     print("\n=== Test 2: Valid hardness range ===")
    #     result2 = await pipeline.validate(params2, "Find ima approved iron minerals with hardness between 5 and 7")
    #     print(json.dumps(result2, indent=2))
        
    #     print("\n=== Test 3: Invalid element ===")
    #     result3 = await pipeline.validate(params3, "Find minerals with iron")
    #     print(json.dumps(result3, indent=2))
    
    # asyncio.run(test())