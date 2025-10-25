import os
import json
import asyncio
from typing import Tuple, Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from mindat_schema_manager import MindatAPISchemaManager

class IntentHallucinationValidationOutput(BaseModel):
    """Structured output for Intent and Hallucination validation"""
    status: Literal["valid", "invalid", "uncertain"] = Field(
        description="Validation status: 'valid' if passes, 'invalid' if fails, 'uncertain' if needs user confirmation"
    )
    issues: Optional[Dict[str, str]] = Field(
        default=None,
        description="Parameters with issues, mapping param name to reason. For 'invalid' status, use '_error' as key for general errors."
    )
    
class ModelValidator:
    """LLM-based validators (slow, asynchronous)"""
    
    def __init__(self, llm):
        """
        Initialize model validator.
        
        Args:
            llm: Language model for validation
        """
        self.llm = llm
        self.schema_manager = MindatAPISchemaManager()
        
        # Ensure schema is loaded
        self._ensure_schema_loaded()
    
    def _ensure_schema_loaded(self):
        """Ensure API schema is downloaded and loaded"""
        if not os.path.exists(self.schema_manager.schema_path):
            # print("Downloading Mindat API schema...")
            self.schema_manager.download_schema()
        self.schema_manager.load_schema()
        # Pre-load geomaterials endpoint
        self.schema_manager.get_geomaterials_endpoint()
    
    async def model_intent_hallucination_validate(
        self,
        params: dict,
        original_query: str,
        api_docs: dict
    ) -> Tuple[str, Optional[Dict[str, Dict[str, Any]]]]:
        """
        Validate Intent and Hallucination using API documentation and LLM.
        
        Intent validation:
        - Are all used parameters relevant to the user's query?
        - Are there any mentioned requirements missing from parameters?
        
        Hallucination validation:
        - Do parameter values comply with API constraints (type, enum, range)?
        - Are values based on user input or reasonable inference?
        - Or are values fabricated without basis?
        
        Args:
            params: Generated parameters to validate
            original_query: Original user query
            api_docs: API documentation for relevant parameters
        
        Returns:
            (status, issues)
            - status: "valid" | "invalid" | "uncertain"
            - issues: Dict with structured issue details (only if not valid):
                {
                    "param_name": {
                        "value": current_value,
                        "reason": "...",
                        "api_doc": {...}
                    }
                }
        """
        prompt = f"""Validate API parameters against user query and documentation.

USER QUERY: "{original_query}"
PARAMETERS: {json.dumps(params, indent=2)}
API DOCS: {json.dumps(api_docs, indent=2)}

Check:
1. Are all GENERATED parameters relevant to the query?
2. For parameters that EXIST in the API docs provided, are there missing values the user mentioned?
3. Do values comply with API constraints (enum values, format)?
4. Are values based on user input or fabricated?

CRITICAL RULES:
- ONLY validate parameters that appear in the provided API DOCS
- Do NOT suggest parameters that are not in the API DOCS, even if they seem relevant
- "Missing parameter" means: user mentioned a requirement that maps to a provided API parameter, but it's not set
- Example: User says "no sulfur", API docs include "el_exc", but el_exc is not in parameters → INVALID (missing el_exc)
- Counter-example: User says "red minerals", but API docs don't include a color parameter → VALID (API doesn't support color filter, not our fault)

IMPORTANT: Do NOT check data types (string vs array, etc). Data type validation is handled by other mechanisms. Focus only on semantic correctness.

Response format:
{{
    "status": "valid" | "invalid" | "uncertain",
    "issues": {{"param_name": "reason"}}  // Only if status is not "valid"
}}

Examples:
1. Invalid: User "with iron, no sulfur", Params {{"el_inc": "Fe"}} → {{"status": "invalid", "issues": {{"el_exc": "User said 'no sulfur' but el_exc is missing"}}}}
2. Uncertain: User "like quartz", Params {{"hardness_min": 7}} → {{"status": "uncertain", "issues": {{"hardness_min": "Inferred from quartz, not explicit"}}}}
3. Valid: User "hardness 5-7", Params {{"hardness_min": 5, "hardness_max": 7}} → {{"status": "valid"}}
"""
        
        try:
            # Use structured output with Pydantic model
            response = await self.llm.with_structured_output(IntentHallucinationValidationOutput).ainvoke(prompt)
            
            # Build structured issues if any
            if response.status == 'valid':
                return "valid", None
            
            # Has issues - add value and api_doc to each
            llm_issues = response.issues or {}
            structured_issues = {}
            
            for param_name, reason in llm_issues.items():
                if param_name == "_error":
                    # General error
                    structured_issues["_error"] = {
                        "value": None,
                        "reason": reason,
                        "api_doc": {}
                    }
                else:
                    # Parameter-specific issue
                    structured_issues[param_name] = {
                        "value": params.get(param_name),
                        "reason": reason,
                        "api_doc": api_docs.get(param_name, {})
                    }
            
            return response.status, structured_issues
        
        except Exception as e:
            # Return as invalid with error
            return "invalid", {
                "_error": {
                    "value": None,
                    "reason": f"Validation error: {str(e)}",
                    "api_doc": {}
                }
            }
    
    async def run_validation(
        self,
        params: dict,
        original_query: str,
        endpoint: str = '/v1/geomaterials/'
    ) -> Dict[str, Any]:
        """
        Run model validation and return structured results.
        
        Args:
            params: Parameters to validate
            original_query: Original user query
            endpoint: API endpoint path (default: '/v1/geomaterials/')
        
        Returns:
            {
                "status": "valid" | "invalid" | "uncertain",
                "issues": {  # Only if status is not "valid"
                    "param_name": {
                        "value": current_value,
                        "reason": "...",
                        "api_doc": {...}
                    }
                }
            }
        """
        # Get API documentation
        api_docs = self.schema_manager.get_params_info(
            param_names=list(params.keys()),
            endpoint=endpoint
        )
        
        # Run validation and get structured result
        status, issues = await self.model_intent_hallucination_validate(
            params,
            original_query,
            api_docs
        )
        
        # Build result
        result: Dict[str, Any] = {"status": status}
        if issues:
            result["issues"] = issues
        
        return result
    
if __name__ == "__main__":
    
    # from langchain_openai import AzureChatOpenAI
    # from dotenv import load_dotenv
    
    # load_dotenv()
    # llm = AzureChatOpenAI(
    #                 deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    #                 api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #                 azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    #                 api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #                 temperature=0.3,
    #             )
    # validator = ModelValidator(llm)
    # async def test():
    #     params = {
    #         "el_inc": "Fe, Cu, Sr",
    #         "hardness_min": 5,
    #         "hardness_max": 9
    #     }
    #     query = "Find minerals with iron and copper and hardness larger than 5 but less than 9"
    #     result = await validator.run_validation(params, query)
    #     print(json.dumps(result, indent=2))
        
    # asyncio.run(test())
    
    pass