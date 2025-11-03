import os
from mcp.server.fastmcp import FastMCP
from opentelemetry import trace
import time
from dotenv import load_dotenv
from phoenix.otel import register
from langchain_openai import AzureChatOpenAI
import asyncio
import json
import ast
from typing import Annotated, List, Tuple, Union, Optional, Type, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from openmindat import GeomaterialRetriever
from collections import Counter
from utils.validation_pipeline import ValidationPipeline, MindatQueryDict


load_dotenv(override=True)

mcp = FastMCP("Mindat_test")

tracer_provider = register(
    project_name="mindat_ai_setup", 
    auto_instrument=True,
    batch=True
)

tracer = tracer_provider.get_tracer(__name__)


    
class ParamGeneration:
    def __init__(
        self,
        user_input: str,
        pydantic_model: Optional[Type[BaseModel]],
        num_generations: int = 1, # number of generations to check for consistency, should be 1 or 3
    ) -> None:
        self.user_input = user_input
        llm = self._initialize_llm()
        self.structured_llm = llm.with_structured_output(pydantic_model)
        self.parser = PydanticOutputParser(pydantic_object=MindatQueryDict)
        self.num_generations = num_generations
        self.validation_pipeline = ValidationPipeline()

    def _initialize_llm(self) -> AzureChatOpenAI:
            """Initialize the Azure OpenAI client with proper error handling."""
            try:
                required_env_vars = [
                    "AZURE_DEPLOYMENT_NAME",
                    "AZURE_OPENAI_API_VERSION", 
                    "AZURE_OPENAI_API_ENDPOINT",
                    "AZURE_OPENAI_API_KEY"
                ]
                
                missing_vars = [var for var in required_env_vars if not os.getenv(var)]
                if missing_vars:
                    raise ValueError(f"Missing required environment variables: {missing_vars}")
                
                return AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    temperature=0.3,
                )
            except Exception as e:
                raise RuntimeError("Failed to initialize AzureChatOpenAI client") from e


    async def _generate_once(self) -> dict:
        system_prompt = (
            "You are a helpful assistant that generates structured search parameters for querying the mindat database. "
            )
        base_human_prompt = (
            f"Given the user input: '{self.user_input}', generate a JSON object with the following fields: "
            )
        structured_llm = self.structured_llm
        last_exception = None
        
        for attempt in range(3):
            try:
                # Build human prompt with error feedback from previous attempt
                if last_exception and attempt > 0:
                    human_prompt = (
                        f"{base_human_prompt}\n\n"
                        f"IMPORTANT: Previous attempt #{attempt} failed with the following error:\n"
                        f"Error type: {type(last_exception).__name__}\n"
                        f"Error message: {str(last_exception)}\n"
                        f"Please fix the issue and generate a valid response."
                    )
                else:
                    human_prompt = base_human_prompt
                
                structured_response = await structured_llm.ainvoke([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ])
                
                return structured_response.model_dump()
                
            except Exception as e:
                last_exception = e
                
                # If this is the last attempt, raise the error
                if attempt == 2:
                    return {"error message": "Error after 3 attempts: " + str(last_exception)}
                
                # Otherwise continue to next attempt
                continue
        
        # This line should never be reached, but just in case
        return {"error message": "Error after 3 attempts: " + str(last_exception)}
    

    async def generate_params(self) -> List[Dict[str, Any]]:
        """
        Generate structured search parameters for querying the mindat database.
        
        Returns:
            List[dict]: On success: [query_params]
                    On failure: [{"consensus_failed": True, "candidates": [...], "message": "..."}]
        """
        if self.num_generations == 1:
            return [await self._generate_once()]
        
        if self.num_generations != 3:
            raise ValueError("num_generations must be either 1 or 3")
        
        async def safe_generate(index: int) -> dict:
            try:
                return await self._generate_once()
            except Exception as e:
                return {
                    "error message": f"Error in generation {index + 1}: {str(e)}",
                    "generation_index": index + 1
                }
        
        max_consensus_attempts = 3
        
        for attempt in range(max_consensus_attempts):
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(safe_generate(i))
                    for i in range(self.num_generations)
                ]
            
            results = [task.result() for task in tasks]
            
            consensus = self._get_consensus(results)
            
            if isinstance(consensus, dict):
                rule_validated_param = self.validation_pipeline._get_rule_validated_param(consensus)
                return [rule_validated_param]
            
            if attempt < max_consensus_attempts - 1:
                continue
        
        # Consensus failed
        valid_candidates = [r for r in results if "error message" not in r]
        
        return [{
            "consensus_failed": True,
            "message": (
                "Parameter generation failed to reach consensus after 3 attempts. "
                "Results were inconsistent. Please select one of the candidates or refine your query."
            ),
            "candidates": valid_candidates,
        }]
    
    def _get_consensus(self, results: List[dict]) -> Union[dict, bool]:
        """
        Get consensus result from multiple generations.
        
        Args:
            results: List of generated JSON objects
        
        Returns:
            dict: The consensus result if at least 2 match
            False: If no consensus
        """
        if not results:
            return False
        
        # Filter out error results (those with "error message" key)
        valid_results = [r for r in results if "error message" not in r]
        
        # Need at least 2 valid results for consensus
        if len(valid_results) < 2:
            return False
        
        # Convert dicts to JSON strings for comparison
        # Use sort_keys=True to ensure consistent ordering
        json_strings = [json.dumps(r, sort_keys=True) for r in valid_results]
        
        # Count occurrences of each unique result
        counts = Counter(json_strings)
        
        # Get the most common result and its count
        most_common_json, count = counts.most_common(1)[0]
        
        # Return consensus if at least 2 results match
        if count >= 2:
            return json.loads(most_common_json)
        else:
            return False




@mcp.tool()
async def query_generation_tool(user_input: str) -> List[dict]:
    """
    Generate structured search parameters for querying the mindat database based on user input.
    Please return all of the querying objects.
    """
    mindat_querist = ParamGeneration(
        user_input=user_input,
        pydantic_model=MindatQueryDict,
        num_generations=3,  # Change to 3 for multiple generations with consensus checking
    )
    response = await mindat_querist.generate_params()
    return response


if __name__ == "__main__":
    mcp.run(transport="stdio")
    
    # user_input = "Query the ima-approved mineral species with hardness between 3-5, in Hexagonal crystal system, must have Neodymium, but without sulfur"
    
    # async def main():
    #     result = await query_generation_tool(user_input)
    #     print(result)

    # asyncio.run(main())
