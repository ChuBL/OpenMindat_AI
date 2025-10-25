import requests
import yaml
from pathlib import Path
import os
from typing import Dict, Any, List, Optional

# ============================================================================
# MindatAPISchemaManager - Manages API schema download and parsing
# ============================================================================

class MindatAPISchemaManager:
    """
    Manages Mindat API schema download, parsing, and querying.
    Provides documentation for API parameters to assist LLM validation.
    """
    
    def __init__(self, schema_url: str = "https://api.mindat.org/v1/schema/", 
                 schema_path: str = "./data/Mindat_API.yaml"):
        """
        Initialize schema manager.
        
        Args:
            schema_url: URL to download the API schema
            schema_path: Local path to save/load the schema file
        """
        self.schema_url = schema_url
        self.schema_path = schema_path
        self.schema_data = None
        
        # Cache for different endpoints
        self.endpoints_cache = {}
    
    def download_schema(self) -> bool:
        """
        Download YAML schema file from Mindat API.
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            schema_dir = Path(self.schema_path).parent
            schema_dir.mkdir(parents=True, exist_ok=True)
            
            # Download schema
            response = requests.get(self.schema_url, timeout=30)
            response.raise_for_status()
            
            # Save to file
            with open(self.schema_path, 'wb') as f:
                f.write(response.content)
            
            # print(f"✅ Schema downloaded successfully to {self.schema_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download schema: {str(e)}")
            return False
    
    def load_schema(self) -> bool:
        """
        Load and parse YAML schema file.
        
        Returns:
            True if load successful, False otherwise
        """
        try:
            if not os.path.exists(self.schema_path):
                # print(f"⚠️  Schema file not found at {self.schema_path}, attempting download...")
                if not self.download_schema():
                    return False
            
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                self.schema_data = yaml.safe_load(f)
            
            # print(f"✅ Schema loaded successfully from {self.schema_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load schema: {str(e)}")
            return False
    
    def get_geomaterials_endpoint(self) -> Dict[str, Any]:
        """
        Extract parameter documentation for /v1/geomaterials/ endpoint.
        
        Returns:
            Dictionary mapping parameter names to their documentation:
            {
                "param_name": {
                    "name": "param_name",
                    "description": "...",
                    "schema": {
                        # Original schema structure preserved as-is
                    }
                }
            }
        """
        endpoint_path = '/v1/geomaterials/'
        
        # Check cache first
        if endpoint_path in self.endpoints_cache:
            return self.endpoints_cache[endpoint_path]
        
        if self.schema_data is None:
            if not self.load_schema():
                return {}
        
        try:
            # Navigate to endpoint in schema
            paths = self.schema_data.get('paths', {})
            endpoint = paths.get(endpoint_path, {})
            get_method = endpoint.get('get', {})
            parameters = get_method.get('parameters', [])
            
            # Extract only name, description, and schema
            endpoint_docs = {}
            for param in parameters:
                param_name = param.get('name')
                if not param_name:
                    continue
                
                # Get schema and remove 'type' to avoid confusion
                schema = param.get('schema', {})
                filtered_schema = {k: v for k, v in schema.items() if k != 'type'}
                
                # Also remove 'type' from nested items if it's an array
                if 'items' in filtered_schema and isinstance(filtered_schema['items'], dict):
                    filtered_schema['items'] = {
                        k: v for k, v in filtered_schema['items'].items() if k != 'type'
                    }
                
                endpoint_docs[param_name] = {
                    'name': param_name,
                    'description': param.get('description', ''),
                    'schema': filtered_schema
                }
            
            # Cache the result
            self.endpoints_cache[endpoint_path] = endpoint_docs
            # print(f"✅ Extracted {len(endpoint_docs)} parameters from {endpoint_path}")
            return endpoint_docs
            
        except Exception as e:
            print(f"❌ Failed to extract parameters from {endpoint_path}: {str(e)}")
            return {}
    
    def get_param_info(self, param_name: str, endpoint: str = '/v1/geomaterials/') -> Optional[Dict[str, Any]]:
        """
        Get documentation for a specific parameter from an endpoint.
        
        Args:
            param_name: Name of the parameter
            endpoint: API endpoint path (default: '/v1/geomaterials/')
        
        Returns:
            Parameter documentation dictionary, or None if not found
        """
        if endpoint == '/v1/geomaterials/':
            endpoint_docs = self.get_geomaterials_endpoint()
        else:
            # Future: support other endpoints
            endpoint_docs = {}
        
        return endpoint_docs.get(param_name)
    
    def get_params_info(self, param_names: List[str], endpoint: str = '/v1/geomaterials/') -> Dict[str, Any]:
        """
        Get documentation for multiple parameters from an endpoint.
        Only returns docs for parameters that exist in param_names.
        
        Args:
            param_names: List of parameter names to extract
            endpoint: API endpoint path (default: '/v1/geomaterials/')
        
        Returns:
            Dictionary mapping parameter names to their documentation
            (only for parameters in param_names that exist in the endpoint)
        """
        if endpoint == '/v1/geomaterials/':
            endpoint_docs = self.get_geomaterials_endpoint()
        else:
            # Future: support other endpoints
            endpoint_docs = {}
        
        # Only return docs for requested parameters
        return {
            name: endpoint_docs[name]
            for name in param_names
            if name in endpoint_docs
        }

if __name__ == "__main__":
    pass
    # manager = MindatAPISchemaManager()
    # manager.load_schema()
    # params_info = manager.get_geomaterials_endpoint()
    # # for param, info in params_info.items():
    # #     print(f"{param}: {info}")
        
    # params = {
    #     "ima": True,
    #     "hardness_min": 3.0,
    #     "hardness_max": 5.0,
    #     "crystal_system": ["Hexagonal"],
    #     "el_inc": "Nd",
    #     "el_exc": "S"
    # }
    
    # print(manager.get_params_info(list(params.keys())))