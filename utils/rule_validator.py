from typing import Tuple, Optional, Dict, Any, List, Callable
# import json
# import asyncio
# from pydantic import BaseModel, Field
from typing import Optional


class RuleValidator:
    """Rule-based validators (non-LLM, fast, synchronous)"""
    
    def __init__(self, valid_fields: set = None):
        """
        Initialize rule validator.
        
        Args:
            valid_fields: Set of valid field names for schema validation
        """
        self.valid_fields = valid_fields
        
        # Constants for validation
        self.mohs_scale_min = 1
        self.mohs_scale_max = 10
        self.valid_crystal_systems = {
            'Amorphous', 'Hexagonal', 'Icosahedral', 'Isometric',
            'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Triclinic', 'Trigonal'
        }
        # self.valid_expand_options = {
        #     'description', 'type_localities', 'locality', 'relations', 'minstats'
        # }
        self.valid_elements = self._load_periodic_table()
        self.search_fields = [
            'ima', 'hardness_min', 'hardness_max', 'crystal_system', 'el_inc', 'el_exc'
        ]
        
        # Register all validation methods
        self.validators: Dict[str, Callable] = {
            'rule_schema': self.rule_schema_validate,
            'rule_hardness_range': self.rule_hardness_range_validate,
            'rule_crystal_system': self.rule_crystal_system_validate,
            'rule_chemical_element': self.rule_chemical_element_validate,
            'rule_element_conflict': self.rule_element_conflict_validate,
            # 'rule_expand_option': self.rule_expand_option_validate,
            'rule_completeness': self.rule_completeness_validate,
        }
    
    def _load_periodic_table(self) -> set:
        """Load all valid element symbols"""
        return {
            "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ra", "Th", "U", "[]", "OH", "H2O", "H3O", "BO3", "NH4", "NH2", "NO3", "CO3", "PO4", "SO4", "SO3", "AsO4", "AsO3", "VO4", "CrO4", "SeO4", "SeO3", "MoO4", "SnOH", "SbO4", "SbO3", "TeO4", "TeO3", "IO3", "WO4", "UO2", "SiO4", "SiO3", "Si3O9", "CH3COO", "HCOO", "C2O4"
        }
    
    def _parse_elements(self, element_str: str) -> set:
        """Parse comma-separated element string into set"""
        return {e.strip() for e in element_str.split(',') if e.strip()}
    
    def _correct_element_case(self, element_str: str) -> str:
        """
        Correct element case to match valid_elements.
        Should only be called after validation passes.
        
        E.g., "fe,cu" -> "Fe,Cu"
        """
        # Create case-insensitive lookup dictionary
        element_lookup = {elem.lower(): elem for elem in self.valid_elements}
        
        elements = self._parse_elements(element_str)
        corrected = set()
        
        for elem in elements:
            elem_lower = elem.lower()
            if elem_lower in element_lookup:
                corrected.add(element_lookup[elem_lower])
            else:
                # This shouldn't happen if validation passed, but keep original as fallback
                corrected.add(elem)
        
        return ','.join(sorted(corrected))

    def rule_schema_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """
        Validate that all parameter keys are recognized field names.
        
        Purpose: Catches typos or unexpected fields that aren't part of MindatQueryDict schema.
        Example failure: {"ima": True, "unknown_field": 123} -> Error: unexpected field 'unknown_field'
        Example success: {"ima": True, "hardness_min": 3} -> All fields are valid
        
        This is a safety net beyond Pydantic's type checking.
        """
        if not self.valid_fields:
            return True, None
        
        unexpected = set(params.keys()) - self.valid_fields
        if unexpected:
            return False, f"Unexpected fields: {unexpected}"
        
        return True, None
    
    def rule_hardness_range_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """Validate Mohs hardness range (1-10)"""
        hardness_min = params.get('hardness_min')
        hardness_max = params.get('hardness_max')
        
        # Validate min
        if hardness_min is not None:
            if not (self.mohs_scale_min <= hardness_min <= self.mohs_scale_max):
                return False, f"hardness_min must be between {self.mohs_scale_min} and {self.mohs_scale_max}"
        
        # Validate max
        if hardness_max is not None:
            if not (self.mohs_scale_min <= hardness_max <= self.mohs_scale_max):
                return False, f"hardness_max must be between {self.mohs_scale_min} and {self.mohs_scale_max}"
        
        # Validate min <= max
        if hardness_min is not None and hardness_max is not None:
            if hardness_min > hardness_max:
                return False, "hardness_min cannot exceed hardness_max"
        
        return True, None
    
    def rule_crystal_system_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """Validate crystal system enum values"""
        crystal_system = params.get('crystal_system')
        
        if crystal_system is None:
            return True, None
        
        invalid = set(crystal_system) - self.valid_crystal_systems
        if invalid:
            return False, f"Invalid crystal systems: {invalid}. Valid options: {self.valid_crystal_systems}"
        
        return True, None
    
    def rule_chemical_element_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """Validate chemical element symbols (case-insensitive)"""
        el_inc = params.get('el_inc')
        el_exc = params.get('el_exc')
        
        # Create case-insensitive lookup
        valid_elements_lower = {elem.lower() for elem in self.valid_elements}
        
        # Validate included elements
        if el_inc:
            elements = self._parse_elements(el_inc)
            elements_lower = {elem.lower() for elem in elements}
            invalid = elements_lower - valid_elements_lower
            if invalid:
                return False, f"Invalid elements in el_inc: {invalid}"
        
        # Validate excluded elements
        if el_exc:
            elements = self._parse_elements(el_exc)
            elements_lower = {elem.lower() for elem in elements}
            invalid = elements_lower - valid_elements_lower
            if invalid:
                return False, f"Invalid elements in el_exc: {invalid}"
        
        return True, None
    
    def rule_element_conflict_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """Validate no conflicts between included and excluded elements"""
        el_inc = params.get('el_inc')
        el_exc = params.get('el_exc')
        
        if not el_inc or not el_exc:
            return True, None
        
        inc_elements = self._parse_elements(el_inc)
        exc_elements = self._parse_elements(el_exc)
        
        conflicts = inc_elements & exc_elements
        if conflicts:
            return False, f"Elements cannot be both included and excluded: {conflicts}"
        
        return True, None
    
    # def rule_expand_option_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
    #     """Validate expand parameter options"""
    #     expand = params.get('expand')
        
    #     if not expand:
    #         return True, None
        
    #     options = {opt.strip() for opt in expand.split(',') if opt.strip()}
    #     invalid = options - self.valid_expand_options
        
    #     if invalid:
    #         return False, f"Invalid expand options: {invalid}. Valid: {self.valid_expand_options}"
        
    #     return True, None
    
    def rule_completeness_validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """
        Validate that at least one search criterion is provided.
        
        Purpose: Ensures the query has at least one parameter to constrain the search.
        Example failure: {} (empty dict) -> Error: no search criteria specified
        Example failure: {} (all fields are None) -> Error: no search criteria specified
        Example success: {"ima": True} -> Has at least one search criterion
        Example success: {"hardness_min": 3, "el_inc": "Fe"} -> Has search criteria
        
        Note: All fields in search_fields (including 'ima') are considered valid search criteria.
        """
        has_criteria = any(
            params.get(field) is not None 
            for field in self.search_fields
        )
        
        if not has_criteria:
            return False, "At least one search criterion must be specified"
        
        return True, None
    
    def apply_corrections(self, params: dict) -> dict:
        """
        Apply automatic corrections to parameters.
        
        Args:
            params: Parameters to correct
        
        Returns:
            Corrected parameters
        """
        corrected = params.copy()
        
        # Normalize and correct case for element strings
        for field in ['el_inc', 'el_exc']:
            if corrected.get(field):
                corrected[field] = self._correct_element_case(corrected[field])
        
        return corrected
    
    def run_validation(self, params: dict) -> Dict[str, Any]:
        """
        Run all rule validators and return structured results.
        
        Args:
            params: Parameters to validate
        
        Returns:
            {
                "status": "valid" | "invalid",
                "issues": {  # Only if status is "invalid"
                    "rule_name_1": "error message",
                    "rule_name_2": "error message"
                },
                "corrected_params": dict  # Only if status is "valid"
            }
        """
        issues = {}
        
        for name in self.validators:
            is_valid, error = self.validators[name](params)
            if not is_valid:
                # Use validator name as key
                issues[name] = error
        
        # Build structured result
        if len(issues) == 0:
            # Apply corrections when validation passes
            corrected_params = self.apply_corrections(params)
            return {
                "status": "valid",
                "corrected_params": corrected_params
            }
        else:
            return {
                "status": "invalid",
                "issues": issues
            }

if __name__ == "__main__":
    pass
    # validator = RuleValidator(valid_fields={
    #     'ima', 'hardness_min', 'hardness_max', 'crystal_system', 'el_inc', 'el_exc'
    # })
    # test_params = {
    #     "ima": True,
    #     "hardness_min": 3,
    #     # "hardness_max": 11,  # Invalid
    #     # "crystal_system": ["Hexagonal", "InvalidSystem"],  # Invalid
    #     # "el_inc": "Fe,Cu,Xx",  # Invalid
    #     "el_exc": "S,Fe"  # Conflict with el_inc
    # }
    # result = validator.run_validation(test_params)
    # print(result)