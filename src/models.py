from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ValidationInfo, field_validator

class DistributionConfig(BaseModel):
    type: str
    parameters: Optional[Dict[str, Any]] = None
    percentages: Optional[Dict[Any, float]] = None

class DependencyConfig(BaseModel):
    field: str
    value: Optional[Any] = None
    value_in: Optional[List[Any]] = None
    condition: Optional[str] = None  # e.g., 'not in'

class ColumnConfig(BaseModel):
    name: str
    description: Optional[str]
    type: str  # 'int', 'float', 'str', 'list of str'
    domain: Optional[Any] = None  # List or str indicating a predefined list
    distribution: Optional[DistributionConfig] = None
    correlated_with: Optional[Dict[str, float]] = None  # {field_name: correlation_coefficient}
    selection_type: Optional[str] = None  # 'single', 'multiple'
    nullable: bool = False
    dependencies: Optional[List[DependencyConfig]] = None
    format: Optional[str] = None  # For date fields
    openai_generation: Optional[bool] = False  # Indicates if OpenAI should generate this field
    prompt_type: Optional[str] = None  # Key to select the prompt from prompts.yaml

    @field_validator('type')
    def validate_type(cls, v: str, info: ValidationInfo) -> str:
        if v not in {'int', 'float', 'str', 'list of str'}:
            raise ValueError("Invalid type. Must be 'int', 'float', 'str', or 'list of str'.")
        return v

    @field_validator('selection_type')
    def validate_selection_type(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if 'type' in info.data and info.data['type'] in {'str', 'list of str'}:
            if v not in {'single', 'multiple', None}:
                raise ValueError("Invalid selection_type. Must be 'single' or 'multiple'.")
        return v
