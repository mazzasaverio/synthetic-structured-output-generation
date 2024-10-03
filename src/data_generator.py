# src/data_generator.py

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from models import ColumnConfig
from loguru import logger
from openai import AzureOpenAI

from tqdm import tqdm


def generate_synthetic_data(columns: List[ColumnConfig], num_rows: int) -> pd.DataFrame:
    data = pd.DataFrame(index=range(num_rows))
    temp_numeric_data = {}

    for col in columns:
        if not should_generate_column(col, data):
            data[col.name] = np.nan
            continue

        if col.type in {'int', 'float'}:
            samples = generate_numeric_column(col, num_rows)
            data[col.name] = samples
            temp_numeric_data[col.name] = samples

        elif col.type == 'str':
            if col.domain and isinstance(col.domain, list):
                samples = generate_categorical_column(col, num_rows)
                data[col.name] = samples
            elif not col.openai_generation:
                # Generate placeholder data or leave as None
                data[col.name] = [f"{col.name}_{i}" for i in range(num_rows)]
            else:
                data[col.name] = None  # Placeholder, will be filled later

        elif col.type == 'list of str':
            samples = generate_categorical_column(col, num_rows)
            data[col.name] = samples

        # Handle nullable fields
        if col.nullable:
            apply_nulls(data, col.name)

    # Apply correlations
    data = apply_correlations(data, columns, temp_numeric_data, num_rows)

    return data


def should_generate_column(col: ColumnConfig, data: pd.DataFrame) -> bool:
    if not col.dependencies:
        return True
    for dep in col.dependencies:
        dep_field = dep.field
        if dep_field not in data.columns:
            return False
        dep_values = data[dep_field]
        if dep.value is not None and not (dep_values == dep.value).any():
            return False
        if dep.value_in is not None and not dep_values.isin(dep.value_in).any():
            return False
        if dep.condition == 'not in' and dep_values.isin(dep.value_in).all():
            return False
    return True


def generate_numeric_column(col: ColumnConfig, num_rows: int) -> np.ndarray:
    if col.distribution:
        dist_type = col.distribution.type
        params = col.distribution.parameters or {}
        
        # Include 'percentages' in params if distribution is 'discrete'
        if dist_type == 'discrete':
            if col.distribution.percentages:
                params['percentages'] = col.distribution.percentages
            else:
                raise ValueError(f"Missing 'percentages' for discrete distribution in column '{col.name}'")
        
        samples = generate_numeric_data(dist_type, params, num_rows)
        if col.type == 'int':
            samples = samples.astype(int)
    else:
        samples = np.zeros(num_rows)
    return samples



def generate_numeric_data(dist_type: str, params: Dict[str, Any], num_rows: int) -> np.ndarray:
    if dist_type == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        return np.random.normal(loc=mean, scale=std, size=num_rows)
    elif dist_type == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 1)
        return np.random.uniform(low=low, high=high, size=num_rows)
    elif dist_type == 'discrete':
        if 'percentages' not in params or not params['percentages']:
            raise ValueError("Missing or empty 'percentages' field in distribution parameters.")
        percentages = params['percentages']
        choices = list(percentages.keys())
        probabilities = [percentages[choice] / 100 for choice in choices]
        return np.random.choice(choices, size=num_rows, p=probabilities)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")



def generate_categorical_column(col: ColumnConfig, num_rows: int):
    if col.distribution and col.distribution.type == 'discrete':
        choices = col.domain
        percentages = col.distribution.percentages

        # Ensure domain and percentages are valid
        if not choices or not isinstance(choices, list):
            raise ValueError(f"Domain for column '{col.name}' must be a list. Please check the config.")
        if not percentages:
            raise ValueError(f"No percentages specified for column '{col.name}'. Please check the config.")

        # Calculate total percentage and guard against zero
        total_percentage = sum(percentages.get(choice, 0) for choice in choices)
        
        if total_percentage == 0:
            raise ValueError(f"Total percentage for column '{col.name}' is zero. Please check the config.")


        probabilities = [percentages.get(choice, 0) / total_percentage for choice in choices]

        if col.selection_type == 'single':
            return np.random.choice(choices, size=num_rows, p=probabilities).tolist()
        elif col.selection_type == 'multiple':
            return [
                np.random.choice(
                    choices,
                    size=np.random.randint(1, len(choices) + 1),
                    replace=False,
                    p=probabilities
                ).tolist()
                for _ in range(num_rows)
            ]
    else:
        if col.selection_type == 'single':
            return np.random.choice(col.domain, size=num_rows).tolist()
        elif col.selection_type == 'multiple':
            return [
                np.random.choice(
                    col.domain,
                    size=np.random.randint(1, len(col.domain) + 1),
                    replace=False
                ).tolist()
                for _ in range(num_rows)
            ]

def apply_nulls(data: pd.DataFrame, column_name: str, null_probability: float = 0.1):
    null_mask = np.random.rand(len(data)) < null_probability
    data.loc[null_mask, column_name] = None


def apply_correlations(data: pd.DataFrame, columns: List[ColumnConfig], temp_numeric_data: Dict[str, Any], num_rows: int) -> pd.DataFrame:
    for col in columns:
        if col.correlated_with and col.type in {'int', 'float'}:
            for corr_field, corr_coef in col.correlated_with.items():
                if corr_field in temp_numeric_data:
                    noise = np.random.normal(0, np.sqrt(1 - corr_coef ** 2), num_rows)
                    data[col.name] = corr_coef * temp_numeric_data[corr_field] + noise
    return data


def llm_generation_data(data: pd.DataFrame, columns: List[ColumnConfig], client: AzureOpenAI, prompts: dict) -> pd.DataFrame:
    num_samples = data.shape[0]
    generated_texts = generate_openai_data(client, prompts['type0'], num_samples)

    # Parse the JSON strings and create a DataFrame
    generated_dicts = [json.loads(text) for text in generated_texts]
    df_llm = pd.DataFrame(generated_dicts)
    return df_llm

import copy
def make_request_structured_output(client, config_prompt):
    model = config_prompt['model']
    messages = copy.deepcopy(config_prompt['messages'])
    response_format = config_prompt['response_format']
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=1
    )

    return response.choices[0].message.content



def generate_openai_data(client: AzureOpenAI, prompt_config: dict, num_samples: int) -> List[str]:
    generated_texts = []

    for i in tqdm(range(0, num_samples), desc="Generating data with OpenAI"):       
        try:

            content = make_request_structured_output(client, prompt_config)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            
        generated_texts.append(content)
    return generated_texts
