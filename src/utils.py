
import os
import yaml
from openai import AzureOpenAI

from dotenv import load_dotenv
from loguru import logger
from typing import Tuple, List
from models import ColumnConfig


# utils.py

def load_config(config_path: str) -> Tuple[int, int, str, List[ColumnConfig]]:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    total_records = config_data.get('total_records', 1000)
    output_base_name = config_data.get('output_base_name', 'synthetic_dataset')
    predefined_domains = config_data.get('predefined_domains', {})
    columns_data = config_data['columns']
    
    columns = []
    for col_data in columns_data:
        if isinstance(col_data.get('domain'), str):
            domain_name = col_data['domain']
            if domain_name in predefined_domains:
                col_data['domain'] = predefined_domains[domain_name]
            else:
                raise ValueError(f"Domain '{domain_name}' not found in predefined_domains")
        columns.append(ColumnConfig(**col_data))
    return total_records, output_base_name, columns


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, 'r') as file:
        prompts_data = yaml.safe_load(file)
    return prompts_data['prompts']


