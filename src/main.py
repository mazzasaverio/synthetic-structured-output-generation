# src/main.py

import os
import uuid
from openai import AzureOpenAI
import pandas as pd
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

from utils import load_config, load_prompts
from data_generator import generate_synthetic_data, llm_generation_data


def save_dataset(data: pd.DataFrame, output_path: str):
    data.to_excel(output_path, index=False)


def main():
    # Load environment variables
    load_dotenv()
    repo_root = os.getenv("REPO_ROOT")

    # Initialize logging
    logger.add("data_generation.log", rotation="1 MB", level="INFO")

    config_path = os.path.join(repo_root, 'config', 'settings.yaml')
    prompts_path = os.path.join(repo_root, 'config', 'prompts.yaml')

    try:
        logger.info("Starting data generation process.")
        total_records, output_base_name, columns = load_config(config_path)
        prompts = load_prompts(prompts_path)
        logger.info("Configuration loaded successfully.")

        data = generate_synthetic_data(columns, total_records)
        logger.info("Synthetic data generated.")


        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not azure_endpoint or not azure_key or not azure_api_version:
            
            logger.error("Azure OpenAI credentials must be set in the .env file.")
            raise EnvironmentError("Azure OpenAI credentials are not set.")

        client = AzureOpenAI(
            azure_endpoint = azure_endpoint, 
            api_key=azure_key,  
            api_version=azure_api_version
        )
        

        logger.info("Azure OpenAI client initialized.")

        df_llm = llm_generation_data(data, columns, client, prompts)
        logger.info("Open string fields filled using Azure OpenAI.")

        # Overwrite data with df_llm columns
        for col in df_llm.columns:
            data[col] = df_llm[col]

        data.insert(0, 'id', [str(uuid.uuid4()) for _ in range(len(data))])
        # Prepare output path
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{output_base_name}_{now}.xlsx"
        output_dir = os.path.join(repo_root, 'data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        save_dataset(data, output_path)
        logger.info(f"Dataset saved to {output_path}.")

    except Exception as e:
        logger.exception("An error occurred during the data generation process.")
        raise e


if __name__ == "__main__":
    main()
