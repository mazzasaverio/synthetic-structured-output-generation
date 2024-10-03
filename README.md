
# Synthetic Structured Output Generation

This repository is designed to generate synthetic structured datasets with both predefined categorical values and fields generated using OpenAI's API. The configuration is done via YAML files that define the schema and metadata for the synthetic data generation, and certain fields can be filled using Azure OpenAI's language models.

## Features
- Generate synthetic datasets with specified column distributions, correlations, and dependencies.
- Integrate Azure OpenAI to generate realistic data for selected fields (e.g., names, cities).
- Support for a variety of data types, including numerical, categorical, and list types.
- Configurable prompts for each OpenAI API call, defined in the `prompts.yaml` file.
- Simple YAML configuration for dataset structure and generation logic.

## Repository Structure
```bash
synthetic-structured-output-generation/
├── config/                 # Configuration files
│   ├── prompts.yaml        # Prompts for OpenAI LLM generation
│   └── settings.yaml       # Schema and generation settings
├── config.example/         # Example configuration files (copy and rename to 'config/')
├── data/                   # Generated datasets
├── data_generation.log     # Log file
├── .env                    # Environment file (not tracked in Git)
├── .env.example            # Example environment file (copy and rename to '.env')
├── LICENSE                 # License information
├── README.md               # This README file
├── requirements.txt        # Python dependencies
└── src/                    # Source code
    ├── main.py             # Main script to run data generation
    ├── data_generator.py   # Core logic for generating synthetic data
    ├── models.py           # Data model definitions using Pydantic
    └── utils.py            # Utility functions (e.g., loading config, OpenAI client setup)
```

## Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/synthetic-structured-output-generation.git
cd synthetic-structured-output-generation
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup the environment

1. Copy `.env.example` to `.env` and set your Azure OpenAI credentials:

    ```bash
    cp .env.example .env
    ```

2. Copy the `config.example/` directory to `config/`:

    ```bash
    cp -r config.example/ config/
    ```

3. Fill in the appropriate values in the `.env` file for:
    - `AZURE_OPENAI_ENDPOINT`
    - `AZURE_OPENAI_KEY`
    - `AZURE_OPENAI_API_VERSION`

### Step 4: Running the data generation

After setting up the environment and configuration, you can generate synthetic data using:

```bash
python src/main.py
```

The generated dataset will be saved in the `data/` directory as an Excel file.

### Configuration

The dataset schema and generation logic are defined in the `config/settings.yaml` file. Fields can have predefined values (domains), distributions (e.g., normal, uniform), or be filled by Azure OpenAI.

You can also customize the prompts for OpenAI generation in `config/prompts.yaml`, specifying different models or prompt structures for each field.

## Logs

Logs of the data generation process are written to `data_generation.log`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.