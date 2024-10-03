
# Synthetic Structured Output Generation

## Overview

This project provides a framework for generating structured synthetic datasets based on configurable schemas. It leverages **Pydantic** for data validation and **OpenAI's API** to generate realistic data for open string fields. The tool is designed to be highly customizable, allowing you to define:

- **Data Schemas**: Specify the fields, types, distributions, and dependencies.
- **OpenAI Prompts**: Define prompts for generating realistic data using OpenAI's language models.
- **Configuration Files**: Use YAML files to configure the data generation process.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- An Azure OpenAI account with appropriate API keys and deployment names.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/synthetic-structured-output-generation.git
   cd synthetic-structured-output-generation
   ```

2. **Create a Virtual Environment and Install Dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   - Copy `.env.example` to `.env`:

     ```bash
     cp .env.example .env
     ```

   - Edit the `.env` file to include your actual Azure OpenAI endpoint, key, and deployment name.

     ```dotenv
     AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
     AZURE_OPENAI_KEY=your_azure_openai_key
     AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
     ```

### Configuration

#### **1. settings.yaml**

Define your data schema and generation settings in `config/settings.yaml`.

```yaml
# config/settings.yaml

total_records: 1000  # Total number of records to generate
batch_size: 50       # Batch size for OpenAI data generation
output_base_name: 'synthetic_dataset'  # Base name of the output file

columns:
  - name: name
    description: "Person's first name"
    type: str
    nullable: false
    openai_generation: true
    prompt_type: "name_generation"

  - name: surname
    description: "Person's last name"
    type: str
    nullable: false
    openai_generation: true
    prompt_type: "surname_generation"

  # Add additional columns as needed
```

- **total_records**: Number of records to generate.
- **batch_size**: Batch size for OpenAI API calls.
- **columns**: List of column definitions with their properties.

#### **2. prompts.yaml**

Define OpenAI prompts for generating data in `config/prompts.yaml`.

```yaml
# config/prompts.yaml

prompts:
  name_generation:
    model: "gpt-3.5-turbo"
    messages:
      - role: "system"
        content: "You are a data generator that creates realistic first names."
      - role: "user"
        content: "Generate a realistic first name."
    response_format:
      type: "string"

  surname_generation:
    model: "gpt-3.5-turbo"
    messages:
      - role: "system"
        content: "You are a data generator that creates realistic last names."
      - role: "user"
        content: "Generate a realistic last name."
    response_format:
      type: "string"

  # Add additional prompts as needed
```

### Running the Data Generation

Execute the main script to start the data generation process:

```bash
python src/main.py
```

The generated dataset will be saved in the `data/` directory with a timestamped filename.

## Usage Example

After configuring your `settings.yaml` and `prompts.yaml`, running the script will generate a synthetic dataset based on your specifications.


## Environment Variables

Set the following environment variables in your `.env` file:

- **AZURE_OPENAI_ENDPOINT**: Your Azure OpenAI endpoint.
- **AZURE_OPENAI_KEY**: Your Azure OpenAI API key.
- **AZURE_OPENAI_DEPLOYMENT_NAME**: Your Azure OpenAI deployment name.

## Logging

Logs are written to `data_generation.log` with rotation after reaching 1 MB. Adjust logging settings in `main.py` as needed.

## Error Handling

The code includes error handling for:

- Configuration loading errors.
- OpenAI API errors.
- Data generation issues.

Errors are logged, and the process continues where possible.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact [your email or GitHub username].