# Jeep Patriot Diagnostic Assistant

A LangGraph-powered diagnostic assistant for 2011 Jeep Patriots that uses the official manual to help diagnose vehicle issues.

## Features

- **Manual-First Approach**: Always consults the official 2011 Jeep Patriot manual before providing diagnostics
- **LangGraph Workflow**: Uses a structured workflow to process queries and generate responses
- **OpenAI GPT-4o Mini**: Powered by OpenAI's latest efficient model
- **Interactive CLI**: Easy-to-use command-line interface

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API Key**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

3. **Ensure Manual is Present**:
   - The `2011-patriot manual.pdf` should be in the project root directory

## Usage

Run the diagnostic assistant:

```bash
python main.py
```

Then describe any issues you're experiencing with your Jeep Patriot. The assistant will:

1. Read and process the manual content
2. Analyze your query to identify relevant systems
3. Search the manual for applicable information
4. Generate a comprehensive diagnostic response
5. Provide recommendations based on official documentation

## Example Queries

- "My engine is making a knocking sound"
- "The transmission is slipping when shifting"
- "Air conditioning not cooling properly"
- "Check engine light is on"
- "Brakes feel spongy"

## Architecture

The application uses a LangGraph workflow with the following nodes:

1. **Read Manual**: Extracts and organizes content from the PDF
2. **Analyze Query**: Understands the user's issue and identifies relevant systems
3. **Search Manual**: Finds applicable sections in the manual
4. **Generate Diagnosis**: Creates comprehensive diagnostic response
5. **Format Response**: Prepares the final output

## Files

- `main.py`: Main application entry point
- `patriot_agent.py`: LangGraph workflow and agent logic
- `pdf_reader.py`: PDF processing and content extraction
- `requirements.txt`: Python dependencies
- `.env.example`: Environment variable template
# langGraph-jeep-assistant
