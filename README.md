# Financial_Agent

A smart research assistant for financial analysts that helps answer complex questions by combining information from earnings calls, stock prices, and news reports.

## Features

- **Heterogeneous Data Handling**: Processes both unstructured text (transcripts) and structured data (CSV).
- **Advanced RAG**: Retrieves information from the appropriate source(s) based on user queries.
- **Agentic Planning & Tool Use**: Analyzes queries, determines which information sources are needed, and plans a sequence of tool uses.
- **Information Synthesis**: Integrates diverse pieces of information into coherent answers.


## Architecture

The system is built using CrewAI and the Gemini API, with the following components:

1. **Data Processors**:
   - `TranscriptProcessor`: Handles earnings call transcripts
   - `StockDataProcessor`: Handles stock price data

2. **Tools**:
   - `TranscriptQueryTool`: Searches and analyzes transcript data
   - `StockDataQueryTool`: Queries and processes stock price data

3. **Agents**:
   - `Financial Researcher`: Gathers information from multiple sources
   - `Financial Analyst`: Synthesizes information into comprehensive answers

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Financial_Agent.git
   cd Financial_Agent
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate         # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Create a `.env` file in the project root
   - Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`

## Project Structure
```
financial_Agent/
├── financial_analysis.py   # Core functionality
├── main.py                 # FastAPI endpoint
├── requirements.txt        # Project dependencies
├── data/                   # Data directory
│   ├── transcripts/        # Earnings call transcripts
│   └── stock_prices.csv    # Stock price data

```

## Run 

1. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```
This will start the server at: http://127.0.0.1:8000

2. Test the API

    Open your browser and go to: http://127.0.0.1:8000/docs
    
    This opens the interactive Swagger UI to test all available endpoints.

## Requirements

- Python 3.12
- CrewAI
- Google Generative AI (Gemini)
- Pandas
- Dotenv
- FastAPI