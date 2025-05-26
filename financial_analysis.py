import os
import re
import csv
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional, Type
from langchain.tools import BaseTool
from langchain.embeddings import HuggingFaceEmbeddings, VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TRANSCRIPTS_DIR = Path("data/transcripts")
STOCK_DATA_PATH = Path("data/stock_prices.csv")

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
STOCK_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

if not STOCK_DATA_PATH.exists():
    with open(STOCK_DATA_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
        logger.info(f"Created empty stock data file at {STOCK_DATA_PATH}")


class TranscriptProcessor:
    def __init__(self, transcripts_dir: Path):
        self.transcripts_dir = transcripts_dir
        self.transcripts = {}
        self.load_transcripts()
        
    def load_transcripts(self):
        """Load all transcripts from the directory."""
        if not self.transcripts_dir.exists():
            logger.warning(f"Transcripts directory {self.transcripts_dir} does not exist.")
            return
            
        for file_path in self.transcripts_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    ticker = file_path.stem.split('_')[0]
                    self.transcripts[file_path.name] = {
                        "ticker": ticker,
                        "content": content,
                        "path": file_path
                    }
                logger.info(f"Loaded transcript: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading transcript {file_path}: {e}")
    
    def get_transcript_list(self) -> List[str]:
        """Get list of available transcripts."""
        return list(self.transcripts.keys())
    
    def get_transcript(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get a specific transcript by filename."""
        return self.transcripts.get(filename)
    
    def get_transcripts_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """Get all transcripts for a specific ticker."""
        ticker = ticker.upper()
        return [t for t in self.transcripts.values() if t["ticker"] == ticker]
    
    def search_transcripts(self, query: str) -> List[Dict[str, Any]]:
        try:
    
            db_directory = os.path.join(os.path.dirname(self.transcripts_dir), "transcript_vectordb")

            model_name = "all-MiniLM-L6-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            if not os.path.exists(db_directory):
                logger.info("Creating new vector database for transcripts...")
                       
                documents = []
                for filename, transcript in self.transcripts.items():
                    metadata = {
                        "filename": filename,
                        "ticker": transcript["ticker"]
                    }
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    chunks = text_splitter.split_text(transcript["content"])
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = i
                        documents.append((chunk, chunk_metadata))
                
                vectordb = Chroma.from_texts(
                    texts=[doc[0] for doc in documents],
                    embedding=embeddings,
                    metadatas=[doc[1] for doc in documents],
                    persist_directory=db_directory
                )
                vectordb.persist()
            else:
                logger.info("Loading existing transcript vector database...")
                vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
       
            search_results = vectordb.similarity_search_with_score(
                query=query,
                k=5 
            )
            if not search_results:
                raise ValueError(f"No relevant results found for query: {query}")
            
            results = {}
            for doc, score in search_results:
                filename = doc.metadata["filename"]
                ticker = doc.metadata["ticker"]

                if filename not in results:
                    results[filename] = {
                        "filename": filename,
                        "ticker": ticker,
                        "snippets": [],
                        "relevance_score": 0
                    }
                
                snippet = f"...{doc.page_content}..."
                results[filename]["snippets"].append(snippet)
                
                relevance = 1 - score  
                results[filename]["relevance_score"] = max(results[filename]["relevance_score"], relevance)
            
            sorted_results = sorted(results.values(), key=lambda x: x["relevance_score"], reverse=True)
            
            return sorted_results
        
        except Exception as e:
            logger.error(f"Error during vector search of transcripts: {e}")
            raise
    
    def _extract_snippets(self, text: str, query: str, context_size: int = 200) -> List[str]:
        query = query.lower()
        text_lower = text.lower()
        snippets = []
        
        start = 0
        while True:
            pos = text_lower.find(query, start)
            if pos == -1:
                break
                
            snippet_start = max(0, pos - context_size)
            snippet_end = min(len(text), pos + len(query) + context_size)
            snippet = text[snippet_start:snippet_end]
            snippets.append(f"...{snippet}...")
            
            start = pos + len(query)
            
        return snippets
    
class TranscriptQueryTool(BaseTool):
    
    name: str = "Transcript Query Tool"
    description: str = "Query earnings call transcripts for specific information"
    
    transcript_processor: TranscriptProcessor = None
    model: str = None
    
    def __init__(self, transcript_processor: TranscriptProcessor):
        super().__init__() 
        self.transcript_processor = transcript_processor
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def _run(self, query: str) -> str:
        try:
            results = self.transcript_processor.search_transcripts(query)
            
            if not results:
                return "No relevant information found in any transcript."
            formatted_results = []
            for result in results:
                formatted_results.append(f"From {result['ticker']} transcript ({result['filename']}):")
                for i, snippet in enumerate(result['snippets'], 1):
                    formatted_results.append(f"  Snippet {i}: {snippet}")
        
            prompt = f"""
            Analyze the following excerpts from earnings call transcripts related to the query: "{query}"
            
            Transcript excerpts:
            {formatted_results}
            
            Please provide:
            1. Accurate response to user's query based on the data provided.
            
            Be concise and focus on the most relevant information to a financial analyst.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error querying transcripts: {e}")
            return f"Error querying transcripts: {e}"
        

logger = logging.getLogger(__name__)

class StockDataProcessor:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = None
        self.load_data()

    def load_data(self):
        try:
            if self.data_path.exists() and self.data_path.stat().st_size > 0:
                self.data = pd.read_csv(self.data_path)
                self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
                logger.info(f"Loaded stock data with {len(self.data)} rows")
            else:
                logger.warning(f"Stock data file empty or non-existent: {self.data_path}")
                self.data = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            self.data = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])

    def get_price_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 10) -> pd.DataFrame:
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        ticker = ticker.upper()
        filtered_data = self.data[self.data['Ticker'] == ticker]
        if start_date:
            self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
            filtered_data = filtered_data[filtered_data['Date'] >= start_date]
        if end_date:
            self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
            filtered_data = filtered_data[filtered_data['Date'] <= end_date]
        return filtered_data.sort_values('Date', ascending=False).head(limit)

    def get_latest_price(self, ticker: str) -> Dict[str, Any]:
        if self.data is None or self.data.empty:
            return {}
        ticker = ticker.upper()
        ticker_data = self.data[self.data['Ticker'] == ticker]
        if ticker_data.empty:
            return {}
        latest = ticker_data.sort_values('Date', ascending=False).iloc[0]
        return {
            'Date': latest['Date'].strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'Open': float(latest['Open']),
            'High': float(latest['High']),
            'Low': float(latest['Low']),
            'Close': float(latest['Close']),
            'Volume': int(latest['Volume'])
        }

    def get_price_on_date(self, ticker: str, date: str) -> Dict[str, Any]:
        if self.data is None or self.data.empty:
            return {}
        ticker = ticker.upper()
        try:
            date_obj = pd.to_datetime(date)
        except:
            return {}
        if not pd.api.types.is_datetime64_any_dtype(self.data['Date']):
            self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
        date_str = date_obj.strftime('%Y-%m-%d')
        date_data = self.data[(self.data['Ticker'] == ticker) & (self.data['Date'].dt.strftime('%Y-%m-%d') == date_str)]
        if date_data.empty:
            return {}
        row = date_data.iloc[0]
        return {
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'Open': float(row['Open']),
            'High': float(row['High']),
            'Low': float(row['Low']),
            'Close': float(row['Close']),
            'Volume': int(row['Volume'])
        }

    def get_available_tickers(self) -> List[str]:
        if self.data is None or self.data.empty:
            return []
        return self.data['Ticker'].unique().tolist()
    
class StockQueryInput(BaseModel):
    query: str = Field(..., description="User query for stock price data")

class StockDataQueryTool(BaseTool):
    
    name: str = "stock_data_query"
    description: str = "Query stock price data for specific tickers and date ranges"
    args_schema: type = StockQueryInput
    stock_data_processor: StockDataProcessor = None
    available_tickers: List[str] = None

    def __init__(self, stock_data_processor: StockDataProcessor):
        super().__init__()
        self.stock_data_processor = stock_data_processor
        self.available_tickers = stock_data_processor.get_available_tickers()

    def _run(self, query: str) -> str:
        try:
            query_lower = query.lower()
            ticker = self._extract_ticker(query)
            if not ticker:
                return "Could not identify a valid stock ticker."

            if any(term in query_lower for term in ["latest", "recent", "current", "today", "now"]):
                price_data = self.stock_data_processor.get_latest_price(ticker)
                if not price_data:
                    return f"No price data available for {ticker}."
                return f"Latest price data for {ticker}:\n" + json.dumps(price_data, indent=2)

            date_info = self._extract_date(query_lower)
            if date_info:
                price_data = self.stock_data_processor.get_price_on_date(ticker, date_info)
                if price_data:
                    return f"Price data for {ticker} on {date_info}:\n" + json.dumps(price_data, indent=2)
                else:
                    return f"No price data for {ticker} on {date_info}."

            date_range = self._extract_date_range(query_lower)
            if date_range:
                start_date, end_date = date_range
                price_data = self.stock_data_processor.get_price_data(ticker, start_date, end_date)
                if price_data.empty:
                    return f"No price data for {ticker} between {start_date} and {end_date}."
                result = price_data.head(10).to_dict(orient='records')
                return f"Price data for {ticker} from {start_date} to {end_date}:\n" + json.dumps(result, indent=2)

            price_data = self.stock_data_processor.get_price_data(ticker, limit=5)
            if price_data.empty:
                return f"No recent price data for {ticker}."
            result = price_data.to_dict(orient='records')
            return f"Recent price data for {ticker}:\n" + json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error querying stock data: {e}")
            return f"Error: {str(e)}"

    def _extract_ticker(self, query: str) -> Optional[str]:
        for ticker in self.available_tickers:
            pattern = rf'\b{ticker}\b'
            if re.search(pattern, query, re.IGNORECASE):
                return ticker.upper()
        ticker_keywords = ["ticker", "symbol", "stock", "shares", "company"]
        for keyword in ticker_keywords:
            if keyword in query.lower():
                pattern = rf'{keyword}\s+(\w+)'
                match = re.search(pattern, query.lower())
                if match:
                    potential_ticker = match.group(1).upper()
                    if potential_ticker in self.available_tickers:
                        return potential_ticker
        return None

    def _extract_date(self, query: str) -> Optional[str]:
        iso_date_pattern = r'\b(\d{4}-\d{1,2}-\d{1,2})\b'
        iso_match = re.search(iso_date_pattern, query)
        if iso_match:
            return iso_match.group(1)
        slash_date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{4})\b'
        slash_match = re.search(slash_date_pattern, query)
        if slash_match:
            date_str = slash_match.group(1)
            for fmt in ('%m/%d/%Y', '%d/%m/%Y'):
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                except:
                    continue
        return None

    def _extract_date_range(self, query: str) -> Optional[List[str]]:
        date_range_pattern = r'\b(\d{4}-\d{1,2}-\d{1,2})\s+to\s+(\d{4}-\d{1,2}-\d{1,2})\b'
        match = re.search(date_range_pattern, query)
        if match:
            return [match.group(1), match.group(2)]
        return None


def create_agents_and_tools():
    transcript_processor = TranscriptProcessor(TRANSCRIPTS_DIR)
    stock_data_processor = StockDataProcessor(STOCK_DATA_PATH)
    
    transcript_tool = TranscriptQueryTool(transcript_processor)
    stock_data_tool = StockDataQueryTool(stock_data_processor)
    
    researcher = Agent(
        role="Financial Researcher",
        goal="Research and gather information from multiple sources to answer financial questions",
        backstory="You are an expert financial researcher with years of experience analyzing earnings calls and stock data.",
        verbose=True,
        allow_delegation=True,
        tools=[transcript_tool, stock_data_tool],
        llm=genai.GenerativeModel('gemini/gemini-2.0-flash'),
    )
    
    analyst = Agent(
        role="Financial Analyst",
        goal="Analyze and synthesize information from multiple sources to provide insightful financial analysis",
        backstory="You are a senior financial analyst with expertise in interpreting earnings calls, stock data, and news to provide actionable insights.",
        verbose=True,
        allow_delegation=True,
        tools=[transcript_tool, stock_data_tool],
        llm=genai.GenerativeModel('gemini/gemini-2.0-flash'),
    )
    
    return researcher, analyst, transcript_tool, stock_data_tool

def create_tasks(researcher, analyst):
    """Create tasks for the financial research assistant."""
    research_task = Task(
        description="Research and gather information from multiple sources to answer a financial question",
        agent=researcher,
        expected_output="A comprehensive collection of relevant information from earnings call transcripts and stock data"
    )
    
    analysis_task = Task(
        description="Analyze and synthesize the collected information to provide a comprehensive answer",
        agent=analyst,
        expected_output="A comprehensive analysis that synthesizes information from multiple sources"
    )
    
    return [research_task, analysis_task]


def run_financial_assistant(query: str):
    try:
        researcher, analyst, transcript_tool, stock_data_tool = create_agents_and_tools()
        
        tasks = create_tasks(researcher, analyst)
        
        tasks[0].description = f"Research and gather information from multiple sources to answer the following question: '{query}'"
        tasks[1].description = f"Analyze and synthesize the collected information to provide a comprehensive answer to: '{query}'"
        
        crew = Crew(
            agents=[researcher, analyst],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return result
    except Exception as e:
        logger.error(f"Error running financial assistant: {e}")
        return f"Error running financial assistant: {e}"

def upload_data_handler(file):

    return {"filename": file.filename}

def remove_data_handler(identifier: str):

    return {"removed": identifier}