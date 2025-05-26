import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool  # Using langchain_core imports for newer versions

logger = logging.getLogger(__name__)

class StockDataProcessor:
    """Processes and manages stock price data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load stock price data from CSV."""
        try:
            if self.data_path.exists() and self.data_path.stat().st_size > 0:
                self.data = pd.read_csv(self.data_path)
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                logger.info(f"Loaded stock data with {len(self.data)} rows")
            else:
                logger.warning(f"Stock data file empty or non-existent: {self.data_path}")
                self.data = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            self.data = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    def get_price_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 10) -> pd.DataFrame:
        """Get stock price data for a specific ticker within date range."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        ticker = ticker.upper()
        filtered_data = self.data[self.data['Ticker'] == ticker]
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data['Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data['Date'] <= end_date]
            
        return filtered_data.sort_values('Date', ascending=False).head(limit)
    
    def get_latest_price(self, ticker: str) -> Dict[str, Any]:
        """Get the latest price data for a ticker."""
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
        """Get price data for a ticker on a specific date."""
        if self.data is None or self.data.empty:
            return {}
            
        ticker = ticker.upper()
        try:
            date_obj = pd.to_datetime(date)
        except:
            return {}
        
        # Ensure the Date column is already converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['Date']):
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Compare dates without time information
        date_str = date_obj.strftime('%Y-%m-%d')
        date_data = self.data[(self.data['Ticker'] == ticker) & 
                            (self.data['Date'].dt.strftime('%Y-%m-%d') == date_str)]
        
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
        """Get list of all available tickers in the dataset."""
        if self.data is None or self.data.empty:
            return []
        return self.data['Ticker'].unique().tolist()


# Define a Pydantic schema for the tool's input for Pydantic v2 compatibility
class StockQuerySchema(BaseModel):
    query: str = Field(description="The natural language query about stock prices")


# Modified version for newer LangChain versions with Pydantic v2
class StockDataQueryTool(BaseTool):
    """Tool for querying stock price data."""
    
    name: str = "stock_data_query"
    description: str = "Query stock price data for specific tickers and date ranges"
    args_schema: Type[BaseModel] = StockQuerySchema
    stock_data_processor: StockDataProcessor = None
    
    def __init__(self, stock_data_processor: StockDataProcessor):
        """Initialize the tool with a StockDataProcessor."""
        self.stock_data_processor = stock_data_processor
        self.available_tickers = self.stock_data_processor.get_available_tickers()
        super().__init__()
    
    def _run(self, query: str) -> str:
        """Query stock price data based on a natural language query."""
        try:
            # Convert query to lowercase for easier parsing
            query_lower = query.lower()
            
            # Extract ticker
            ticker = self._extract_ticker(query)
            if not ticker:
                return "Could not identify a valid stock ticker in your query. Please specify a ticker symbol like AAPL, MSFT, etc."
            
            # Check if we want the latest data
            if any(term in query_lower for term in ["latest", "recent", "current", "today", "now"]):
                price_data = self.stock_data_processor.get_latest_price(ticker)
                if not price_data:
                    return f"No price data available for {ticker}."
                
                return f"Latest price data for {ticker}:\n" + json.dumps(price_data, indent=2)
            
            # Check for specific date request
            date_info = self._extract_date(query_lower)
            if date_info:
                price_data = self.stock_data_processor.get_price_on_date(ticker, date_info)
                if price_data:
                    return f"Price data for {ticker} on {date_info}:\n" + json.dumps(price_data, indent=2)
                else:
                    return f"No price data available for {ticker} on {date_info}."
            
            # Check for date range
            date_range = self._extract_date_range(query_lower)
            if date_range:
                start_date, end_date = date_range
                price_data = self.stock_data_processor.get_price_data(ticker, start_date, end_date)
                if price_data.empty:
                    return f"No price data available for {ticker} between {start_date} and {end_date}."
                
                result = price_data.head(10).to_dict(orient='records')
                return f"Price data for {ticker} from {start_date} to {end_date}:\n" + json.dumps(result, indent=2)
            
            # Default to returning a range of recent data
            price_data = self.stock_data_processor.get_price_data(ticker, limit=5)
            if price_data.empty:
                return f"No price data available for {ticker}."
            
            result = price_data.to_dict(orient='records')
            return f"Recent price data for {ticker}:\n" + json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error querying stock data: {e}")
            return f"Error querying stock data: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run - for BaseTool compatibility."""
        # Just delegate to the synchronous version
        return self._run(query)
    
    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract ticker symbol from query."""
        # First check if any ticker is mentioned directly
        for ticker in self.available_tickers:
            pattern = rf'\b{ticker}\b'
            if re.search(pattern, query, re.IGNORECASE):
                return ticker.upper()
        
        # List of common ticker keywords to check
        ticker_keywords = ["ticker", "symbol", "stock", "shares", "company"]
        for keyword in ticker_keywords:
            if keyword in query.lower():
                # Look for words after the keyword that might be tickers
                pattern = rf'{keyword}\s+(\w+)'
                match = re.search(pattern, query.lower())
                if match:
                    potential_ticker = match.group(1).upper()
                    if potential_ticker in self.available_tickers:
                        return potential_ticker
        
        return None
    
    def _extract_date(self, query: str) -> Optional[str]:
        """Extract a specific date from query."""
        # Check for date formats like YYYY-MM-DD
        iso_date_pattern = r'\b(\d{4}-\d{1,2}-\d{1,2})\b'
        iso_match = re.search(iso_date_pattern, query)
        if iso_match:
            try:
                date_str = iso_match.group(1)
                return date_str
            except:
                pass
        
        # Check for date formats like MM/DD/YYYY or DD/MM/YYYY
        slash_date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{4})\b'
        slash_match = re.search(slash_date_pattern, query)
        if slash_match:
            try:
                date_str = slash_match.group(1)
                # Try to parse as MM/DD/YYYY
                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                return date_obj.strftime('%Y-%m-%d')
            except:
                try:
                    # Try to parse as DD/MM/YYYY
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    pass
        
        # Check for natural language dates
        month_names = ["january", "february", "march", "april", "may", "june", 
                       "july", "august", "september", "october", "november", "december"]
        month_abbr = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        
        # Map month names to month numbers
        month_map = {**{name: i+1 for i, name in enumerate(month_names)},
                    **{abbr: i+1 for i, abbr in enumerate(month_abbr)}}
        
        # Check for patterns like "January 15, 2023" or "15 January 2023"
        for month_pattern in month_map:
            # Pattern for "Month Day, Year"
            pattern1 = rf'{month_pattern}\s+(\d{{1,2}})[,\s]+(\d{{4}})'
            match = re.search(pattern1, query)
            if match:
                try:
                    day, year = int(match.group(1)), int(match.group(2))
                    month = month_map[month_pattern]
                    return f"{year}-{month:02d}-{day:02d}"
                except:
                    pass
            
            # Pattern for "Day Month Year"
            pattern2 = rf'(\d{{1,2}})\s+{month_pattern}[,\s]+(\d{{4}})'
            match = re.search(pattern2, query)
            if match:
                try:
                    day, year = int(match.group(1)), int(match.group(2))
                    month = month_map[month_pattern]
                    return f"{year}-{month:02d}-{day:02d}"
                except:
                    pass
        
        return None
    
    def _extract_date_range(self, query: str) -> Optional[tuple]:
        """Extract date range from query."""
        # Check for common patterns indicating date ranges
        range_keywords = ["between", "from", "since", "starting", "ending"]
        
        if any(keyword in query for keyword in range_keywords):
            # Try to find two dates in the query
            dates = []
            
            # First try ISO format dates (YYYY-MM-DD)
            iso_dates = re.findall(r'\b(\d{4}-\d{1,2}-\d{1,2})\b', query)
            dates.extend(iso_dates)
            
            # Then try slash format dates (MM/DD/YYYY or DD/MM/YYYY)
            slash_dates = re.findall(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', query)
            for date_str in slash_dates:
                try:
                    # Try MM/DD/YYYY
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    dates.append(date_obj.strftime('%Y-%m-%d'))
                except:
                    try:
                        # Try DD/MM/YYYY
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                        dates.append(date_obj.strftime('%Y-%m-%d'))
                    except:
                        pass
            
            # If we found at least two dates, use them as a range
            if len(dates) >= 2:
                return (dates[0], dates[1])
        
        return None