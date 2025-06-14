# Text-to-SQL Data Analysis Chatbot

A Streamlit-based web application that allows users to upload CSV datasets and query them using natural language, which is automatically converted to SQL queries using AI.

## Features

- Upload CSV files and automatically convert them to SQL tables
- Ask questions in plain English about your data
- AI-powered natural language to SQL query conversion using Groq LLM
- Execute custom SQL queries with built-in SQL terminal
- Download query results as CSV files
- Interactive data preview and column information display

## Technology Stack

- **Frontend**: Streamlit
- **Database**: SQLite (in-memory)
- **AI/LLM**: Groq API with Llama3-8b model
- **Data Processing**: Pandas
- **Language**: Python

## Setup and Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install streamlit pandas langchain-groq sqlite3
   ```
3. Set up environment variable:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a CSV file using the sidebar file uploader
2. Set a table name and click "Load Data"
3. Ask questions about your data in natural language
4. View the generated SQL query and results
5. Optionally run custom SQL queries using the SQL terminal

## Project Structure

```
├── app.py              # Main application file
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies (if needed)
```

## Example Queries

- "How many records are in the table?"
- "Show me the first 10 rows"
- "What is the average value of column X?"
- "Filter records where column Y is greater than 100"
- "Group by column Z and count the records"

## Deployment

The application is designed for cloud deployment with in-memory storage and caching optimizations. No persistent database setup required.