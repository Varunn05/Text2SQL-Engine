import os
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import io
import sqlite3
import tempfile
import threading


# Set page config first
st.set_page_config(page_title="Text To SQL", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def process_uploaded_file(file_content, file_name):
    """Process uploaded CSV file and return DataFrame - cached for efficiency"""
    try:
        df = pd.read_csv(io.StringIO(file_content))
        # Clean column names for SQL compatibility
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
        return df, None
    except Exception as e:
        return None, str(e)


def get_database_connection():
    """Get a thread-safe database connection"""
    # Create a new connection for each thread
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    return conn


def setup_database_with_data(df, table_name):
    """Setup database with DataFrame data - returns connection"""
    try:
        conn = get_database_connection()
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        return conn, None
    except Exception as e:
        return None, str(e)


def get_table_info_from_dataframe(df, table_name):
    """Get table information from DataFrame (thread-safe)"""
    try:
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).tolist()
        row_count = len(df)
        sample_data = df.head(3).values.tolist()
        
        return {
            'columns': columns,
            'dtypes': dtypes,
            'row_count': row_count,
            'sample_data': sample_data
        }
    except Exception as e:
        return {}


def create_dynamic_sql_prompt(table_info, table_name):
    """Create a dynamic prompt based on table structure"""
    if not table_info:
        return """You are an expert in converting English questions to SQL queries!
                However, no data is currently available.
                Please ask the user to upload a CSV file first."""
    
    columns_str = ", ".join(table_info['columns'])
    
    return f"""You are an expert in converting English questions to SQL queries!
    The table '{table_name}' has the following structure:
    - Columns: {columns_str}
    - Row count: {table_info['row_count']}
    
    Convert the user's question into a SQL query. Return ONLY the SQL query without explanations.
    
    Examples:
    - "How many records?" → SELECT COUNT(*) FROM {table_name}
    - "Show first 10 rows" → SELECT * FROM {table_name} LIMIT 10
    - "Count unique values in column X" → SELECT COUNT(DISTINCT X) FROM {table_name}
    - "Average of column Y" → SELECT AVG(Y) FROM {table_name}
    - "Filter where column Z > 100" → SELECT * FROM {table_name} WHERE Z > 100
    - "Group by column A and count" → SELECT A, COUNT(*) FROM {table_name} GROUP BY A
    
    Important: 
    - Use '{table_name}' as the table name
    - Return only executable SQL query
    - Use exact column names: {columns_str}
    - Don't use semicolon at the end
    
    Convert this question: {{user_query}}
    """


@st.cache_data
def get_sql_query_from_text(user_query, table_info, table_name, _api_key):
    """Generate SQL query from natural language - cached for efficiency"""
    if not table_info:
        return "-- No data available. Please upload a CSV file first."
    
    prompt_template = create_dynamic_sql_prompt(table_info, table_name)
    groq_sys_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    llm = ChatGroq(
        groq_api_key=_api_key,
        model_name="llama3-8b-8192"
    )
    
    chain = groq_sys_prompt | llm | StrOutputParser()
    
    try:
        sql_query = chain.invoke({"user_query": user_query})
        return sql_query.strip()
    except Exception as e:
        return f"-- Error generating query: {str(e)}"


def execute_sql_query(query, df, table_name):
    """Execute SQL query safely using a fresh connection"""
    try:
        # Remove comments and clean query
        query = query.strip()
        if query.startswith('--'):
            return None, "Invalid query"
        
        # Create fresh connection for this query
        conn = get_database_connection()
        
        # Recreate table with current data
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
        # Execute the query
        cursor = conn.execute(query)
        
        # Get column names
        columns = [description[0] for description in cursor.description] if cursor.description else []
        
        # Fetch results
        results = cursor.fetchall()
        
        # Close connection
        conn.close()
        
        # Convert to DataFrame for easier handling
        if results and columns:
            df_result = pd.DataFrame(results, columns=columns)
            return df_result, None
        elif not results and columns:
            # Query executed but no results (like COUNT that returns 0)
            return pd.DataFrame(columns=columns), None
        else:
            # Non-SELECT queries (though we shouldn't have them)
            return f"Query executed successfully", None
        
    except Exception as e:
        return None, f"SQL execution error: {str(e)}"


def format_sql_result_for_display(result, user_query):
    """Format the SQL result for better display"""
    if result is None:
        return "No result to display"
    
    # Handle different types of results
    if isinstance(result, str):
        return f"Result: **{result}**"
    elif isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return "No data found matching your criteria"
        elif len(result) == 1 and len(result.columns) == 1:
            # Single value result (like COUNT, AVG, etc.)
            value = result.iloc[0, 0]
            return f"Result: **{value}**"
        elif len(result) == 1:
            return f"Found 1 record:\n{result.to_string(index=False)}"
        else:
            return f"Found {len(result)} records"
    else:
        return f"Result: {str(result)}"


@st.cache_data
def generate_natural_response(user_query, result_summary, _api_key):
    """Generate natural language response - cached for efficiency"""
    response_prompt = ChatPromptTemplate.from_template("""
    You are a helpful data analyst. Based on the user's question and SQL query results, provide a clear, conversational answer.
    
    User Question: {user_query}
    Results: {result_summary}
    
    Provide a natural, conversational response that directly answers the question.
    Keep it concise and helpful.
    """)
    
    llm = ChatGroq(
        groq_api_key=_api_key,
        model_name="llama3-8b-8192"
    )
    
    chain = response_prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "user_query": user_query,
            "result_summary": result_summary
        })
        return response.strip()
    except:
        return result_summary


def main():
    st.title("Talk to Your Data")
    st.markdown("Upload CSV data and ask questions in natural language")
    
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY environment variable not set!")
        st.stop()
    
    # Initialize session state for in-memory storage
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'table_name' not in st.session_state:
        st.session_state.table_name = None
    if 'table_info' not in st.session_state:
        st.session_state.table_info = {}
    
    # Sidebar for data management
    with st.sidebar:
        st.header("Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload a CSV file to analyze with SQL"
        )
        
        if uploaded_file is not None:
            # Read file content
            file_content = uploaded_file.getvalue().decode('utf-8')
            file_name = uploaded_file.name.replace('.csv', '').replace(' ', '_')
            
            # Process file
            df, error = process_uploaded_file(file_content, file_name)
            
            if error:
                st.error(f"Error processing file: {error}")
            else:
                st.success(f"File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Show preview
                with st.expander("Data Preview"):
                    st.dataframe(df.head())
                
                # Table name input
                table_name = st.text_input("Table name:", value=file_name)
                
                if st.button("Load Data", type="primary"):
                    # Store data in session state
                    st.session_state.current_data = df
                    st.session_state.table_name = table_name
                    st.session_state.table_info = get_table_info_from_dataframe(df, table_name)
                    st.success(f"Table '{table_name}' loaded successfully!")
                    st.rerun()
        
        # Current data info
        st.subheader("Current Dataset")
        if st.session_state.current_data is not None and st.session_state.table_info:
            info = st.session_state.table_info
            st.success(f"Table: {st.session_state.table_name}")
            st.info(f"Rows: {info['row_count']} | Columns: {len(info['columns'])}")
            
            with st.expander("Column Information"):
                for col, dtype in zip(info['columns'], info['dtypes']):
                    st.write(f"• **{col}** ({dtype})")
            
            if st.button("Clear Data"):
                st.session_state.current_data = None
                st.session_state.table_name = None
                st.session_state.table_info = {}
                st.success("Data cleared!")
                st.rerun()
        else:
            st.info("No data loaded. Upload a CSV file to get started.")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask Your Question")
        
        if st.session_state.current_data is None:
            st.warning("Upload and load a CSV file first.")
            user_query = st.text_input("Your question:", disabled=True)
            submit = st.button("Ask Question", disabled=True)
        else:
            # Example questions
            st.markdown("**Example questions:**")
            examples = [
                f"How many records are in the table?",
                f"Show me the first 5 rows",
                f"What are the column names?",
                "Show me unique values in each column"
            ]
            
            # Create example buttons
            cols = st.columns(2)
            for i, example in enumerate(examples):
                with cols[i % 2]:
                    if st.button(example, key=f"ex_{i}", use_container_width=True):
                        st.session_state.example_query = example
            
            # Query input
            default_query = st.session_state.get('example_query', '')
            user_query = st.text_input("Your question:", value=default_query)
            submit = st.button("Ask Question", type="primary")
    
    with col2:
        st.subheader("SQL Tools")
        
        if st.session_state.current_data is not None and user_query:
            with st.expander("Generated SQL"):
                sql_query = get_sql_query_from_text(
                    user_query, 
                    st.session_state.table_info, 
                    st.session_state.table_name,
                    api_key
                )
                st.code(sql_query, language="sql")
        
        # Manual SQL execution
        with st.expander("Run Custom SQL"):
            custom_sql = st.text_area("SQL query:", placeholder=f"SELECT * FROM {st.session_state.table_name or 'your_table'} LIMIT 5")
            if st.button("Execute SQL") and custom_sql and st.session_state.current_data is not None:
                result, error = execute_sql_query(custom_sql, st.session_state.current_data, st.session_state.table_name)
                if error:
                    st.error(error)
                else:
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    else:
                        st.write(result)
    
    # Process main query
    if submit and user_query and st.session_state.current_data is not None:
        with st.spinner("Processing your question..."):
            # Generate SQL query
            sql_query = get_sql_query_from_text(
                user_query, 
                st.session_state.table_info, 
                st.session_state.table_name,
                api_key
            )
            
            # Execute SQL query
            result, error = execute_sql_query(sql_query, st.session_state.current_data, st.session_state.table_name)
            
            if error:
                st.error(f"Error: {error}")
                st.write("**Troubleshooting:**")
                st.write("- Check if column names are correct")
                st.write("- Try rephrasing your question")
                st.write("- View column information in the sidebar")
                st.write("- Check the generated SQL query above")
            else:
                # Format and display results
                result_summary = format_sql_result_for_display(result, user_query)
                
                # Generate natural response
                natural_response = generate_natural_response(user_query, result_summary, api_key)
                
                # Display answer
                st.subheader("Answer")
                st.success(natural_response)
                
                # Show detailed results
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    st.subheader("Query Results")
                    
                    if len(result) <= 20:
                        st.dataframe(result, use_container_width=True)
                    else:
                        display_option = st.radio(
                            "Display options:",
                            ["First 20 rows", "Last 20 rows", "Summary stats"],
                            horizontal=True
                        )
                        
                        if display_option == "First 20 rows":
                            st.dataframe(result.head(20), use_container_width=True)
                        elif display_option == "Last 20 rows":
                            st.dataframe(result.tail(20), use_container_width=True)
                        else:
                            numeric_cols = result.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                st.dataframe(result[numeric_cols].describe())
                            else:
                                st.info("No numeric columns for statistics")
                    
                    # Download option
                    if len(result) > 0:
                        csv = result.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "sql_results.csv",
                            "text/csv"
                        )
                
                # Technical details
                with st.expander("Technical Details"):
                    st.code(sql_query, language="sql")
                    if isinstance(result, pd.DataFrame):
                        st.write(f"Result shape: {result.shape}")


if __name__ == '__main__':
    main()