import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import io


def get_table_schema(conn, table_name):
    """Get the schema of a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return columns


def get_all_tables(conn):
    """Get all table names in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def create_table_from_dataframe(df, table_name, conn):
    """Create a table from pandas dataframe"""
    # Clean column names (remove spaces, special characters)
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
    
    # Write dataframe to SQL
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    return df.columns.tolist()


def get_database_info(conn):
    """Get information about all tables and their schemas"""
    tables = get_all_tables(conn)
    db_info = {}
    
    for table in tables:
        schema = get_table_schema(conn, table)
        db_info[table] = {
            'columns': [col[1] for col in schema],  # col[1] is column name
            'types': [col[2] for col in schema]     # col[2] is column type
        }
    
    return db_info


def clear_database_and_reset():
    """Clear database and reset all session state"""
    try:
        if os.path.exists(st.session_state.database_path):
            os.remove(st.session_state.database_path)
        
        # Clear relevant session state
        keys_to_clear = ['example_query', 'last_query', 'db_info_cache']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        return True
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        return False


def delete_table(table_name):
    """Delete a specific table from the database"""
    try:
        with sqlite3.connect(st.session_state.database_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Clear cache
        if 'db_info_cache' in st.session_state:
            del st.session_state['db_info_cache']
        
        return True
    except Exception as e:
        st.error(f"Error deleting table {table_name}: {str(e)}")
        return False


def get_cached_database_info():
    """Get database info with caching to avoid repeated queries"""
    if 'db_info_cache' not in st.session_state:
        try:
            with sqlite3.connect(st.session_state.database_path) as conn:
                st.session_state.db_info_cache = get_database_info(conn)
        except:
            st.session_state.db_info_cache = {}
    
    return st.session_state.db_info_cache


def refresh_database_info():
    """Manually refresh database info"""
    try:
        with sqlite3.connect(st.session_state.database_path) as conn:
            st.session_state.db_info_cache = get_database_info(conn)
        return True
    except:
        st.session_state.db_info_cache = {}
        return False


def create_dynamic_prompt(db_info):
    """Create a dynamic prompt based on available tables"""
    if not db_info:
        return """You are an expert in converting English questions to SQL query!
                However, no tables are currently available in the database.
                Please ask the user to upload a CSV file first."""
    
    # Build table descriptions
    table_descriptions = []
    for table_name, info in db_info.items():
        columns_str = ", ".join(info['columns'])
        table_descriptions.append(f"Table '{table_name}' has columns: {columns_str}")
    
    tables_info = "\n".join(table_descriptions)
    
    return f"""You are an expert in converting English questions to SQL query!
    The SQL database contains the following tables and columns:
    
    {tables_info}
    
    Examples:
    - "How many records are in [table_name]?" → SELECT COUNT(*) FROM [table_name];
    - "Show all data from [table_name]" → SELECT * FROM [table_name];
    - "Find records where [column] equals [value]" → SELECT * FROM [table_name] WHERE [column] = '[value]';
    
    Important: 
    - Do not include ``` or 'sql' in your response
    - Return only valid SQL query
    - Use exact table and column names as provided above
    
    Convert this question to SQL: {{user_query}}
    """


def get_sql_query_from_text(user_query, db_info):
    """Generate SQL query from natural language using dynamic schema"""
    if not db_info:
        return "-- No tables available. Please upload a CSV file first."
    
    prompt_template = create_dynamic_prompt(db_info)
    groq_sys_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    model = "llama3-8b-8192"
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )
    
    chain = groq_sys_prompt | llm | StrOutputParser()
    sql_query = chain.invoke({"user_query": user_query})
    
    return sql_query.strip()


def generate_natural_language_response(user_query, sql_query, results, column_names):
    """Generate natural language response based on query results"""
    if not results:
        return "No data found for your query."
    
    # Create a summary of the results
    result_summary = ""
    if len(results) == 1 and len(results[0]) == 1:
        # Single value result (like COUNT, MAX, MIN, SUM, AVG)
        value = results[0][0]
        result_summary = f"The answer is: **{value}**"
    elif len(results) == 1:
        # Single row result
        row_data = []
        for i, col in enumerate(column_names):
            row_data.append(f"{col}: {results[0][i]}")
        result_summary = f"Here's the result: {', '.join(row_data)}"
    elif len(results) <= 5:
        # Few rows - show all
        result_summary = f"Found {len(results)} records:\n"
        for i, row in enumerate(results, 1):
            row_data = []
            for j, col in enumerate(column_names):
                row_data.append(f"{col}: {row[j]}")
            result_summary += f"{i}. {', '.join(row_data)}\n"
    else:
        # Many rows - show summary
        result_summary = f"Found {len(results)} records. Here are the first few:\n"
        for i, row in enumerate(results[:3], 1):
            row_data = []
            for j, col in enumerate(column_names):
                row_data.append(f"{col}: {row[j]}")
            result_summary += f"{i}. {', '.join(row_data)}\n"
        result_summary += f"... and {len(results) - 3} more records."
    
    # Use LLM to generate a more natural response
    response_prompt = ChatPromptTemplate.from_template("""
    You are a helpful data analyst assistant. Based on the user's question and the query results, 
    provide a clear, conversational answer that directly addresses what the user asked.
    
    User Question: {user_query}
    SQL Query Used: {sql_query}
    Results Summary: {result_summary}
    
    Provide a natural, conversational response that answers the user's question clearly and concisely.
    Start your response by directly answering the question, then provide additional context if helpful.
    
    Examples of good responses:
    - "The highest blood pressure in your dataset is 180 mmHg."
    - "There are 150 patients in your diabetes dataset."
    - "The average age of patients is 45.2 years."
    - "The most common diagnosis is Type 2 Diabetes, appearing in 85% of cases."
    
    Keep it conversational and helpful!
    """)
    
    model = "llama3-8b-8192"
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )
    
    chain = response_prompt | llm | StrOutputParser()
    
    try:
        natural_response = chain.invoke({
            "user_query": user_query,
            "sql_query": sql_query,
            "result_summary": result_summary
        })
        return natural_response.strip()
    except Exception as e:
        # Fallback to simple response if LLM fails
        return result_summary


def get_data_from_database(sql_query, database_path):
    """Execute SQL query and return results"""
    try:
        with sqlite3.connect(database_path) as conn:
            result = conn.execute(sql_query).fetchall()
            # Also get column names
            cursor = conn.cursor()
            cursor.execute(sql_query)
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            return result, column_names
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return [], []


def main():
    st.set_page_config(page_title="Text To SQL", layout="wide")
    st.title("Talk to Your Database")
    st.markdown("Upload your CSV data and ask questions in natural language!")
    
    # Initialize session state
    if 'database_path' not in st.session_state:
        st.session_state.database_path = "temp_database.db"
    
    # Sidebar for file upload and database management
    with st.sidebar:
        st.header("Data Management")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload a CSV file to create a new table in the database"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Get table name
                default_name = uploaded_file.name.replace('.csv', '').replace(' ', '_')
                table_name = st.text_input(
                    "Table name:", 
                    value=default_name,
                    help="Enter a name for your table (no spaces or special characters)"
                )
                
                if st.button("Create Table", type="primary"):
                    if table_name:
                        # Clean table name
                        table_name = table_name.replace(' ', '_').replace('[^A-Za-z0-9_]', '')
                        
                        # Create table
                        with sqlite3.connect(st.session_state.database_path) as conn:
                            columns = create_table_from_dataframe(df, table_name, conn)
                            
                        # Clear cache to refresh database info
                        if 'db_info_cache' in st.session_state:
                            del st.session_state['db_info_cache']
                            
                        st.success(f"Table '{table_name}' created successfully!")
                        st.info(f"Columns: {', '.join(columns)}")
                        st.rerun()
                    else:
                        st.error("Please enter a table name")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        st.divider()
        
        # Database management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Data", help="Remove all tables and data"):
                if clear_database_and_reset():
                    st.success("Database cleared!")
                    st.rerun()
        
        with col2:
            if st.button("Refresh DB", help="Refresh database information"):
                if refresh_database_info():
                    st.success("Refreshed!")
                    st.rerun()
                else:
                    st.error("Error refreshing database")
        
        st.divider()
        
        # Show current database info
        st.subheader("Current Database")
        db_info = get_cached_database_info()
        
        if db_info:
            st.success(f"{len(db_info)} table(s) loaded")
            
            for table_name, info in db_info.items():
                with st.expander(f"Table: {table_name}"):
                    st.write("**Columns:**")
                    for col, dtype in zip(info['columns'], info['types']):
                        st.write(f"• {col} ({dtype})")
                    
                    # Individual table delete button
                    if st.button(f"Delete {table_name}", key=f"delete_{table_name}"):
                        if delete_table(table_name):
                            st.success(f"Table '{table_name}' deleted!")
                            st.rerun()
        else:
            st.info("No tables found. Upload a CSV file to get started!")
    
    # Main query interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask Your Question")
        
        # Get current database info
        db_info = get_cached_database_info()
        
        if not db_info:
            st.warning("No data available. Please upload a CSV file using the sidebar.")
            user_query = st.text_input("Your question:", disabled=True, placeholder="Upload data first...")
            submit = st.button("Ask Question", disabled=True)
        else:
            # Show active tables
            table_names = list(db_info.keys())
            st.info(f"Active tables: {', '.join(table_names)}")
            
            # Example questions based on available tables
            st.markdown("**💡 Example questions you can ask:**")
            example_table = table_names[0]  # Get first table name
            examples = [
                f"How many records are in {example_table}?",
                f"Show me all data from {example_table}",
                f"What are the unique values in the first column?",
                "Show me the first 10 rows"
            ]
            
            # Create columns for example buttons
            example_cols = st.columns(2)
            for i, example in enumerate(examples):
                with example_cols[i % 2]:
                    if st.button(f"{example}", key=f"example_{i}", use_container_width=True):
                        st.session_state.example_query = example
            
            # Query input
            default_query = st.session_state.get('example_query', '')
            user_query = st.text_input(
                "Your question:", 
                value=default_query
            )
            submit = st.button("Ask Question", type="primary")
    
    with col2:
        st.subheader("Query Tools")
        
        # Show generated SQL
        if user_query and db_info:
            with st.expander("Generated SQL Query"):
                sql_query = get_sql_query_from_text(user_query, db_info)
                st.code(sql_query, language="sql")
        
        # Manual SQL input
        with st.expander("Advanced: Run Custom SQL"):
            st.markdown("*For advanced users*")
            custom_sql = st.text_area("Enter SQL query:", placeholder="SELECT * FROM your_table LIMIT 10;")
            if st.button("Execute SQL") and custom_sql:
                try:
                    results, columns = get_data_from_database(custom_sql, st.session_state.database_path)
                    if results:
                        df_result = pd.DataFrame(results, columns=columns)
                        st.success(f"Found {len(results)} records")
                        st.dataframe(df_result)
                    else:
                        st.info("Query executed successfully (no results returned)")
                except Exception as e:
                    st.error(f"SQL Error: {str(e)}")
    
    # Process the main query
    if submit and user_query and db_info:
        with st.spinner("Analyzing your question..."):
            try:
                # Generate SQL query
                sql_query = get_sql_query_from_text(user_query, db_info)
                
                # Execute query
                results, columns = get_data_from_database(sql_query, st.session_state.database_path)
                
                # Generate natural language response
                if results:
                    natural_response = generate_natural_language_response(user_query, sql_query, results, columns)
                else:
                    natural_response = "I couldn't find any data matching your question."
                
                # Display results
                st.subheader("Answer")
                
                # Main answer in a highlighted box
                st.success(f"**{natural_response}**")
                
                # Technical details in expandable section
                with st.expander("Technical Details", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(sql_query, language="sql")
                    with col2:
                        st.metric("Records Found", len(results))
                
                # Data display section
                if results:
                    st.subheader("Data Details")
                    
                    # Convert to DataFrame for better display
                    df_result = pd.DataFrame(results, columns=columns)
                    
                    # Smart display based on result size
                    if len(results) == 1 and len(columns) == 1:
                        # Single value - already shown in natural response
                        st.info("ℹSingle value result shown above")
                    elif len(results) <= 10:
                        # Small result set - show as table
                        st.dataframe(df_result, use_container_width=True)
                    else:
                        # Large result set - show with options
                        display_option = st.radio(
                            "Choose display format:",
                            ["First 10 rows", "All data", "Summary statistics"],
                            horizontal=True
                        )
                        
                        if display_option == "First 10 rows":
                            st.dataframe(df_result.head(10), use_container_width=True)
                            if len(results) > 10:
                                st.info(f"Showing first 10 of {len(results)} records")
                        elif display_option == "All data":
                            st.dataframe(df_result, use_container_width=True)
                        else:
                            # Show summary statistics for numeric columns
                            numeric_cols = df_result.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                st.write("**Summary Statistics:**")
                                st.dataframe(df_result[numeric_cols].describe())
                            else:
                                st.info("No numeric columns found for statistics")
                    
                    # Download option
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data found matching your criteria.")
                    st.write("**Suggestions:**")
                    st.write("- Check if the column names are spelled correctly")
                    st.write("- Try a broader search term")
                    st.write("- View your table structure in the sidebar")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.write("**Troubleshooting tips:**")
                st.write("- Make sure your question relates to the uploaded data")
                st.write("- Check column names in the sidebar")
                st.write("- Try rephrasing your question")
                
                # Show available tables for debugging
                if db_info:
                    st.write("**Available tables and columns:**")
                    for table_name, info in db_info.items():
                        st.write(f"- **{table_name}**: {', '.join(info['columns'])}")


if __name__ == '__main__':
    # Ensure GROQ_API_KEY is set
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY environment variable not set!")
        st.info("Please set your GROQ API key in the environment variables.")
        st.stop()
    
    main()