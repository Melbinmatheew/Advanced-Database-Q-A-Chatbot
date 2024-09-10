# import streamlit as st
# import pandas as pd
# import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
# import re

# # Load the GPT-2 model and tokenizer
# @st.cache_resource
# def load_model():
#     model_name = "gpt2"
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     return tokenizer, model

# tokenizer, model = load_model()

# def generate_response(prompt, max_new_tokens=100):
#     inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
#     with torch.no_grad():
#         outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     response = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
#     return response.strip()

# def read_file(file, file_type):
#     if file_type == 'csv':
#         df = pd.read_csv(file)
#     elif file_type == 'excel':
#         df = pd.read_excel(file)
#     elif file_type == 'sql':
#         # Read the SQL file content
#         sql_content = file.read().decode('utf-8')
        
#         # Split the content into individual SQL statements
#         statements = sql_content.split(';')
        
#         # Find CREATE TABLE statements
#         create_statements = [stmt for stmt in statements if 'CREATE TABLE' in stmt.upper()]
        
#         # Extract table names and column definitions
#         tables = {}
#         for stmt in create_statements:
#             match = re.search(r'CREATE TABLE `(\w+)`\s*\((.*?)\)', stmt, re.DOTALL)
#             if match:
#                 table_name = match.group(1)
#                 columns = [col.strip().split()[0].replace('`', '') for col in match.group(2).split(',') if 'CONSTRAINT' not in col]
#                 tables[table_name] = columns
        
#         # Create a DataFrame with table information
#         df = pd.DataFrame([(table, ', '.join(columns)) for table, columns in tables.items()], 
#                           columns=['Table Name', 'Columns'])
#     else:
#         raise ValueError(f"Unsupported file type: {file_type}")
    
#     # Convert column names to lowercase
#     df.columns = df.columns.str.lower()
#     return df

# def summarize_dataframe(df):
#     summary = []
#     summary.append(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")
#     summary.append("Columns: " + ", ".join(df.columns))
#     for column in df.columns:
#         if pd.api.types.is_numeric_dtype(df[column]):
#             summary.append(f"{column}: Min={df[column].min()}, Max={df[column].max()}, Mean={df[column].mean():.2f}")
#         elif pd.api.types.is_string_dtype(df[column]):
#             unique_values = df[column].nunique()
#             summary.append(f"{column}: {unique_values} unique values")
#     return "\n".join(summary)

# def question_to_python_query(df, question):
#     question = question.lower()
    
#     # Helper function to find column names in the question
#     def find_column(q):
#         return next((col for col in df.columns if col in q), None)

#     if "how many" in question:
#         return "len(df)"
    
#     elif "average" in question or "mean" in question:
#         col = find_column(question)
#         return f"df['{col}'].mean()" if col else None
    
#     elif "minimum" in question or "min" in question:
#         col = find_column(question)
#         return f"df['{col}'].min()" if col else None
    
#     elif "maximum" in question or "max" in question:
#         col = find_column(question)
#         return f"df['{col}'].max()" if col else None
    
#     elif "sum" in question:
#         col = find_column(question)
#         return f"df['{col}'].sum()" if col else None
    
#     elif "unique" in question:
#         col = find_column(question)
#         return f"df['{col}'].nunique()" if col else None
    
#     elif "describe" in question:
#         return "df.describe()"
    
#     elif "top" in question or "highest" in question:
#         col = find_column(question)
#         match = re.search(r'top (\d+)', question)
#         n = int(match.group(1)) if match else 5
#         return f"df.nlargest({n}, '{col}')['{col}'].tolist()" if col else None
    
#     elif "bottom" in question or "lowest" in question:
#         col = find_column(question)
#         match = re.search(r'bottom (\d+)', question)
#         n = int(match.group(1)) if match else 5
#         return f"df.nsmallest({n}, '{col}')['{col}'].tolist()" if col else None
    
#     elif "correlation" in question:
#         cols = [col for col in df.columns if col in question]
#         if len(cols) == 2:
#             return f"df['{cols[0]}'].corr(df['{cols[1]}'])"
    
#     elif "group by" in question:
#         group_col = find_column(question.split("group by")[1])
#         agg_col = find_column(question.split("group by")[0])
#         if group_col and agg_col:
#             return f"df.groupby('{group_col}')['{agg_col}'].mean().to_dict()"
    
#     return None

# def execute_python_query(df, query):
#     try:
#         result = eval(query)
#         return result
#     except Exception as e:
#         return f"Error executing query: {str(e)}"

# def main():
#     st.title("Advanced Database Q&A Chatbot")

#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     uploaded_file = st.file_uploader("Upload your database file", type=["csv", "xlsx", "sql"])

#     if uploaded_file is not None:
#         try:
#             file_type = uploaded_file.name.split('.')[-1].lower()
#             if file_type == 'xlsx':
#                 file_type = 'excel'
#             df = read_file(uploaded_file, file_type)
            
#             st.write("Database Preview:")
#             st.dataframe(df.head())

#             df_summary = summarize_dataframe(df)

#             st.subheader("Chat with your database")

#             for message in st.session_state.chat_history:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])

#             user_input = st.chat_input("Ask a question about your database:")

#             if user_input:
#                 st.session_state.chat_history.append({"role": "user", "content": user_input})
                
#                 with st.chat_message("user"):
#                     st.markdown(user_input)

#                 python_query = question_to_python_query(df, user_input)
                
#                 if python_query:
#                     result = execute_python_query(df, python_query)
#                     if isinstance(result, str) and result.startswith("Error"):
#                         response = "I'm sorry, I couldn't find the answer to your question."
#                     else:
#                         response = f"The answer to your question is: {result}"
#                 else:
#                     response = "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

#                 with st.chat_message("assistant"):
#                     st.markdown(response)

#                 st.session_state.chat_history.append({"role": "assistant", "content": response})

#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#             st.write("Please make sure you've uploaded a valid database file.")

#     else:
#         st.write("Please upload a CSV, Excel, or SQL file to start.")

#     st.sidebar.header("Supported File Types")
#     st.sidebar.write("- CSV (.csv)")
#     st.sidebar.write("- Excel (.xlsx)")
#     st.sidebar.write("- SQL (.sql)")
#     st.sidebar.write("Note: For SQL files, please ensure they contain valid SQL queries.")

# if __name__ == "__main__":
#     main()








# import streamlit as st
# import pandas as pd
# import numpy as np
# import torch
# from transformers import BertTokenizer, BertForQuestionAnswering
# import re
# import sqlite3
# from sqlalchemy import create_engine, text

# # Load the BERT model and tokenizer
# @st.cache_resource
# def load_model():
#     model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForQuestionAnswering.from_pretrained(model_name)
#     return tokenizer, model

# tokenizer, model = load_model()

# def generate_sql_query(question, table_name, columns):
#     question = question.lower()
    
#     if "how many" in question:
#         return f"SELECT COUNT(*) FROM {table_name}"
#     elif "average" in question or "mean" in question:
#         col = next((col for col in columns if col.lower() in question), None)
#         return f"SELECT AVG({col}) FROM {table_name}" if col else None
#     elif "maximum" in question or "highest" in question:
#         col = next((col for col in columns if col.lower() in question), None)
#         return f"SELECT MAX({col}) FROM {table_name}" if col else None
#     elif "minimum" in question or "lowest" in question:
#         col = next((col for col in columns if col.lower() in question), None)
#         return f"SELECT MIN({col}) FROM {table_name}" if col else None
    
#     return f"SELECT * FROM {table_name} LIMIT 5"

# def execute_sql_query(conn, query):
#     try:
#         df = pd.read_sql_query(query, conn)
#         return df
#     except Exception as e:
#         return f"Error executing query: {str(e)}"

# def read_file(file, file_type):
#     if file_type in ['csv', 'excel']:
#         if file_type == 'csv':
#             df = pd.read_csv(file)
#         else:  # excel
#             df = pd.read_excel(file)
#         return {'type': 'dataframe', 'data': df}
#     elif file_type == 'sql':
#         conn = sqlite3.connect(':memory:')
#         sql_script = file.read().decode('utf-8')
#         statements = sql_script.split(';')
#         cursor = conn.cursor()
#         for statement in statements:
#             try:
#                 cursor.execute(statement)
#             except sqlite3.Error as e:
#                 st.warning(f"Error executing SQL statement: {e}")
#                 st.write("Skipping this statement and continuing with the rest.")
#                 continue
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         tables = [table[0] for table in cursor.fetchall()]
#         if not tables:
#             st.error("No valid tables were created from the SQL file. Please check the file content.")
#             return None
#         return {'type': 'sql', 'connection': conn, 'tables': tables}
#     else:
#         raise ValueError(f"Unsupported file type: {file_type}")

# def get_table_preview(conn, table_name):
#     query = f"SELECT * FROM {table_name} LIMIT 5"
#     return pd.read_sql_query(query, conn)

# def answer_question(question, context):
#     inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model(**inputs)
    
#     answer_start = torch.argmax(outputs.start_logits)
#     answer_end = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
#     # Clean up the answer
#     answer = answer.strip()
#     answer = re.sub(r'^\[CLS\]|\[SEP\]$', '', answer).strip()
#     answer = re.sub(r'\s+', ' ', answer)
    
#     return answer if answer else "I couldn't find a specific answer to your question in the data."

# def main():
#     st.title("Advanced Database Q&A Chatbot")

#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     uploaded_file = st.file_uploader("Upload your database file", type=["csv", "xlsx", "sql"])

#     if uploaded_file is not None:
#         try:
#             file_type = uploaded_file.name.split('.')[-1].lower()
#             if file_type == 'xlsx':
#                 file_type = 'excel'
            
#             file_content = read_file(uploaded_file, file_type)
            
#             if file_content is None:
#                 st.error("Failed to process the uploaded file. Please check the file content and try again.")
#                 return
            
#             if file_content['type'] == 'dataframe':
#                 df = file_content['data']
#                 st.write("Data Preview:")
#                 st.dataframe(df.head())
                
#                 st.subheader("Chat with your data")
                
#                 for message in st.session_state.chat_history:
#                     with st.chat_message(message["role"]):
#                         st.markdown(message["content"])
                
#                 user_input = st.chat_input("Ask a question about your data:")
                
#                 if user_input:
#                     st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
#                     with st.chat_message("user"):
#                         st.markdown(user_input)
                    
#                     query = generate_sql_query(user_input, "data", df.columns)
                    
#                     if query:
#                         result = df.query(query.split("WHERE")[-1]) if "WHERE" in query else df
                        
#                         if not result.empty:
#                             context = result.to_string()
#                             answer = answer_question(user_input, context)
#                             response = answer
#                         else:
#                             response = "I couldn't find any data matching your question."
#                     else:
#                         response = "I'm sorry, I couldn't understand your question. Could you please rephrase it?"
                    
#                     with st.chat_message("assistant"):
#                         st.markdown(response)
                    
#                     st.session_state.chat_history.append({"role": "assistant", "content": response})
            
#             elif file_content['type'] == 'sql':
#                 conn = file_content['connection']
#                 tables = file_content['tables']
                
#                 st.write("Available tables:")
#                 for table in tables:
#                     st.write(f"- {table}")
                
#                 selected_table = st.selectbox("Select a table to view:", tables)
                
#                 if selected_table:
#                     st.write(f"Preview of {selected_table}:")
#                     preview_df = get_table_preview(conn, selected_table)
#                     st.dataframe(preview_df)
                
#                 st.subheader("Chat with your database")
                
#                 for message in st.session_state.chat_history:
#                     with st.chat_message(message["role"]):
#                         st.markdown(message["content"])
                
#                 user_input = st.chat_input("Ask a question about your database:")
                
#                 if user_input:
#                     st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
#                     with st.chat_message("user"):
#                         st.markdown(user_input)
                    
#                     sql_query = generate_sql_query(user_input, selected_table, preview_df.columns)
                    
#                     if sql_query:
#                         result = execute_sql_query(conn, sql_query)
#                         if isinstance(result, pd.DataFrame):
#                             if not result.empty:
#                                 context = result.to_string()
#                                 answer = answer_question(user_input, context)
#                                 response = answer
#                             else:
#                                 response = "I couldn't find any data matching your question."
#                         else:
#                             response = f"Error: {result}"
#                     else:
#                         response = "I'm sorry, I couldn't generate a SQL query for your question. Could you please rephrase it?"
                    
#                     with st.chat_message("assistant"):
#                         st.markdown(response)
                    
#                     st.session_state.chat_history.append({"role": "assistant", "content": response})

#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#             st.write("Please make sure you've uploaded a valid database file.")

#     else:
#         st.write("Please upload a CSV, Excel, or SQL file to start.")

#     st.sidebar.header("Supported File Types")
#     st.sidebar.write("- CSV (.csv)")
#     st.sidebar.write("- Excel (.xlsx)")
#     st.sidebar.write("- SQL (.sql)")
#     st.sidebar.write("Note: For SQL files, please ensure they contain valid SQLite statements.")

# if __name__ == "__main__":
#     main()

####-----------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
# from sentence_transformers import SentenceTransformer, util
# import torch
# import re
# from sqlalchemy import create_engine, text

# # Caching resources for efficiency
# @st.cache_resource
# def load_models():
#     # Load BERT for Question Answering
#     qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
#     qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
#     qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
#     qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
    
#     # Load Sentence Transformer for semantic search
#     sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

#     return qa_pipeline, sentence_transformer

# qa_pipeline, sentence_transformer = load_models()



# def read_file(file, file_type):
#     """
#     Read the uploaded file based on its type and return a pandas DataFrame.
#     """
#     if file_type == 'csv':
#         df = pd.read_csv(file)
#     elif file_type == 'excel':
#         df = pd.read_excel(file)
#     elif file_type == 'sql':
#         # Create an SQLite in-memory database
#         engine = create_engine('sqlite:///:memory:')
#         conn = engine.connect()

#         # Read the SQL file and execute its commands
#         sql_commands = file.read().decode('utf-8')
#         for command in sql_commands.split(';'):
#             if command.strip():
#                 try:
#                     conn.execute(text(command))
#                 except Exception as e:
#                     st.error(f"Error executing SQL command: {e}")
        
#         # Now read the data from the in-memory database into a DataFrame
#         df = pd.read_sql('SELECT * FROM <table_name>', conn)  # Replace <table_name> with the actual table name

#         conn.close()
#     else:
#         raise ValueError(f"Unsupported file type: {file_type}")
    
#     # Convert column names to lowercase
#     df.columns = df.columns.str.lower()
#     return df


# def summarize_dataframe(df):
#     """
#     Summarize the DataFrame by providing basic statistics and column information.
#     """
#     summary = []
#     summary.append(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")
#     summary.append("Columns: " + ", ".join(df.columns))
#     for column in df.columns:
#         if pd.api.types.is_numeric_dtype(df[column]):
#             summary.append(f"{column}: Min={df[column].min()}, Max={df[column].max()}, Mean={df[column].mean():.2f}")
#         elif pd.api.types.is_string_dtype(df[column]):
#             unique_values = df[column].nunique()
#             summary.append(f"{column}: {unique_values} unique values")
#     return "\n".join(summary)

# def embed_question(question):
#     """
#     Embed a question using a Sentence Transformer model.
#     """
#     return sentence_transformer.encode(question, convert_to_tensor=True)

# def semantic_search(embedded_question, df, top_k=3):
#     """
#     Perform semantic search to find the most relevant columns based on the embedded question.
#     """
#     column_embeddings = {col: sentence_transformer.encode(col, convert_to_tensor=True) for col in df.columns}
#     similarities = {col: util.pytorch_cos_sim(embedded_question, embedding).item() for col, embedding in column_embeddings.items()}
#     sorted_cols = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
#     return sorted_cols[:top_k]

# def generate_answer(question, context):
#     """
#     Generate an answer using BERT's question-answering model.
#     """
#     result = qa_pipeline(question=question, context=context)
#     return result['answer']

# def question_to_context(df, question):
#     """
#     Convert user question to a relevant context (text) using semantic understanding.
#     """
#     embedded_question = embed_question(question)
#     relevant_columns = semantic_search(embedded_question, df)
    
#     if not relevant_columns:
#         return None
    
#     # Select the most relevant column and get its context (text data)
#     most_relevant_col = relevant_columns[0][0]
#     context = df[most_relevant_col].astype(str).str.cat(sep=' ')
    
#     return context

# def question_to_python_query(df, question):
#     """
#     Convert user question to a Python query using semantic understanding and keyword matching.
#     """
#     embedded_question = embed_question(question)
#     relevant_columns = semantic_search(embedded_question, df)
    
#     if not relevant_columns:
#         return None
    
#     col = relevant_columns[0][0]  # Taking the most relevant column
#     if "how many" in question:
#         return "len(df)"
#     elif "average" in question or "mean" in question:
#         return f"df['{col}'].mean()" if col else None
#     elif "minimum" in question or "min" in question:
#         return f"df['{col}'].min()" if col else None
#     elif "maximum" in question or "max" in question:
#         return f"df['{col}'].max()" if col else None
#     elif "sum" in question:
#         return f"df['{col}'].sum()" if col else None
#     elif "unique" in question:
#         return f"df['{col}'].unique().tolist()" if col else None
#     elif "describe" in question:
#         return "df.describe()"
#     elif "top" in question or "highest" in question:
#         match = re.search(r'top (\d+)', question)
#         n = int(match.group(1)) if match else 5
#         return f"df.nlargest({n}, '{col}')['{col}'].tolist()" if col else None
#     elif "bottom" in question or "lowest" in question:
#         match = re.search(r'bottom (\d+)', question)
#         n = int(match.group(1)) if match else 5
#         return f"df.nsmallest({n}, '{col}')['{col}'].tolist()" if col else None
#     elif "correlation" in question:
#         cols = [col for col in df.columns if col in question]
#         if len(cols) == 2:
#             return f"df['{cols[0]}'].corr(df['{cols[1]}'])"
#     elif "group by" in question:
#         group_col = relevant_columns[1][0] if len(relevant_columns) > 1 else None
#         agg_col = relevant_columns[0][0]
#         if group_col and agg_col:
#             return f"df.groupby('{group_col}')['{agg_col}'].mean().to_dict()"
#     elif "columns" in question:
#         return "df.columns.tolist()"
#     elif "data types" in question or "dtype" in question:
#         return "df.dtypes"
    
#     return None

# def execute_python_query(df, query):
#     """
#     Safely execute a dynamically generated Python query.
#     """
#     try:
#         result = eval(query)
#         return result
#     except Exception as e:
#         return f"Error executing query: {str(e)}"

# def main():
#     st.title("Advanced Database Q&A Chatbot Using BERT")

#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     uploaded_file = st.file_uploader("Upload your database file", type=["csv", "xlsx", "sql"])

#     if uploaded_file is not None:
#         try:
#             file_type = uploaded_file.name.split('.')[-1].lower()
#             if file_type == 'xlsx':
#                 file_type = 'excel'
#             df = read_file(uploaded_file, file_type)
            
#             st.write("Database Preview:")
#             st.dataframe(df.head())

#             df_summary = summarize_dataframe(df)

#             st.subheader("Chat with your database")

#             for message in st.session_state.chat_history:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])

#             user_input = st.chat_input("Ask a question about your database:")

#             if user_input:
#                 st.session_state.chat_history.append({"role": "user", "content": user_input})
                
#                 with st.chat_message("user"):
#                     st.markdown(user_input)

#                 # Use BERT for extractive question answering
#                 context = question_to_context(df, user_input)
#                 if context:
#                     answer = generate_answer(user_input, context)
#                     response = f"The answer to your question is: {answer}"
#                 else:
#                     response = "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

#                 with st.chat_message("assistant"):
#                     st.markdown(response)

#                 st.session_state.chat_history.append({"role": "assistant", "content": response})

#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#             st.stop()

# if __name__ == "__main__":
#     main()


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from sqlalchemy import create_engine, text

# Caching resources for efficiency
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_transformer = load_model()

def read_file(file, file_type):
    if file_type == 'csv':
        return pd.read_csv(file)
    elif file_type == 'excel':
        return pd.read_excel(file)
    elif file_type == 'sql':
        engine = create_engine('sqlite:///:memory:')
        conn = engine.connect()
        sql_commands = file.read().decode('utf-8')
        for command in sql_commands.split(';'):
            if command.strip():
                try:
                    conn.execute(text(command))
                except Exception as e:
                    st.error(f"Error executing SQL command: {e}")
        df = pd.read_sql('SELECT * FROM <table_name>', conn)  # Replace <table_name> with the actual table name
        conn.close()
        return df
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def summarize_dataframe(df):
    summary = [f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns.", 
               "Columns: " + ", ".join(df.columns)]
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            summary.append(f"{column}: Min={df[column].min()}, Max={df[column].max()}, Mean={df[column].mean():.2f}")
        elif pd.api.types.is_string_dtype(df[column]):
            summary.append(f"{column}: {df[column].nunique()} unique values")
    return "\n".join(summary)

def semantic_search(question, df, top_k=3):
    question_embedding = sentence_transformer.encode(question, convert_to_tensor=True)
    column_embeddings = {col: sentence_transformer.encode(col, convert_to_tensor=True) for col in df.columns}
    similarities = {col: util.pytorch_cos_sim(question_embedding, embedding).item() for col, embedding in column_embeddings.items()}
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

def question_to_query(df, question):
    relevant_columns = semantic_search(question, df)
    if not relevant_columns:
        return None
    
    col = relevant_columns[0][0]
    question_lower = question.lower()
    
    if "how many" in question_lower:
        return f"df['{col}'].count()"
    elif any(word in question_lower for word in ["average", "mean"]):
        return f"df['{col}'].mean()"
    elif any(word in question_lower for word in ["minimum", "min"]):
        return f"df['{col}'].min()"
    elif any(word in question_lower for word in ["maximum", "max"]):
        return f"df['{col}'].max()"
    elif "sum" in question_lower:
        return f"df['{col}'].sum()"
    elif "unique" in question_lower:
        return f"df['{col}'].nunique()"
    elif "describe" in question_lower:
        return f"df['{col}'].describe().to_dict()"
    elif any(word in question_lower for word in ["top", "highest"]):
        match = re.search(r'top (\d+)', question_lower)
        n = int(match.group(1)) if match else 5
        return f"df.nlargest({n}, '{col}')['{col}'].tolist()"
    elif any(word in question_lower for word in ["bottom", "lowest"]):
        match = re.search(r'bottom (\d+)', question_lower)
        n = int(match.group(1)) if match else 5
        return f"df.nsmallest({n}, '{col}')['{col}'].tolist()"
    elif "correlation" in question_lower:
        cols = [col for col, _ in relevant_columns[:2]]
        if len(cols) == 2:
            return f"df['{cols[0]}'].corr(df['{cols[1]}'])"
    elif "group by" in question_lower:
        group_col = relevant_columns[1][0] if len(relevant_columns) > 1 else None
        agg_col = relevant_columns[0][0]
        if group_col and agg_col:
            return f"df.groupby('{group_col}')['{agg_col}'].mean().to_dict()"
    elif "columns" in question_lower:
        return "df.columns.tolist()"
    elif any(phrase in question_lower for phrase in ["data types", "dtype"]):
        return "df.dtypes.to_dict()"
    
    return None

def execute_query(df, query):
    try:
        result = eval(query)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

def format_answer(result):
    if isinstance(result, (int, float)):
        return f"The result is {result:.2f}"
    elif isinstance(result, list):
        return f"The result is: {', '.join(map(str, result))}"
    elif isinstance(result, dict):
        return "The result is:\n" + "\n".join([f"{k}: {v}" for k, v in result.items()])
    else:
        return f"The result is: {result}"

def main():
    st.title("Advanced Database Q&A Chatbot")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload your database file", type=["csv", "xlsx", "sql"])

    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'xlsx':
                file_type = 'excel'
            df = read_file(uploaded_file, file_type)
            
            st.write("Database Preview:")
            st.dataframe(df.head())

            df_summary = summarize_dataframe(df)
            st.write("Database Summary:")
            st.write(df_summary)

            st.subheader("Chat with your database")

            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_input = st.chat_input("Ask a question about your database:")

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.markdown(user_input)

                query = question_to_query(df, user_input)
                if query:
                    result = execute_query(df, query)
                    response = format_answer(result)
                else:
                    response = "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

                with st.chat_message("assistant"):
                    st.markdown(response)

                st.session_state.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()