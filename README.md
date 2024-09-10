# Advanced Database Q&A Chatbot

This project implements an advanced chatbot that can answer questions about uploaded database files using natural language processing and data analysis techniques.

## Features

- Supports CSV, Excel, and SQL file uploads
- Provides a summary of the uploaded database
- Uses semantic search to understand user questions
- Converts natural language questions into data queries
- Presents answers in a user-friendly format

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/advanced-database-qa-chatbot.git
   cd advanced-database-qa-chatbot
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Upload your database file (CSV, Excel, or SQL).

4. Ask questions about your data in natural language.

## How it Works

1. **File Upload**: The app accepts CSV, Excel, or SQL files and loads them into a pandas DataFrame.

2. **Data Summary**: It provides a summary of the uploaded data, including the number of rows, columns, and basic statistics.

3. **Question Processing**: When a user asks a question, the app uses a Sentence Transformer model to perform semantic search and identify the most relevant columns.

4. **Query Generation**: Based on the question and relevant columns, the app generates a Python query to extract the required information from the DataFrame.

5. **Query Execution**: The generated query is safely executed on the DataFrame.

6. **Answer Formatting**: The result is formatted into a human-readable answer and presented to the user.

## Limitations

- The chatbot's ability to understand questions is based on keyword matching and semantic similarity. Complex or ambiguous questions may not be interpreted correctly.
- The current implementation supports a limited set of query types. More complex analytical questions may not be supported.

## Contributing

Contributions to improve the chatbot's capabilities are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
