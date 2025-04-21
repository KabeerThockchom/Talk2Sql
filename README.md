# Talk2SQL

A powerful natural language to SQL query tool with visualization capabilities and voice assistant features.

## Overview

Talk2SQL lets you query databases using plain English. It translates natural language questions into SQL, executes the queries, and displays the results with visualizations. The app includes a voice assistant for hands-free operation and maintains a history of all queries for easy reference.

## Demo

https://github.com/user-attachments/assets/d140eff1-50b6-4d06-8f5a-2a835bd16be0

## Setup

### Prerequisites

- Python 3.8+
- Node.js and npm
- SQLite database(s) to query

### Backend Setup

1. **Clone the repository**

```bash
git clone https://github.com/KabeerThockchom/Talk2Sql
cd Talk2SQL
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

Key Python dependencies include:
- flask - Web framework
- pandas & numpy - Data processing
- plotly - Data visualization
- sounddevice & soundfile - Audio processing
- openai - Azure OpenAI integration
- groq - text to speech and speech to text capabilities
- python-dotenv - Environment variable management

4. **Create a .env file**

Create a `.env` file in the root directory with the following variables:

```
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT=gpt-4o-mini

#Optional Anthropic Usage
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Qdrant vector database (for persistent storage)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Groq API (for voice features)
GROQ_API_KEY=your_groq_api_key
```

### Frontend Setup

1. **Navigate to the frontend directory**

```bash
cd frontend
```

2. **Install npm dependencies**

```bash
npm install
```

Key frontend dependencies include:
- React & React DOM - Frontend framework
- Plotly.js - Interactive visualizations
- Axios - HTTP requests
- Tailwind CSS - Styling
- TypeScript - Type safety

3. **Build the frontend**

```bash
npm run build
```

4. **Move the build files to connect with Flask**

The build process creates a `dist` directory. The Flask app is configured to serve these files automatically, so no manual copying is needed.

## Running the Application

1. **Start the backend server**

From the project root:

```bash
python app.py
```

2. **Access the application**

Open your browser and go to:
```
https://text2sql.fly.dev
```

## Features and Dashboard Pages

### Main Dashboard / Query Interface
- Type natural language questions in the main input
- The interface displays:
  - Generated SQL query with syntax highlighting
  - Data results in a tabular format
  - Automatic visualizations based on the data
  - Summary of the query results
  - Thumbs up/down feedback buttons

### Database Connection Page
- Upload or select SQLite databases from the available list
- View schema information automatically extracted from the database
- Upload training examples in JSON format to improve query accuracy
- Monitor vector database status (if using persistent storage)

### Voice Assistant Interface
- Click the microphone icon to activate voice input
- Choose from multiple voice options for responses
- View real-time transcription and results streaming
- Get spoken answers to your questions

### Query History Page
- Browse all past queries with timestamps
- View the original questions, SQL queries, and results
- Re-run previous queries with a single click
- Export history in various formats (JSON, CSV, ZIP)
- Filter by success/failure status

### Metrics & Analytics Page
- View performance dashboards with:
  - Success rate charts and trend analysis
  - Average query times and performance metrics
  - Error type distribution and analysis
  - Query complexity statistics
  - Interactive time-series charts of usage patterns

## Detailed Usage

### Connecting to a Database

1. On the main dashboard, select a database from the list (to have them there put .sqlite files in databases folder) or upload a new SQLite file.
2. The system will automatically extract schema information and prepare the database for queries.
3. You can also upload training examples to improve query accuracy.

### Asking Questions

1. Type your question in natural language (e.g., "How many teams are in the NBA?")
2. View the generated SQL, data results, and visualizations
3. Provide feedback (thumbs up/down) to improve future queries saving queries to qdrant

### Using Voice Features

1. Click the microphone icon to activate voice input
2. Speak your question clearly
3. The system will transcribe your question, execute the query, and provide a spoken response
4. The voice streaming interface will show results in real-time

### Viewing History and Metrics

1. Navigate to the History tab to see past queries
2. Use the Metrics page to analyze performance and patterns
3. Export query history for further analysis
4. View detailed statistics about your query patterns

## Troubleshooting

- **Database connection issues**: Ensure your SQLite file is valid and not corrupted
- **API key errors**: Verify all environment variables are correctly set in the .env file
- **Voice features not working**: Check your microphone permissions and Groq API key
- **Frontend build errors**: Make sure all npm dependencies are installed correctly
- **Missing visualizations**: Check that the data returned from queries is suitable for visualization

## Development

### File Structure

- `app.py`: Main Flask application with all API endpoints
- `frontend/`: React frontend code
  - `src/`: Source files
    - `App.tsx`: Main application component
    - `components/`: UI components like History, Metrics pages
- `databases/`: Directory for SQLite databases
- `training_data/`: Example queries and responses
- `audio_cache/`: Temporary storage for voice recordings

### Adding New Features

1. Backend changes go in `app.py`
2. Frontend changes should be made in the `frontend/src` directory
3. After frontend changes, run `npm run build` again

### Custom Database Integration

To use your own SQLite database:
1. Place your .sqlite or .db file in the `databases/` directory
2. Connect to it through the UI
3. You can create training examples specific to your database by creating a JSON file with examples
