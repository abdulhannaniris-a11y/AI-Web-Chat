# AI Web Chat

An AI website chat assistant built with FastAPI. Enter a public website URL, let the app extract its readable text, and ask questions about that website using a Groq-hosted language model.

## Features

- Fetches and extracts readable text from HTTP and HTTPS websites
- Answers questions using only the loaded website content
- Supports follow-up questions with short in-memory conversation history
- Includes a browser interface served directly by FastAPI
- Uses environment variables for API credentials

## Requirements

- Python 3.10 or newer
- A Groq API key

## Installation

Clone the repository and open its directory:

```powershell
git clone https://github.com/abdulhannaniris-a11y/AI-Web-Chat.git
cd AI-Web-Chat
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can optionally choose a different Groq model:

```env
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit or share `.env`. It is excluded by `.gitignore`.

## Run Locally

Start the development server:

```powershell
python -m uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## API Endpoints

### Load a website

```http
POST /api/load-website
Content-Type: application/json
```

Request:

```json
{
	"url": "https://example.com"
}
```

### Ask a question

```http
POST /api/ask
Content-Type: application/json
```

Request:

```json
{
	"question": "What is this website about?"
}
```

Load a website before asking questions. The application stores the current website and conversation history in memory, so the state is cleared when the server restarts or a new website is loaded.

## Project Structure

```text
.
├── main.py            # FastAPI application and embedded frontend
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── .gitignore         # Local files and secrets excluded from Git
```

## Notes

- The website must return readable HTML or text content.
- Extracted content is limited to 12,000 characters per website.
- The application is currently a single-user prototype with in-memory state.
- Do not expose your Groq API key in frontend code or commit it to GitHub.

## License

No license has been specified yet.
