from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import BaseModel
from agno.agent import Agent
import os
from typing import List, Any # Import Any
import requests
from agno.models.google import Gemini
from textwrap import dedent
from dotenv import load_dotenv
from agno.tools import Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.crawl4ai import Crawl4aiTools
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Define the request body model
class RequestData(BaseModel):
    content: str

# Define the response body model for HTML content and messages
class HtmlResponse(BaseModel):
    html_content: str
    messages: List[Any] # Allow any type for messages

# --- Agent Setup Logic (Moved from agent.py) ---

class GoogleImageSearchTools(Toolkit):
    def __init__(self, api_key: str, cse_id: str):
        super().__init__(name="google_image_search_tools")
        self.api_key = api_key
        self.cse_id = cse_id
        self.register(self.search_images)

    def search_images(self, query: str) -> List[str]:
        """
        Search for images using Google Custom Search API.

        Args:
            query (str): The search term.

        Returns:
            list: A list of image URLs.
        """
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.cse_id,
            'searchType': 'image',
            'num': 3
        }

        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise Exception(f"Google API Error: {response.status_code} - {response.text}")

        results = response.json()
        image_urls = [item['link'] for item in results.get('items', [])]

        return image_urls

# Get keys from .env file
API_KEY_CUSTOM_SEARCH = os.environ.get("API_KEY_CUSTOM_SEARCH")
CSE_ID = os.environ.get("CSE_ID")

# Initialize the agent
agent = Agent(
    model=Gemini(id="gemini-2.5-pro-preview-03-25"),
    description=dedent("""You are an advanced Notes Generator / Answer generator Agent designed to create comprehensive study notes based on syllabus topics. When provided with syllabus topics, you'll research each topic thoroughly and create well-organized, educational notes.
Your Workflow:
-Receive a list of syllabus topics/ questions from the user
-You will only reply with html code for webpage the notes or answers
                       
First generate a function calling plan and call the functions's then generate the notes

For each topic:
1)Use DuckDuckGo search tool to gather accurate, up-to-date information
2)Add visual aids descriptions where appropriate (diagrams, charts, etc.) from google image search               
3)Analyze and synthesize the information from multiple sources(use crawl4ai tool to read webpages if needed)
4)Structure the information into clear, concise notes
5)Include key definitions, concepts, examples, and applications


Websites Priority for Research:              
-https://www.geeksforgeeks.org/ (for computer science and programming questions)
-https://www.tutorialspoint.com/ (for technical concepts and explanations)
-https://www.javatpoint.com/ (for programming and technical content)

Note Format:
-Output html code for the notes

For Topics:    
- Topic Title: Clear heading for each topic
- Key Concepts: Essential terms and ideas
- Detailed Explanation: Thorough but concise explanation
- Examples: Practical applications or illustrations
- Important Relationships: How this topic connects to others
- Summary: Brief recap of main points
- Practice Questions: (Optional) 2-3 questions to test understanding
                       
For Questions:
- answers will be thorough but concise, focusing on clarity

Guidelines:

- Include 2-3 images in each topics or answer from google image search
- Be thorough but concise - prioritize clarity over verbosity
- Use bullet points, numbering, and headings for easy readability
- Include both theoretical knowledge and practical applications
- Cite credible sources when appropriate
- Adapt the depth based on the apparent educational level of the topics
- Format notes consistently across all topics
- Focus on accuracy and educational value

Style Guide:
- Use modern and minimal styles for education content
- Use subtle gray ,white and blue colors"""),
    tools=[DuckDuckGoTools(fixed_max_results=5),Crawl4aiTools(max_length=None), GoogleImageSearchTools(api_key=API_KEY_CUSTOM_SEARCH, cse_id=CSE_ID) ],
    debug_mode=True,
)

# --- FastAPI Application ---

app = FastAPI()

origins = [
    "http://localhost:3000", # Your frontend origin
    "http://localhost:3001", # Added origin from user feedback
    "https://gennotes-frontend.vercel.app", # Allow frontend origin
    # Add any other origins if needed
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.post("/generate", response_model=HtmlResponse) # Specify the response model
async def generate_notes(request: Request):
    """
    Accepts syllabus content or question paper text via JSON body ({ "content": "..." })
    and returns the generated notes/answers as an HTML string within a JSON response: {"html_content": "..."}.
    """
    try:
        data = await request.json()
        content = data.get("content")

        if not content:
            raise HTTPException(status_code=400, detail="Missing 'content' in request body")

        # Generate the full response directly
        try:
            response = agent.run(content)
            print()
            # Manually create a serializable list of messages
            serializable_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in agent.memory.messages
                if hasattr(msg, 'role') and hasattr(msg, 'content') # Ensure messages have expected attributes
            ]
            return {"html_content": response.content, "messages": serializable_messages}
        except Exception as e:
            # Log the error server-side
            print(f"Error during agent execution: {e}")
            # Raise an HTTP exception to send a standard error response
            raise HTTPException(status_code=500, detail=f"An error occurred during generation: {str(e)}")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Handle other errors (e.g., request parsing)
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

# --- Optional: Add root endpoint for basic info ---
@app.get("/")
async def root():
    return {"message": "Notes Generator API is running. Use the /generate endpoint (POST) to submit content."}

