from google.adk.agents import Agent
import vertexai
from vertexai.preview import rag
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
import datetime
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv

load_dotenv()


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

rag_corpus = os.getenv("RAG_CORPUS")
rag_region = os.getenv("RAG_REGION", "us-east4") 
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
vertexai.init(project=project_id, location=rag_region)
rag_tool = VertexAiRagRetrieval(
    name="rag_retrieval",
    description="Retrieve passages from the Vertex AI RAG corpus for grounded answers.",
    rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
    similarity_top_k=5  # small k for precision in tests
)

root_agent = Agent(
    name="multi_tool_bot",
    model='gemini-2.0-flash',
    description="A multi-tool bot that can use multiple tools to perform tasks",
    instruction="You are a helpful assistant that can use multiple tools to answer user queries. Cite retrieved sources for RAG answers.",
    tools=[get_current_time, get_weather, rag_tool]
)