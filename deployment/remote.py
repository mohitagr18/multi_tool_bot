import os
import sys
import argparse

from dotenv import load_dotenv

import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines

from multi_tool.agent import root_agent


def create_remote_agent():
    """Deploys the agent to Vertex AI Agent Engine."""
    app = reasoning_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True
    )

    remote_app = agent_engines.create(
        agent_engine=app,
        display_name="test_agent4",
        requirements=[
            "google-adk>=1.7.0,<2.0.0",
            "google-cloud-aiplatform[adk,agent_engines]>=1.49.0",
            "pydantic>=2.11.3,<3.0.0",
            "cloudpickle>=3.1.0,<4.0.0"
        ],
        # Uncomment the line below if your agent code is in a local package
        extra_packages=["./multi_tool"]
    )

    print(f"✅ Created remote agent. Resource Name:\n{remote_app.resource_name}")
    # Extract the shorter resource ID for easier use
    resource_id = remote_app.resource_name.split('/')[-1]
    print(f"\nUse this Resource ID for other commands: {resource_id}")


def create_remote_session(resource_id: str, user_id: str):
    """Creates a new session to interact with a deployed agent."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    resource_name = f"projects/{project}/locations/{location}/reasoningEngines/{resource_id}"
    
    remote_app = agent_engines.get(resource_name=resource_name)
    remote_session = remote_app.create_session(user_id=user_id)
    
    print(f"✅ Created remote session. Session ID: {remote_session.id}")
    return remote_session.id


def send_message(resource_id: str, session_id: str, message: str, user_id: str):
    """Sends a message to an agent session and prints the streaming response."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    resource_name = f"projects/{project}/locations/{location}/reasoningEngines/{resource_id}"

    remote_app = agent_engines.get(resource_name=resource_name)
    
    print("Agent Response:")
    for event in remote_app.stream_query(
        user_id=user_id,
        session_id=session_id,
        message=message,
    ):
        # Print all event types for detailed observability
        print(event)


def main():
    """Initializes environment and handles command-line arguments."""
    load_dotenv()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    bucket_name = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")

    if not all([project_id, location, bucket_name]):
        print("❌ Missing required environment variables (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STAGING_BUCKET). Please check your .env file.")
        return
    
    print(f"Initializing Vertex AI:")
    print(f"  Project: {project_id}")
    print(f"  Location: {location}")
    print(f"  Staging Bucket: {bucket_name}")

    if os.getenv("RAG_CORPUS"):
        print(f"  RAG Region: {os.getenv('RAG_REGION', 'us-east4')}")

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket_name,
    )

    # create_remote_agent()
    # create_remote_session('5404200805588795392', 'test_user_234')
    # send_message('5404200805588795392', '6869123577984057344', "Hi how are you? What all can you do for me?", 'test_user_234')

    parser = argparse.ArgumentParser(description="Vertex AI Agent Deployment & Interaction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: create-agent
    subparsers.add_parser("create-agent", help="Deploy a new agent to Vertex AI.")

    # Command: create-session
    parser_create_session = subparsers.add_parser("create-session", help="Create a new interaction session for a deployed agent.")
    parser_create_session.add_argument("--resource-id", required=True, help="The unique ID of the deployed agent.")
    parser_create_session.add_argument("--user-id", default="test_user_123", help="A unique identifier for the end-user.")

    # Command: send-message
    parser_send = subparsers.add_parser("send-message", help="Send a message to an agent session.")
    parser_send.add_argument("--resource-id", required=True, help="The unique ID of the deployed agent.")
    parser_send.add_argument("--session-id", required=True, help="The ID of the session to use.")
    parser_send.add_argument("--message", required=True, help="The message to send.")
    parser_send.add_argument("--user-id", default="test_user_123", help="A unique identifier for the end-user.")

    args = parser.parse_args()

    if args.command == "create-agent":
        create_remote_agent()
    elif args.command == "create-session":
        create_remote_session(args.resource_id, args.user_id)
    elif args.command == "send-message":
        send_message(args.resource_id, args.session_id, args.message, args.user_id)


if __name__ == "__main__":
    main()


# python -m deployment.remote_send-message \
# --resource-id 5404200805588795392 \
# --session-id 6869123577984057344 \
# --user-id test_user_234 \
# --message "Hello, what can you do?"