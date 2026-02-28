# -------------------------
# Imports
# -------------------------

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from config import OPENAI_API_KEY
from tools import query_medgemma, call_emergency


# -------------------------
# Tool Definitions
# -------------------------

@tool
def ask_mental_health_specialist(query: str) -> str:
    """
    Generate a therapeutic response using the MedGemma model.
    Use this for general emotional or mental health related queries.
    """
    return query_medgemma(query)


@tool
def emergency_call_tool() -> str:
    """
    Place an emergency call via Twilio.
    Use ONLY if user expresses suicidal ideation or crisis.
    """
    call_emergency()
    return "Emergency services have been notified."


@tool
def find_nearby_therapists_by_location(location: str) -> str:
    """
    Finds licensed therapists near a specified location.
    """
    return (
        f"Here are some therapists near {location}:\n"
        "- Dr. Ayesha Kapoor - +1 (555) 123-4567\n"
        "- Dr. James Patel - +1 (555) 987-6543\n"
        "- MindCare Counseling Center - +1 (555) 222-3333"
    )


# -------------------------
# System Prompt
# -------------------------

SYSTEM_PROMPT = """
You are an AI engine supporting mental health conversations with warmth and vigilance.

You have access to three tools:

1. ask_mental_health_specialist  
   → Use this tool for emotional support, therapy-style responses, anxiety, depression, stress, or general mental health concerns.

2. find_nearby_therapists_by_location  
   → Use this tool when the user asks about therapists near a city or when recommending professional help locally.

3. emergency_call_tool  
   → Use this immediately if the user expresses suicidal thoughts, intent to self-harm, or is in immediate danger.

Always respond with empathy, clarity, and care.
"""


# -------------------------
# Create Agent
# -------------------------

tools = [
    ask_mental_health_specialist,
    emergency_call_tool,
    find_nearby_therapists_by_location,
]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
)

graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# -------------------------
# Optional: CLI Testing Mode
# -------------------------

def parse_response(stream):
    tool_called_name = "None"
    final_response = None

    for s in stream:
        tool_data = s.get("tools")
        if tool_data:
            tool_messages = tool_data.get("messages")
            if tool_messages:
                for msg in tool_messages:
                    tool_called_name = getattr(msg, "name", "None")

        agent_data = s.get("agent")
        if agent_data:
            messages = agent_data.get("messages")
            if messages:
                for msg in messages:
                    if msg.content:
                        final_response = msg.content

    return tool_called_name, final_response


if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")

        inputs = {
            "messages": [
                ("user", user_input)
            ]
        }

        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)

        print("\nTOOL CALLED:", tool_called_name)
        print("ANSWER:", final_response)