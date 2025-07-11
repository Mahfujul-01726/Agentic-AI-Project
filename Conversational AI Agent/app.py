# conversational_agent_app.py - Build a full LangChain-powered conversational agent with tools and memory

import os
import openai
import datetime
import requests
import wikipedia
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
import panel as pn
import param
from langchain_community.chat_models import ChatOpenAI

# Load API Key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


# ========== Define Tools ==========
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature using Open-Meteo API for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return "Weather API failed."
    data = response.json()
    now = datetime.datetime.utcnow()
    times = [datetime.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data['hourly']['time']]
    temps = data['hourly']['temperature_2m']
    index = min(range(len(times)), key=lambda i: abs(times[i] - now))
    return f"The current temperature is {temps[index]}°C"

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a query and return top summaries."""
    titles = wikipedia.search(query)
    summaries = []
    for title in titles[:3]:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            summaries.append(f"**{title}**\n{page.summary}")
        except:
            pass
    return "\n\n".join(summaries) if summaries else "No good result found."

@tool
def create_your_own(query: str) -> str:
    """Reverse the input text as a custom example tool."""
    return f"You sent: {query}. This reverses it: {query[::-1]}"

# ========== Register Tools ==========
tools = [get_current_temperature, search_wikipedia, create_your_own]

# ========== Panel Chatbot UI ==========
pn.extension()

class ConversationalBot(param.Parameterized):
    def __init__(self, tools, **params):
        super().__init__(**params)
        self.panels = []
        self.tool_funcs = [format_tool_to_openai_function(t) for t in tools]
        self.llm = ChatOpenAI(temperature=0).bind(functions=self.tool_funcs)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.llm | OpenAIFunctionsAgentOutputParser()

        self.executor = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)

    def interact(self, query):
        if not query:
            return
        result = self.executor.invoke({"input": query})
        self.answer = result['output']
        
        # Enhanced styling for chat messages
        user_message = pn.pane.Markdown(
            f"**You:** {query}", 
            width=600,
            styles={
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "color": "white",
                "padding": "15px 20px",
                "border-radius": "20px 20px 5px 20px",
                "margin": "10px 0",
                "box-shadow": "0 4px 15px rgba(0,0,0,0.1)",
                "font-size": "14px"
            }
        )
        
        bot_message = pn.pane.Markdown(
            f"**🤖 Assistant:** {self.answer}", 
            width=600,
            styles={
                "background": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                "color": "white",
                "padding": "15px 20px",
                "border-radius": "20px 20px 20px 5px",
                "margin": "10px 0",
                "box-shadow": "0 4px 15px rgba(0,0,0,0.1)",
                "font-size": "14px"
            }
        )
        
        self.panels.extend([
            pn.Row(pn.Spacer(width=50), user_message),
            pn.Row(bot_message, pn.Spacer(width=50))
        ])
        return pn.WidgetBox(*self.panels, scroll=True, styles={"max-height": "500px"})


# ========== Launch the Enhanced Panel Chat App ==========
cb = ConversationalBot(tools)

# Enhanced input field with modern styling
inp = pn.widgets.TextInput(
    placeholder='💬 Ask me anything... (Weather, Wikipedia, or custom queries)',
    width=600,
    height=50,
    styles={
        "border": "2px solid #e0e0e0",
        "border-radius": "25px",
        "padding": "10px 20px",
        "font-size": "16px",
        "background": "white",
        "box-shadow": "0 2px 10px rgba(0,0,0,0.1)"
    }
)

# Send button with attractive styling
send_btn = pn.widgets.Button(
    name="Send 🚀",
    button_type="primary",
    width=100,
    height=50,
    styles={
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "border": "none",
        "border-radius": "25px",
        "color": "white",
        "font-weight": "bold",
        "cursor": "pointer",
        "box-shadow": "0 4px 15px rgba(0,0,0,0.2)"
    }
)

# Bind the send button to the interact function
def send_message(event):
    if inp.value.strip():
        cb.interact(inp.value)
        inp.value = ""

send_btn.on_click(send_message)

conversation = pn.bind(cb.interact, inp)

# Enhanced layout with modern design
input_row = pn.Row(
    pn.Spacer(width=50),
    inp,
    pn.Spacer(width=20),
    send_btn,
    pn.Spacer(width=50),
    styles={
        "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        "padding": "20px",
        "border-radius": "15px",
        "margin": "20px 0"
    }
)

# Main tab with enhanced styling
tab = pn.Column(
    input_row,
    pn.layout.Divider(styles={"border-color": "#e0e0e0", "margin": "20px 0"}),
    pn.panel(
        conversation, 
        loading_indicator=True, 
        height=500,
        styles={
            "background": "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
            "border-radius": "15px",
            "padding": "20px",
            "margin": "10px 0"
        }
    ),
    styles={
        "background": "white",
        "border-radius": "20px",
        "box-shadow": "0 8px 30px rgba(0,0,0,0.1)",
        "padding": "30px",
        "margin": "20px"
    }
)

# Enhanced header with gradient and styling
header = pn.pane.Markdown(
    '# 🧠 AI Conversational Agent',
    styles={
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "color": "white",
        "padding": "30px",
        "text-align": "center",
        "border-radius": "20px",
        "margin": "20px",
        "box-shadow": "0 8px 30px rgba(0,0,0,0.2)",
        "font-size": "28px"
    }
)

# Feature highlights
features = pn.pane.Markdown(
    """
    ### 🌟 **Available Features:**
    - 🌡️ **Weather Information** - Get current temperature for any location
    - 📚 **Wikipedia Search** - Search and get summaries from Wikipedia
    - 🔄 **Custom Tools** - Text reversal and custom queries
    - 💭 **Memory** - Remembers conversation context
    """,
    styles={
        "background": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
        "padding": "20px",
        "border-radius": "15px",
        "margin": "20px",
        "border-left": "5px solid #667eea"
    }
)

# Enhanced dashboard with modern layout
dashboard = pn.Column(
    header,
    features,
    pn.Tabs(
        ('💬 Chat Interface', tab),
        styles={
            "background": "white",
            "border-radius": "15px",
            "box-shadow": "0 4px 20px rgba(0,0,0,0.1)",
            "margin": "20px"
        }
    ),
    styles={
        "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        "min-height": "100vh",
        "padding": "20px"
    }
)

dashboard.servable()