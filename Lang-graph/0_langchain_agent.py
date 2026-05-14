from dotenv import load_dotenv
# from langchain import hub
from langchain_classic.agents import AgentExecutor,create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 2. Define the Tool (Our Scout)
search_tool = TavilySearchResults(k=3)
tools = [search_tool]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0 # Keep it at 0 for analytical tasks so it doesn't hallucinate tools
)

"""# 4. Pull the ReAct Prompt Blueprint
prompt = hub.pull("hwchase17/react")""";

from langsmith import Client

client = Client()
prompt = client.pull_prompt("hwchase17/react")

# 5. Construct the ReAct Agent
agent = create_react_agent(llm, tools, prompt)

# 6. Create the Executor (The Body)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # Essential for watching the Thought/Action/Observation loop unfold
    handle_parsing_errors=True
)

# 7. Execute the Mission
task = "What is the current stock price of NVIDIA and how does it compare to its price at the start of 2024?"
response = agent_executor.invoke({"input": task})

print("Final Answer ---")
print(response["output"])