import os
import asyncio
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
# from opentelemetry import trace
from phoenix.otel import register

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'), 
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_API_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature=0.7,
)

client = MultiServerMCPClient(
    {
        "math": {
            "command": "uv",
            "args": ["run", "servers/server_math.py"],
            "transport": "stdio",
        },
        "mindat_query_generation": {
            "command": "uv",
            "args": ["run", "servers/server_mindat_query_generation.py"],
            "transport": "stdio",
        }
    }
)


# async main function
async def main():
    
    # Initialize Phoenix tracer
    tracer_provider = register(
        project_name="mindat_ai_setup",
        auto_instrument=True,
        batch=True,
        protocol="http/protobuf", # mute the warning about default protocol
    )
    
    # Get mcp client tools
    tools = await client.get_tools()
    
    def filter_tools(tools, keywords):
        return [t for t in tools if any(k in t.name for k in keywords)]
    
    math_tools = filter_tools(tools, ["multiply", "add", "divide"])
    mindat_tools = filter_tools(tools, ["query_generation_tool"])

    math_agent = create_react_agent(
        model=llm,
        tools=math_tools,
        name="math_agent",
        prompt="You are a math agent that performs arithmetic calculations. You can use tools of multiply, add, and divide to perform calculations.",
    )
    
    mindat_agent = create_react_agent(
        model=llm,
        tools=mindat_tools,
        name="mindat_query_generation_agent",
        prompt="You are a mindat agent that can generate search query parameters for mineral entities in mindat database with the tools. You MUST return the raw json results from the tool without any additional explanation or text.",
    )
    


    workflow = create_supervisor(
        [math_agent, mindat_agent],
        model=llm,
        prompt=(
            "You are a team supervisor managing a geological data processing pipeline with the agents."
            "For user requests involving mineral data queries, always delegate to mindat_query_generation_agent and return its raw results in json."
            "You should respond to the user requests by delegating tasks to the appropriate agents, and terminate when the tasks are completed or invalid."
            "AGENTS:"
            "- math_agent: For mathematical calculations"
            "- mindat_query_generation_agent: For mineral entity queries, will generate structured search parameters for querying the mindat database."
        ),
    )
    
    # Compile and run
    app = workflow.compile(name="top_level_supervisor")
    result = await app.ainvoke({
        "messages": [
            {
                "role": "user",
                # "content": "Please help me calculate the result of 123123+88888"# using the math agent",
                # "content": "This is a test message to the mindat agent. Please call the mindat agent to generate the search query for quartz in mindat database.",
                "content": "Query the ima-approved mineral species with hardness between 3-5, in Hexagonal crystal system, must have Neodymium, but without sulfur."
                # "content": "Please help me extract the text from the PDF files in 'data/source' using OCR agent, then process the text entries using the preprocessing agent, and normalize the mineral and rock entities using the mindat agent. Finally, match the geosciml vocabulary using the geosciml agent.",
            }
        ]
    }, {"recursion_limit": 25})


    for m in result["messages"]:
        m.pretty_print()
        
    # Ensure all phoenix traces are sent before shutdown
    # Explicit cleanup at the end of main
    if hasattr(tracer_provider, 'force_flush'):
        success = tracer_provider.force_flush(timeout_millis=30000)

    
    # Only shutdown if this is truly the end
    if hasattr(tracer_provider, 'shutdown'):
        tracer_provider.shutdown()



# execute the main function
if __name__ == "__main__":
    asyncio.run(main())