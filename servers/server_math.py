from mcp.server.fastmcp import FastMCP
from opentelemetry import trace
import time
from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

mcp = FastMCP("Math_test")
tracer_provider = register(
    project_name="mindat_ai_setup", 
    auto_instrument=True,
    batch=True
)

tracer = tracer_provider.get_tracer(__name__)

@mcp.tool()
@tracer.tool(name= __name__ + ".add") 
def add(a: int, b: int) -> int:
    """Add two numbers"""
    with tracer.start_as_current_span(
        "add_tool_span"
    ) as span:
        time.sleep(0.1)  # Simulate some processing time
        span.set_attribute("component", "math_tool")
        span.set_attribute("operation", "add")
        span.set_attribute("input_a", a)
        span.set_attribute("input_b", b)
        result = a + b
        span.set_attribute("result", result)
        return result
    # return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b

if __name__ == "__main__":
    mcp.run(transport="stdio")