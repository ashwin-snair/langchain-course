from dotenv import load_dotenv
import re
import inspect
load_dotenv()
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langsmith import traceable
import ollama

MAX_ITERATIONS_= 5
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product_name: str) -> str:
    '''Look up a price in catalog'''
    print(f"Looking up price for {product_name}...")
    prices = {
        "laptop": "$999",
        "smartphone": "$699",
        "headphones": "$199",
    }
    return prices.get(product_name.lower(), 0)

@traceable(run_type="tool")
def apply_discount(price: str, discount_tier: str) -> str:
    '''Apply a discount to a price and return the discounted price
    Available tiers: Gold: 10%, Silver: 5%, Bronze: 2%
    '''
    print(f"Applying {discount_tier} discount to {price}...")
    discount_percentages = {
        "gold": 10,
        "silver": 5,
        "bronze": 2
    }
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    discount = float(price.strip('$')) * (discount_percentage / 100)
    return f"${float(price.strip('$')) - discount:.2f}"

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        original_function = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature}: {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ','.join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. Never guess or assume product prices
2. Call apply discount only after getting the product price, do not pass a made up number
3. Never apply discount yourself using math, always call the tool
4. If user does not specify a discount tier, ask them to choose one of the following: Gold, Silver, Bronze

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""

@traceable(name = "Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# Agent Loop
@traceable(name = "Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("="*50)

    prompt = react_prompt.format(question = question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS_ + 1):
        print(f"\nIteration {iteration}:")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop":["\nObservation:"], "temperature": 0}
        )

        output = response.message.content
        print(f"AI output: {output}")

        print("Looking for final answer in the output...")
        final_answer_match = re.search(r"Final Answer:\s*(.*)", output, re.IGNORECASE)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("="*50)
            print(f"Final answer: {final_answer}")
            break

        print("No final answer found, looking for tool calls...")
        tool_call_match = re.search(r"Action:\s*(\w+)\s*Action Input:\s*(.*)", output, re.IGNORECASE)

        if not tool_call_match:
            print("No tool call found in the output. Stopping the agent.")
            break

        tool_name = tool_call_match.group(1).strip()
        tool_input_raw = tool_call_match.group(2).strip()

        # Handle JSON dict/list, or fall back to comma-separated values
        try:
            parsed = json.loads(tool_input_raw.strip())
            if isinstance(parsed, dict):
                args = [str(v) for v in parsed.values()]
            elif isinstance(parsed, list):
                args = [str(v) for v in parsed]
            else:
                args = [str(parsed)]
        except Exception:
            cleaned_input = tool_input_raw.strip().strip("[](){}")
            raw_args = [arg.strip() for arg in cleaned_input.split(',') if arg.strip()]
            args = [arg.split('=', 1)[-1].strip().strip("'\": ") for arg in raw_args]

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found."
            print(observation)
        else:
            tool_func = tools[tool_name]
            try:
                observation = tool_func(*args)
            except Exception as e:
                observation = f"Error calling tool '{tool_name}': {str(e)}"
                print(observation)

        scratchpad += f"\nThought: I have called the tool {tool_name}\nAction: {tool_name}\nAction Input: {args}\nObservation: {observation}\n"
    pass


if __name__ == "__main__":
    question = "Hello Langchain Agent"
    run_agent("What is the price of a laptop after applying a gold tier discount?")