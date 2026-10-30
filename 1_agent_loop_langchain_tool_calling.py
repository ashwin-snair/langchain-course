from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langsmith import traceable

MAX_ITERATIONS_= 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product_name: str) -> str:
    '''Look up a price in catalog'''
    print(f"Looking up price for {product_name}...")
    prices = {
        "laptop": "$999",
        "smartphone": "$699",
        "headphones": "$199",
    }
    return prices.get(product_name.lower(), 0)

@tool
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

# Agent Loop
@traceable(name = "Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant."
            "You have access to a product catalog and can apply discounts to products."
            "You must strict rules"
            "1. Never guess or assume product prices"
            "2. Call apply discount only after getting the product price, do not pass a made up number"
            "3. Never apply discount yourself using math, always call the tool"
            "4. If user does not specify a discount tier, ask them to choose one of the following: Gold, Silver, Bronze"
        )),
        HumanMessage(content=question)   
    ]

    for iteration in range(1, MAX_ITERATIONS_ + 1):
        print(f"\nIteration {iteration}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            break

        else:
            print(f"Tool calls: {tool_calls}")
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_call_id = tool_call.get("id")
            tool_args = tool_call.get("args", {})

            print(f"[Tool Selected] {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool {tool_name} not found")

            observation = tool_to_use.invoke(tool_args)
            print(f"[Tool Observation] {observation}")

            messages.append(ai_message)
            messages.append(
                ToolMessage(content=observation, tool_name=tool_name, tool_call_id=tool_call_id)
            )

        if iteration == MAX_ITERATIONS_:
            print("Max iterations reached. Stopping the agent.")
            break
    pass


if __name__ == "__main__":
    question = "Hello Langchain Agent"
    run_agent("What is the price of a laptop after applying a gold tier discount?")