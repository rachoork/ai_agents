from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10 # no of max Agent Executions

@traceable(run_type="tool")
def get_product_price(product:str) -> float:
    """ Look for the price of the product in the Catalogue"""
    print(f" Executing the get_product_price tool for '{product}'")
    prices = {"laptop": 1299.99, "headphones": 149.99, "keyboard": 89.50}
    return prices.get(product, 0.0)

@traceable(run_type="tool")
def apply_discount(price:float, discount_tier: str) -> float:
    """Apply discount tier to the price and return the final price
        Available tiers are bronze, silver and gold """
    print(f" Executing the apply_discount tool for '{discount_tier}'")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    final_price = round(price * (1 - discount_percentages[discount_tier] / 100), 2)
    return final_price

# Agents Loop

@traceable(name="Langchain Agent Loop")
def run_agent(question:str):
    Model = "gpt-4o-mini"
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}
    # llm = init_chat_model(f"ollama: {Model}", temperature=0)
    llm = init_chat_model(f"openai:{Model}", temperature=0)

    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            "you are a shopping assistant"
            "You have access to product catalogue tool"
            "and a discount tool \n\n"
            "STRICT RULES - you must follow these exactly"
            "1. Never guess or assume any product price or discount"
            "You must call get_product_price to get price of the product"
            "2. Only call discount after you have received the price of the product" 
            "Pass exact price and do not pass a made up number"
            "3.Never calculate the discounts yourself using Math"
            "Always use apply_discount tool"
            "4. If user doesnot specify tier then do not assume tier no tier is applied"
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n Iteraiton {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        # messages.append(ai_message)
        tool_calls = ai_message.tool_calls

        # if no tool calls then this is the final answer

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content
        
        #  process only FIRST tool call - force one tool per iteration

        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected]: {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        observation = tool_to_use.invoke(tool_args)
        print(f"Observation: {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content = str(observation),
                tool_call_id = tool_call_id
            )
        )
    print("Error : Max Iterations reached without final answer")
    return None    




if __name__=="__main__":
    print("Starting Agent Loop")
    print()
    result = run_agent("What is the price of a laptop after applying gold discount?")
    print(result)

