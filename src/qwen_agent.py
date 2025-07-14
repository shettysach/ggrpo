# /home/sword/Desktop/work/ggrpo/src/main.py

import json
from qwen_agent.llm import get_chat_model


# ----------------------------
# Tool implementations
# ----------------------------
def getA():
    return {"value": 46}


def getB():
    return {"value": 38}


def add(a: int, b: int):
    return {"result": a + b}


def get_function_by_name(name):
    if name == "getA":
        return getA
    if name == "getB":
        return getB
    if name == "add":
        return add
    raise ValueError(f"Unknown function: {name}")


# ----------------------------
# JSON-Schema tool definitions
# ----------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "getA",
            "description": "Return the integer A",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getB",
            "description": "Return the integer B",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    },
]

# Extract just the function schemas for model input
functions = [tool["function"] for tool in TOOLS]

# ----------------------------
# Initialize Qwen via Transformers
# ----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_agent.llm import Transformers

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # GPU-friendly dtype
    device_map="auto",  # Automatically place on GPU(s)
)

# Wrap manual model into Qwen-Agent's Transformers loader
llm = Transformers(
    {
        "model": model_name,
        "model_type": "transformers",
        "hf_model": hf_model,  # inject preloaded model
        "tokenizer": tokenizer,
        "generate_cfg": {},
    }
)

# ----------------------------
# Initial messages
# ----------------------------
messages = [
    {
        "role": "system",
        "content": "You are an assistant who must think before calling tools.",
    },
    {"role": "user", "content": "Add two numbers using tools."},
]

# ----------------------------
# Run LLM loop with tool calling
# ----------------------------
while True:
    for responses in llm.chat(messages=messages, functions=functions):
        messages.extend(responses)

        for msg in responses:
            if fn_call := msg.get("function_call"):
                fn_name = fn_call["name"]
                fn_args = json.loads(fn_call["arguments"])
                fn_result = get_function_by_name(fn_name)(**fn_args)
                messages.append(
                    {
                        "role": "function",
                        "name": fn_name,
                        "content": json.dumps(fn_result),
                    }
                )
                break  # Call LLM again with updated message history
        else:
            # No tool call, we assume the final answer
            print(messages[-1]["content"])
            exit()
