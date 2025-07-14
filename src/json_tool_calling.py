import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Model setup
# ----------------------------
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)


# ----------------------------
# Tool definitions
# ----------------------------
def getA():
    return {"value": 46}


def getB():
    return {"value": 38}


def add(a, b):
    return {"value": a + b}


TOOLS = {
    "getA": lambda: getA(),
    "getB": lambda: getB(),
    "add": lambda a, b: add(a, b),
}

# ----------------------------
# Prompt template (updated)
# ----------------------------
system_prompt = """You are an assistant who uses tools to compute answers step-by-step.

Available tools:
- getA() -> int
- getB() -> int
- add(int, int) -> int

You must think before every tool call.
Use this format:
<think>...</think>
<tool_call>{"name": ..., "arguments": {...}}</tool_call>
Receive tool outputs like this:
<tool_response>output</tool_response>
Continue until you reach:
<final_answer>answer</final_answer>

Example: 
(The numbers in this example aren't necessarily the numbers you will receive.)

<think>I need the first number.</think>
<tool_call>{"name": "getA", "arguments": {}}</tool_call>
<tool_response>{"value": 2}</tool_response> (Example, you may not get 2.)
<think>I need the second number.</think>
<tool_call>{"name": "getB", "arguments": {}}</tool_call>
<tool_response>{"value": 3}</tool_response> (Example, you may not get 3.)
<think>I will now add them.</think>
<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>
<tool_response>{"value": 5}</tool_response> (Example, you may not get 5.)
<final_answer>5</final_answer>

Now solve this:
<think>Let's begin.</think>
"""

# ----------------------------
# Initial message history
# ----------------------------
history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Add two numbers using tools."},
]

# ----------------------------
# Block extraction pattern
# ----------------------------
pattern = re.compile(r"<(think|tool_call|tool_response|final_answer)>(.*?)</\1>", re.S)

# ----------------------------
# Loop: LLM ↔ Tool ↔ LLM
# ----------------------------
for step in range(10):
    # Format current history as prompt
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    output_ids = model.generate(**inputs, max_new_tokens=256)[0][
        inputs.input_ids.shape[1] :
    ]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # Extract block(s)
    blocks = pattern.findall(response)

    # Handle each block
    for tag, body in blocks:
        body = body.strip()

        if tag == "think":
            print(f"<think>{body}</think>")
            history.append({"role": "assistant", "content": f"<think>{body}</think>"})

        elif tag == "tool_call":
            print(f"<tool_call>{body}</tool_call>")
            try:
                payload = json.loads(body)
                tool = payload["name"]
                args = payload.get("arguments", {})

                result = TOOLS[tool](**args)
                tool_result_str = json.dumps(result)

                print(f"<tool_response>{tool_result_str}</tool_response>")
                history.append(
                    {"role": "assistant", "content": f"<tool_call>{body}</tool_call>"}
                )
                history.append(
                    {
                        "role": "tool",
                        "content": f"<tool_response>{tool_result_str}</tool_response>",
                    }
                )
                break  # Go back to LLM with tool response

            except Exception as e:
                error = f"Error: {str(e)}"
                print(f"<tool_response>{error}</tool_response>")
                history.append(
                    {"role": "assistant", "content": f"<tool_call>{body}</tool_call>"}
                )
                history.append(
                    {
                        "role": "tool",
                        "content": f"<tool_response>{error}</tool_response>",
                    }
                )
                break

        elif tag == "final_answer":
            print(f"FINAL ANSWER: {body}")
            exit()

        else:
            print(f"<{tag}>{body}</{tag}>")

print("Reached step limit without final answer.")
