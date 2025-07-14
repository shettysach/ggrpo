import re
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
    return 45


def getB():
    return 56


def add(a, b):
    return a + b


TOOLS = {
    "getA": lambda: getA(),
    "getB": lambda: getB(),
    "add": lambda a, b: add(a, b),
}

# ----------------------------
# Prompt template
# ----------------------------
system_prompt = """You are an assistant who uses tools to compute answers step-by-step.

Available tools:
- getA() -> int
- getB() -> int
- add(int, int) -> int

You must think before every tool call.
Use this format:
<think>...</think>
<tool_call>ToolName[args]</tool_call>
Receive tool outputs like this:
<tool_response>output</tool_response>
Continue until you reach:
<final_answer>answer</final_answer>

Example: 
(The numbers in this example aren't necessarily the numbers you will recieve.)

<think>I need the first number.</think>
<tool_call>getA[]</tool_call>
<tool_response>2</tool_response> (Example, you may not get 2.)
<think>I need the second number.</think>
<tool_call>getB[]</tool_call>
<tool_response>3</tool_response> (Example, you may not get 3.)
<think>I will now add them.</think>
<tool_call>add[2, 3]</tool_call> 
<tool_response>5</tool_response> (Example, you may not get 5.)
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
            tool_match = re.match(r"(\w+)\[(.*?)\]", body)
            if not tool_match:
                print(f"Invalid tool call: {body}")
                continue

            tool, args = tool_match.groups()
            args = [eval(x) for x in args.split(",")] if args.strip() else []

            try:
                result = TOOLS[tool](*args)
            except Exception as e:
                result = f"Error: {str(e)}"

            print(f"<tool_response>{result}</tool_response>")
            history.append(
                {"role": "assistant", "content": f"<tool_call>{body}</tool_call>"}
            )
            history.append(
                {"role": "tool", "content": f"<tool_response>{result}</tool_response>"}
            )
            break  # go back to LLM

        elif tag == "final_answer":
            print(f"FINAL ANSWER: {body}")
            exit()

        else:
            print(f"<{tag}>{body}</{tag}>")

print("Reached step limit without final answer.")
