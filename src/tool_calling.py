from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import Retriever, NODE_TEXT_KEYS
from graph_funcs import graph_funcs
import json
import re
import csv
import os
import torch


graph_path = "/home/sword/Desktop/work/ggrpo/data/biomedical/graph.json"
graph = json.load(open(graph_path))

data_path = "/home/sword/Desktop/work/ggrpo/data/biomedical/data.json"
dataset = json.load(open(data_path))

# ----------------------------
# Model setup
# ----------------------------

model_name = "qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda"
)


# ----------------------------
# Tool definitions
# ----------------------------

# Load graph and initialize tools


class Args:
    dataset = "biomedical"
    faiss_gpu = False  # WARN:
    embedder_name = "sentence-transformers/all-mpnet-base-v2"
    embed_cache = True
    embed_cache_dir = "/tmp"  # Adjust to a writable cache dir
    node_text_keys = NODE_TEXT_KEYS["biomedical"]


args = Args()
retriever = Retriever(args, graph)
graph_tool = graph_funcs(graph)


def RetrieveNode(key: str):
    node_id, _ = retriever.search_single(key)
    return node_id


def NodeFeature(node: str, feature: str):
    return graph_tool.check_nodes(node, feature)


def NodeDegree(node: str, neighbor_type: str):
    return graph_tool.check_degree(node, neighbor_type)


def NeighbourCheck(node: str, neighbor_type: str):
    return graph_tool.check_neighbours(node, neighbor_type)


TOOLS = {
    "RetrieveNode": RetrieveNode,
    "NodeFeature": NodeFeature,
    "NodeDegree": NodeDegree,
    "NeighbourCheck": NeighbourCheck,
}

# ----------------------------
# Prompt template
# ----------------------------

graph_description = """
There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
Each node has the feature 'name'.
There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
"""


examples = [
    """Question: What compounds can be used to treat Crohn's disease? Give the final answer in compound names rather than IDs.
<think>The question is related to a disease node (Crohn's disease). We need to find the node in the graph.</think>
<tool_call>RetrieveNode[Crohn's disease]</tool_call>
<tool_response>DOID:8778</tool_response>
<think>The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph.</think>
<tool_call>NeighbourCheck[DOID:8778, Compound-treats-Disease]</tool_call>
<tool_response>['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033']</tool_response>
<think>We need to get the names of these compound IDs.</think>
<tool_call>NodeFeature[DB01014, name], NodeFeature[DB00244, name], NodeFeature[DB00795, name], NodeFeature[DB00993, name], NodeFeature[DB00635, name], NodeFeature[DB01033, name]</tool_call>
<tool_response>Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine</tool_response>
<final_answer>[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]</final_answer>""",
    #
    """How many side effects does Caffeine have?
Question: What is the inchikey of Caffeine?
<think>The question is related to a compound node (Caffeine). We need to find the node in the graph.</think>
<tool_call>RetrieveNode[Caffeine]</tool_call>
<tool_response>DB00201</tool_response>
<think>The question is asking the inchikey feature of the compound node.</think>
<tool_call>NodeFeature[DB00201, inchikey]</tool_call>
<tool_response>InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N</tool_response>
<final_answer>InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N</final_answer>""",
    #
    """
<think>The question is related to a compound node (Caffeine). We need to find the node in the graph.</think>
<tool_call>RetrieveNode[Caffeine]</tool_call>
<tool_response>DB00201</tool_response>
<think>We need to calculate the number of 'Compound-causes-Side Effect' neighbors.</think>
<tool_call>NodeDegree[DB00201, Compound-causes-Side Effect]</tool_call>
<tool_response>58</tool_response>
<final_answer>58</final_answer>""",
]

example_block = "\n\n".join(examples)

system_prompt = """You are an assistant who uses tools to compute answers step-by-step. 

Graph description:
{graph_description}

Available tools:
- RetrieveNode(str) -> str               # returns a node ID given a name
- NodeFeature(str node_id, str feature) -> str   # returns a feature of the node
- NodeDegree(str node_id, str neighbor_type) -> str   # returns count of neighbors
- NeighbourCheck(str node_id, str neighbor_type) -> str   # returns list of neighbor IDs

You must think before every tool call.
Use this format:
<think>...</think>
<tool_call>ToolName[args]</tool_call>
Receive tool outputs like this:
<tool_response>output</tool_response>
Continue until you reach:
<final_answer>answer</final_answer>

You can call multiple tool calls at a time, separated by ','.

Example:
<tool_call>ToolName[a], ToolName[b]</tool_call>

Examples:
{example_block}

Assume you have no prior biomedical data, excluding the graph description and tools.
Now solve this.
"""

# Think at each step what data you have. Do not hallucinate data or tool calls. If you need data use tool calls.


formatted_prompt = system_prompt.format(
    graph_description=graph_description,
    example_block=example_block,
)

pattern = re.compile(r"<(think|tool_call|tool_response|final_answer)>(.*?)</\1>", re.S)


def run_query(query: str, max_steps: int = 20, print_out: bool = False):
    history = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": f"Question: {query}"},
    ]

    if not print_out:
        print("steps - ", end="", flush=True)

    for step in range(max_steps):
        if not print_out:
            print(f"{step} ", end="", flush=True)

        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256)[0][
            inputs.input_ids.shape[1] :
        ]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        blocks = pattern.findall(response)
        for tag, body in blocks:
            body = body.strip()

            if tag == "think":
                if print_out:
                    print(f"<think>{body}</think>")

                history.append(
                    {"role": "assistant", "content": f"<think>{body}</think>"}
                )

            elif tag == "tool_call":
                if print_out:
                    print(f"<tool_call>{body}</tool_call>")

                calls = re.findall(r"(\w+\[.*?\])", body)
                full_tool_call = f"<tool_call>{body}</tool_call>"
                tool_outputs = []

                for call in calls:
                    tool_match = re.match(r"(\w+)\[(.*?)\]", call)
                    if not tool_match:
                        if print_out:
                            print(f"Invalid tool call: {call}")
                        continue

                    tool, args = tool_match.groups()
                    args = (
                        [x.strip().strip('"').strip("'") for x in args.split(",")]
                        if args.strip()
                        else []
                    )

                    try:
                        result = TOOLS[tool](*args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    tool_outputs.append(str(result))

                joined_response = ", ".join(tool_outputs)
                if print_out:
                    print(f"<tool_response>{joined_response}</tool_response>")

                history.append({"role": "assistant", "content": full_tool_call})
                history.append(
                    {
                        "role": "tool",
                        "content": f"<tool_response>{joined_response}</tool_response>",
                    }
                )
                break  # Return to LLM for next step

            elif tag == "final_answer":
                if print_out:
                    print(f"<final_answer>{body}</final_answer>")
                return body

            else:
                if print_out:
                    print(f"<{tag}>{body}</{tag}>")

    if print_out:
        print("Reached step limit without final answer.")

    return None


def save_to_csv(print_out: bool):
    output_csv_path = "./results.csv"
    file_exists = os.path.isfile(output_csv_path)

    with open(output_csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["qid", "question", "predicted_answer", "gt_answer"]
        )

        if not file_exists:
            writer.writeheader()

        for item in dataset[11:]:
            qid = item["qid"]
            question = item["question"]
            real = item["answer"]
            pred = run_query(question, 25, print_out)

            print("")
            print(f"pred: {pred}")
            print(f"real: {real}")
            print("")

            row = {
                "qid": qid,
                "question": question,
                "predicted_answer": pred,
                "gt_answer": real,
            }
            writer.writerow(row)

    print(f"Saved to {output_csv_path}")


if __name__ == "__main__":
    # save_to_csv(True)
    run_query(
        "Can you tell me the count of compounds that are similar to Pipazethate?",
        max_steps=25,
        print_out=True,
    )
