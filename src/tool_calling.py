from vllm import LLM, SamplingParams
from retriever import Retriever, NODE_TEXT_KEYS
from graph_funcs import graph_funcs
import json
import re
import csv
import os

# ----------------------------
# Load graph and dataset
# ----------------------------

graph_path = "/workspace/graph.json"
graph = json.load(open(graph_path))

data_path = "/workspace/data.json"
dataset = json.load(open(data_path))

# ----------------------------
# Model setup with vLLM
# ----------------------------

llm = LLM(
    "qwen/Qwen3-4B-Thinking-2507",
    gpu_memory_utilization=0.9,
    max_num_seqs=2,
    max_model_len=4096,
    swap_space=24,
)

# ----------------------------
# Tool definitions
# ----------------------------


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

Examples:
{example_block}

You need not call tool calls one by one. You can call multiple tool calls at a time, separated by ",".

Example:
<tool_call>ToolName[a], ToolName[b], ToolName[c]</tool_call>
<tool_response>output a, output b, output c</tool_response>


Assume you have no prior biomedical data, excluding the graph description and tools.
Now solve this.
"""

# Think at each step what data you have. Do not hallucinate data or tool calls. If you need data use tool calls.

formatted_prompt = system_prompt.format(
    graph_description=graph_description,
    example_block=example_block,
)

pattern = re.compile(r"<(think|tool_call|tool_response|final_answer)>(.*?)</\1>", re.S)


def run_query(
    query: str,
    max_think_steps: int = 25,
    hard_cap: int = 100,  # absolute safety cap
    print_out: bool = False,
):
    history = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": f"Question: {query}"},
    ]

    step_total = 0  # all model calls
    step_think = 0  # <think> turns that count

    if not print_out:
        print("think-steps - ", end="", flush=True)

    while step_think < max_think_steps and step_total < hard_cap:
        # ---------------------------------------------------------------
        # 1) Call the model
        # ---------------------------------------------------------------
        sampling_params = SamplingParams(max_tokens=256, temperature=0.6)
        chat_out = llm.chat(
            [history],
            sampling_params,
            chat_template_kwargs={"enable_thinking": True},
        )
        step_total += 1

        response = chat_out[0].outputs[0].text.strip()
        blocks = pattern.findall(response)

        # ---------------------------------------------------------------
        # 2) Parse the response
        # ---------------------------------------------------------------
        for tag, body in blocks:
            body = body.strip()

            if tag == "think":
                step_think += 1
                if not print_out:
                    print(f"{step_think} ", end="", flush=True)
                else:
                    print(f"<think>{body}</think>")

                history.append(
                    {"role": "assistant", "content": f"<think>{body}</think>"}
                )

            elif tag == "tool_call":
                if print_out:
                    print(f"<tool_call>{body}</tool_call>")

                # Execute tool(s) exactly as before ----------------------
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
                # DO NOT change step_think here
                break  # go back to model

            elif tag == "final_answer":
                if print_out:
                    print(f"<final_answer>{body}</final_answer>")
                return body

    if print_out:
        print("\nReached limit without final_answer.")
    return None


def save_to_csv(print_out: bool = False):
    output_csv_path = "./results.csv"
    file_exists = os.path.isfile(output_csv_path)

    with open(output_csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["qid", "question", "predicted_answer", "gt_answer"]
        )

        if not file_exists:
            writer.writeheader()

        for item in dataset:
            qid = item["qid"]
            question = item["question"]
            real = item["answer"]
            pred = run_query(question, 25, 50, print_out)

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
    args = Args()
    retriever = Retriever(args, graph)

    retriever.reset()
    save_to_csv(False)

    llm.stop()
