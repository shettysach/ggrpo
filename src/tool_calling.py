from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import Retriever, NODE_TEXT_KEYS
from graph_funcs import graph_funcs
import json
import re


# ----------------------------
# Model setup
# ----------------------------
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="cuda"
)


# ----------------------------
# Tool definitions
# ----------------------------

# Load graph and initialize tools
graph_path = "/home/sword/Desktop/work/ggrpo/data/biomedical/graph.json"
graph = json.load(open(graph_path))


class Args:
    dataset = "biomedical"
    faiss_gpu = False
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
    """Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
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

{example_block}

Now solve this.

"""


formatted_prompt = system_prompt.format(
    graph_description=graph_description,
    example_block=example_block,
)

pattern = re.compile(r"<(think|tool_call|tool_response|final_answer)>(.*?)</\1>", re.S)


def run_query(query, max_steps=10):
    history = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": f"Question: {query}"},
    ]

    for step in range(max_steps):
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
                print(f"<think>{body}</think>")
                history.append(
                    {"role": "assistant", "content": f"<think>{body}</think>"}
                )

            elif tag == "tool_call":
                print(f"<tool_call>{body}</tool_call>")
                calls = re.findall(r"(\w+\[.*?\])", body)
                full_tool_call = []
                full_tool_response = []

                for call in calls:
                    tool_match = re.match(r"(\w+)\[(.*?)\]", call)
                    if not tool_match:
                        print(f"Invalid tool call: {call}")
                        continue

                    tool, args = tool_match.groups()
                    args = [eval(x) for x in args.split(",")] if args.strip() else []

                    try:
                        result = TOOLS[tool](*args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    print(f"<tool_response>{result}</tool_response>")
                    full_tool_call.append(f"<tool_call>{call}</tool_call>")
                    full_tool_response.append(
                        f"<tool_response>{result}</tool_response>"
                    )

                history.extend(
                    {"role": "assistant", "content": call} for call in full_tool_call
                )
                history.extend(
                    {"role": "tool", "content": resp} for resp in full_tool_response
                )
                break  # Return to LLM for next step

            elif tag == "final_answer":
                print(f"FINAL ANSWER: {body}")
                return body

            else:
                print(f"<{tag}>{body}</{tag}>")

    print("Reached step limit without final answer.")
    return None


data_path = "/home/sword/Desktop/work/ggrpo/data/biomedical/graph.json"
data = json.load(open(graph_path))


if __name__ == "__main__":
    run_query("what are the side effects of compound Pyridoxine?")
