from retriever import Retriever, NODE_TEXT_KEYS
from graph_funcs import graph_funcs
import json

graph_path = "/home/sword/Desktop/work/ggrpo/data/biomedical/graph.json"
graph = json.load(open(graph_path))


class Args:
    dataset = "biomedical"
    faiss_gpu = False
    embedder_name = "sentence-transformers/all-mpnet-base-v2"
    embed_cache = True
    embed_cache_dir = "/tmp"
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


if __name__ == "__main__":
    print(TOOLS["RetrieveNode"]("Caffeine"))
    print(TOOLS["NodeFeature"]("DB00200", "inchikey"))
    print(TOOLS["NodeDegree"]("DB00201", "Compound-causes-Side Effect"))
    print(TOOLS["NeighbourCheck"]("DOID:8778", "Compound-treats-Disease"))
