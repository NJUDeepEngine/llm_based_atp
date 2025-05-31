from graphviz import Digraph
from collections import defaultdict
from typing import List, Optional

from config import settings
from search.search_tree import Node, SolvedNode, InvalidNode, UnsolvedNode,Edge

def split_text(text, max_len=30):
    max_total_len = 3 * max_len
    if len(text) > max_total_len:
        text = text[:max_total_len - 3] + "..."
    lines = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    return "\n".join(lines)

def visualize_proof_tree(
        node_list: List[Node],
        success_edge_list: Optional[List['Edge']] = None,
        backedge_list: Optional[List['Edge']] = None,
        dir: str = settings.RESULT_SAVE_PATH + "/visual",
        filename: str = "proof_search_tree",
        mode: List[str] = ["simple", "detail"],
    ):
    def _draw(filename: str, simple: bool):
        dot = Digraph(comment="proof_search_tree")
        dot.attr(rankdir="TB", size="8")
        depth_to_nodes = defaultdict(list)

        for node in node_list:
            label = ""
            color = "black"
            shape = "ellipse"
            fillcolor = "white"
            style = "solid"
            penwidth = 1
            fontsize = 10
            depth = node.depth

            if isinstance(node, UnsolvedNode):
                if simple:
                    label = f"id={node.id}"
                else:
                    label = f"Internal\nid={node.id}\nscore={node.priority:.2f}"
                shape = "box"
            elif isinstance(node, SolvedNode):
                label = f"✓Proof\nid={node.id}"
                color = "green"
                fillcolor = "green"
                style = "filled"
                penwidth = 3
                fontsize = 14
            elif isinstance(node, InvalidNode):
                label = f"✗LeanError"
                color = "red"
                depth = node.depth

            dot.node(
                str(node.id), 
                label=label, 
                color=color, 
                shape=shape,
                penwidth=str(penwidth), 
                fontsize=str(fontsize),
                fillcolor=fillcolor,
                style=style
            )
            depth_to_nodes[depth].append(str(node.id))

        for depth, node_ids in depth_to_nodes.items():
            with dot.subgraph() as s:
                s.attr(rank='same')
                for nid in node_ids:
                    s.node(nid)

        for node in node_list:
            if isinstance(node, UnsolvedNode) and node.out_edges:
                for edge in node.out_edges:
                    if edge in success_edge_list:
                        edge_color = "green"
                        edge_style = "solid"
                        penwidth = "2"
                    elif edge in backedge_list:
                        edge_color = "purple"
                        edge_style = "dashed"
                        penwidth = "2"
                    else:
                        edge_color = "black"
                        edge_style = "solid"
                        penwidth = "1"
                    if simple:
                        edge_tactic=""
                    else:
                        edge_tactic = split_text(edge.tactic)
                    dot.edge(
                        str(edge.src.id), 
                        str(edge.dst.id),
                        label=f"{edge_tactic}\n{edge.score:.2f}",
                        color=edge_color, 
                        style=edge_style, 
                        penwidth=penwidth
                    )
        
        dot.render(filename, dir, format='pdf', cleanup=True)
        print(f"✅ Visualization saved as {filename}.pdf")

    if "simple" in mode:
        _draw(filename + "_simple", True)
    if "detail" in mode:
        _draw(filename + "_detail", False)
