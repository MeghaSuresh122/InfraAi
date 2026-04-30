from langgraph.graph import END, START, StateGraph

from infra_ai.nodes.infra_nodes import infra_builder_node, infra_validator_node
from infra_ai.state import InfraGraphState


def build_infra_subgraph():
    """Modular builder then deterministic validator (extend with more nodes later)."""
    g = StateGraph(InfraGraphState)
    g.add_node("infra_builder", infra_builder_node)
    g.add_node("infra_validator", infra_validator_node)
    g.add_edge(START, "infra_builder")
    g.add_edge("infra_builder", "infra_validator")
    g.add_edge("infra_validator", END)
    return g.compile()
