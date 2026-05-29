import networkx as nx
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag

"""
This module provides helper functions for:

1. Graph transformations:
   - Converting between different graph representations (NetworkX, causallearn DAG, CPDAG).
   - Preparing graph structures for evaluation.

2. Evaluation metrics:
   - Comparing true and predicted graphs.
   - Computing structural metrics such as SHD (Structural Hamming Distance),
     precision, recall, etc.
   - Supporting evaluation for DAGs, CPDAGs, and undirected skeletons.
   - Measuring node-type prediction accuracy.

"""


def dag_to_cpdag(dag_in):

    """
    Convert a NetworkX DAG into a CPDAG using causallearn.

    Steps:
    1. Create GraphNode objects (required by causallearn).
    2. Build a causallearn Dag object from these nodes.
    3. Add directed edges from the input DAG.
    4. Convert the DAG into its Markov equivalence class (CPDAG).

    Args:
        dag_in (nx.DiGraph): Input directed acyclic graph.

    Returns:
        cpdag: CPDAG object in causallearn format.
    """

    nodes = {n: GraphNode(n) for n in dag_in.nodes()}
    dag = Dag(list(nodes.values()))

    for u, v in dag_in.edges():
        dag.add_directed_edge(nodes[u], nodes[v])

    cpdag = dag2cpdag(dag)

    return cpdag


def cpdag_to_networkx(cpdag):

    """
    Convert a causallearn CPDAG into a NetworkX graph.

    Representation:
    - Uses an undirected NetworkX graph (nx.Graph)
    - Edge attributes encode orientation:
        - orientation: "directed" or "undirected"
        - direction: (u, v) if directed

    Logic:
    - Nodes are copied directly.
    - Edge endpoints (TAIL/ARROW) determine orientation.

    Args:
        cpdag: CPDAG from causallearn.

    Returns:
        nx.Graph: Graph with orientation metadata on edges.
    """

    G = nx.Graph()  

    for node in cpdag.get_nodes():
        G.add_node(node.get_name())

    for edge in cpdag.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()

        ep1 = edge.get_endpoint1().name
        ep2 = edge.get_endpoint2().name

        if ep1 == "TAIL" and ep2 == "ARROW":
            G.add_edge(node1, node2, orientation="directed", direction=(node1, node2))

        elif ep1 == "ARROW" and ep2 == "TAIL":
            G.add_edge(node1, node2, orientation="directed", direction=(node2, node1))

        elif ep1 == "TAIL" and ep2 == "TAIL":
            G.add_edge(node1, node2, orientation="undirected")

    return G


def type_distance(node_types_true, node_types_pred):

    """
    Compute Type Hamming Distance (THD) between true and predicted node types.

    Logic:
    - For each node:
        - If missing in prediction → error
        - If type mismatch → error
        - If correct → counted separately

    Args:
        node_types_true (dict): True node types {node: type}.
        node_types_pred (dict): Predicted node types {node: type}.

    Returns:
        tuple:
            thd (int): Number of mismatches.
            correct (int): Number of correctly predicted types.
    """

    thd = 0
    correct = 0

    for node in node_types_true:
        if node not in node_types_pred:
            thd += 1
        elif node_types_true[node] != node_types_pred[node]:
            thd += 1
        else:
            correct += 1

    return thd, correct


def final_metrics(skel_true, skel_pred, thd=0):
    """
    Compute skeleton metrics over edges only.

    Args:
        skel_true (nx.Graph): Ground truth skeleton.
        skel_pred (nx.Graph): Predicted skeleton.
        thd (int): Type Hamming Distance, kept for compatibility with callers.

    Returns:
        dict: SHD, Precision_total, Recall_total
    """

    edges_true = {frozenset(e) for e in skel_true.edges()}
    edges_pred = {frozenset(e) for e in skel_pred.edges()}

    fn = len(edges_true - edges_pred)
    fp = len(edges_pred - edges_true)
    shd = fn + fp

    k = len(edges_true & edges_pred)

    n_true = len(edges_true)
    n_pred = len(edges_pred)

    recall_total = (
        k / n_true
        if n_true > 0 else 0.0
    )

    precision_total = (
        k / n_pred
        if n_pred > 0 else 0.0
    )

    return {
        "SHD": shd,
        "Recall_total": recall_total,
        "Precision_total": precision_total
    }

