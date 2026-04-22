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


def dag_metrics(dag_true, dag_pred):

    """
    Compute structural metrics between two DAGs.

    Metrics include:
    - K: number of correctly predicted edges
    - Recall and Precision
    - FN / FP (missing and extra edges)
    - SHD (Structural Hamming Distance)
    - TN (true negatives)
    - GlobalPrecision (accuracy over all possible edges)

    Special handling:
    - Edge reversals count as a single error in SHD

    Args:
        dag_true (nx.DiGraph): Ground truth DAG.
        dag_pred (nx.DiGraph): Predicted DAG.

    Returns:
        dict: Dictionary with all computed metrics.
    """

    edges_true = set(dag_true.edges())
    edges_pred = set(dag_pred.edges())

    correct = edges_true & edges_pred

    missing = edges_true - edges_pred

    extra = edges_pred - edges_true

    reversals = {
        (u, v) for (u, v) in missing
        if (v, u) in edges_pred
    }

    n_reversals = len(reversals)

    fn = len(missing) - n_reversals
    fp = len(extra) - n_reversals

    shd = fn + fp + n_reversals

    n_true = len(edges_true)
    n_pred = len(edges_pred)
    k = len(correct)

    recall = k / n_true if n_true > 0 else 0.0
    precision = k / n_pred if n_pred > 0 else 0.0

    nodes = dag_true.nodes()
    n = len(nodes)
    total_possible = n * (n - 1)

    tp = k
    fn_total = len(missing)
    fp_total = len(extra)
    tn = total_possible - tp - fn_total - fp_total

    global_precision = (
        (tp + tn) / total_possible if total_possible > 0 else 0.0
    )

    return {
        "K": k,
        "Recall": recall,
        "Precision": precision,
        "FN": len(missing),
        "FP": len(extra),
        "SHD": shd,
        "TN": tn,
        "GlobalPrecision": global_precision
    }


def cpdag_metrics(cpdag: nx.Graph, dag: nx.DiGraph) -> int:

    """
    Compute structural metrics comparing a CPDAG with a true DAG.

    Key idea:
    - Directed edges must match orientation.
    - Undirected edges are considered correct if any direction exists in the DAG.

    Metrics:
    - SHD, Precision, Recall
    - FN / FP counts
    - GlobalPrecision based on undirected edge space

    Args:
        cpdag (nx.Graph): CPDAG with orientation metadata.
        dag (nx.DiGraph): Ground truth DAG.

    Returns:
        dict: Dictionary with evaluation metrics.
    """

    shd = 0
    correct_edges = 0
    missing_edges = 0
    extra_edges = 0

    dag_edges = set(dag.edges())

    for u, v, data in cpdag.edges(data=True):

        orientation = data.get("orientation")

        if orientation == "directed":

            correct_u, correct_v = data["direction"]

            if dag.has_edge(correct_u, correct_v):
                correct_edges += 1
                continue

            elif dag.has_edge(correct_v, correct_u):
                shd += 1
                missing_edges += 1
                extra_edges += 1

            else:
                shd += 1
                missing_edges += 1

        elif orientation == "undirected":

            if dag.has_edge(u, v) or dag.has_edge(v, u):
                correct_edges += 1
            else:
                shd += 1
                missing_edges += 1

    for u, v in dag_edges:

        if not cpdag.has_edge(u, v):
            shd += 1
            extra_edges += 1
            continue

        data = cpdag.get_edge_data(u, v)
        orientation = data.get("orientation")

        if orientation == "undirected":
            continue

        if orientation == "directed":
            correct_u, correct_v = data["direction"]

            if (u, v) != (correct_u, correct_v):
                continue

    total_cpdag_edges = cpdag.number_of_edges()

    recall = (
        correct_edges / total_cpdag_edges
        if total_cpdag_edges > 0 else 0.0
    )

    total_predicted = correct_edges + extra_edges

    precision = (
        correct_edges / total_predicted
        if total_predicted > 0 else 0.0
    )

    edges_true_und = {frozenset(e) for e in dag.edges()}
    edges_pred_und = {frozenset((u, v)) for u, v in cpdag.edges()}

    nodes = dag.nodes()
    n = len(nodes)
    total_possible = n * (n - 1) // 2

    tp = len(edges_true_und & edges_pred_und)
    fn = len(edges_true_und - edges_pred_und)
    fp = len(edges_pred_und - edges_true_und)
    tn = total_possible - tp - fn - fp

    global_precision = (
        (tp + tn) / total_possible if total_possible > 0 else 0.0
    )

    
    return {
        "K": correct_edges,
        "Recall": recall,
        "Precision": precision,
        "FN": missing_edges,
        "FP": extra_edges,
        "SHD": shd,
        "TN": tn,
        "GlobalPrecision": global_precision
    }



def skeleton_metrics(skel_true, skel_pred):

    """
    Compute metrics for undirected graph skeletons.

    Approach:
    - Ignore edge direction.
    - Compare edges as unordered pairs (frozenset).

    Metrics:
    - K (correct edges)
    - Recall / Precision
    - FN / FP
    - SHD (since no direction, SHD = FN + FP)
    - GlobalPrecision

    Args:
        skel_true (nx.Graph): True skeleton.
        skel_pred (nx.Graph): Predicted skeleton.

    Returns:
        dict: Dictionary with evaluation metrics.
    """

    edges_true = {frozenset(e) for e in skel_true.edges()}
    edges_pred = {frozenset(e) for e in skel_pred.edges()}

    k = len(edges_true & edges_pred)

    fn = len(edges_true - edges_pred)
    fp = len(edges_pred - edges_true)

    shd = fn + fp

    n_true = len(edges_true)
    n_pred = len(edges_pred)

    pct1 = k / n_true if n_true > 0 else 0.0
    pct2 = k / n_pred if n_pred > 0 else 0.0

    nodes = skel_true.nodes()
    n = len(nodes)
    total_possible = n * (n - 1) // 2

    tp = k
    tn = total_possible - tp - fn - fp

    global_precision = (
        (tp + tn) / total_possible if total_possible > 0 else 0.0
    )

    return {
        "K": k,
        "Recall": pct1,
        "Precision": pct2,
        "FN": fn,
        "FP": fp,
        "SHD": shd,
        "TN": tn,
        "GlobalPrecision": global_precision
    }


