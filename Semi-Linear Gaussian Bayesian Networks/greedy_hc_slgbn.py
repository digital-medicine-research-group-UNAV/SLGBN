import networkx as nx
import numpy as np
from pygam import LinearGAM, s, te
from functools import reduce
import operator
from joblib import Parallel, delayed
from itertools import combinations
import time

"""
This module implements hill-climbing algorithms for learning probabilistic graphical models.

It includes two main approaches:
1. Full structure learning (hc_oneloop):
   - Simultaneously learns graph structure (edges) and node types (linear/nonlinear).

2. Fixed-structure learning with type optimization (hc_type_change):
   - Assumes a known graph structure and only optimizes node types.

The module also provides:
- Parameter estimation for linear and nonlinear models.
- Likelihood and score computation (LL, BIC, EBIC).
- Evaluation of candidate graph modifications (add/remove/reverse edges, type changes).

These methods are used as the core learning procedures in the experimental pipeline.
"""


def estimate_parameters(train_data, G, node_types, n_bases, penalization, params_prev=None, nodes_to_update=None, score_cache=None):

    """
    Estimate local conditional distributions for each node given its parents.

    Supports:
    - Linear Gaussian models
    - Nonlinear models using GAM (Generalized Additive Models)

    Key ideas:
    - Each node is modeled conditionally on its parents.
    - Linear: least squares regression.
    - Nonlinear: spline-based GAM.
    - Uses caching to avoid recomputing parameters.

    Args:
        train_data (pd.DataFrame): Training dataset.
        G (nx.DiGraph): Current graph structure.
        node_types (dict): Mapping node -> "linear" or "nonlinear".
        n_bases (int): Number of spline bases (for nonlinear).
        penalization (float): Regularization parameter for GAM.
        params_prev (dict): Previous parameters (for incremental updates).
        nodes_to_update (list): Nodes to recompute.
        score_cache (dict): Cache for parameter reuse.

    Returns:
        dict: Estimated parameters per node.
    """

    if params_prev is None:
        params = {}
    else:
        params = params_prev.copy()

    if nodes_to_update is None:
        nodes_to_update = list(G.nodes())

    for node in nodes_to_update:

        # Initialize dictionary for the node
        if node not in params:
            params[node] = {}

        # Parents
        parents = sorted(G.predecessors(node))
        node_type = node_types[node]

        key = (node, tuple(parents), node_type)

        subset = train_data

        y = subset[node].values

        if score_cache is not None and key in score_cache:
            params[node] = score_cache[key]["params"]
            continue

        if len(parents) == 0:

            if node_type in ["linear", "nonlinear"]:

                beta = np.array([np.mean(y)])
                sigma2 = np.var(y)

                params[node] = {
                    "type": node_type,
                    "beta": beta,
                    "sigma": sigma2,
                    "model": None
                }

            continue

        else:
        
            X = subset[parents].values

            if node_type == "linear":
                X_lin = np.column_stack([np.ones(X.shape[0]), X])
                beta = np.linalg.lstsq(X_lin, y, rcond=None)[0]
                sigma2 = np.sum((y - X_lin @ beta)**2) / (X_lin.shape[0]) # MLE
                # sigma2 = np.sum((y - X_lin @ beta)**2) / (X_lin.shape[0] - X_lin.shape[1]) # Corrected

                params[node] = {
                    "type": "linear",
                    "beta": beta,
                    "sigma": sigma2,
                    "model": None
                }

            elif node_type == "nonlinear":

                if n_bases is None:

                    terms = [s(i, spline_order=3, lam=penalization) for i in range(len(parents))]

                else:

                    terms = [s(i, n_splines=n_bases, spline_order=3, lam=penalization) for i in range(len(parents))]

                # if len(parents) > 1:
                #     interaction_terms = [te(i, j, n_splines=4) for i, j in combinations(range(len(parents)), 2)]
                #     terms.extend(interaction_terms)
                    
                terms = reduce(operator.add, terms)
                gam = LinearGAM(terms).fit(X, y)

                params[node] = {
                    "type": "nonlinear",
                    "beta": None,
                    "sigma": None,
                    "model": gam
                }

        if score_cache is not None:
            score_cache[key] = {
                "params": params[node]
            }

    return params


def compute_total_ll_bic(train_data, params, G, method = 'BIC', k_cache = None, ll_cache = None, nodes = None, spar = 0.75):

    """
    Compute total log-likelihood and penalized scores for the model.

    Supports:
    - LL (log-likelihood)
    - BIC
    - EBIC

    Key ideas:
    - Computes node-wise likelihood and aggregates.
    - Uses caches for efficiency (incremental updates).
    - Penalizes model complexity based on number of parameters.

    Args:
        train_data (pd.DataFrame): Dataset.
        params (dict): Node parameters.
        G (nx.DiGraph): Graph structure.
        method (str): Scoring method ("LL", "BIC", "EBIC").
        k_cache (dict): Cached parameter counts.
        ll_cache (dict): Cached likelihoods.
        nodes (list): Subset of nodes to recompute.
        spar (float): EBIC sparsity parameter.

    Returns:
        tuple: (score, updated_ll_cache, updated_k_cache)
    """

    total_ll = 0.0
    total_params = 0
    n_samples = len(train_data)

    if ll_cache is None:
        ll_cache_temp = {}
    
    else:
        ll_cache_temp = ll_cache.copy()

    if k_cache is None:
        k_cache_temp = {}

    else:
        k_cache_temp = k_cache.copy()

    if nodes is None:

        nodes_to_update = params.keys()

    else:
        
        # print(f"Computing LL for nodes: {nodes}")
        nodes_to_update = set(nodes)

    for node, node_params in params.items():

        # if nodes is None:
            # print(f"Computing LL for node: {node}")

        if node not in nodes_to_update:
            continue

        # Parents (sorted for consistent ordering)
        parents = sorted(G.predecessors(node))

        p = node_params

        node_ll = 0.0

        if p["beta"] is not None:
            # Caso lineal o no-lineal sin padres
            y_obs = train_data[node].values
            beta = p["beta"]
            sigma2 = p["sigma"]

            if len(parents) > 0:
                X = train_data[parents].values
                X = np.column_stack([np.ones(X.shape[0]), X])  # add intercept
            else:
                X = np.ones((len(y_obs), 1))

            mu = X @ beta  # vectorized dot product
            ll = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * ((y_obs - mu)**2) / sigma2
            node_ll += ll.sum()
            k_node = len(parents) + 2

        else:           
            if p["type"] == "nonlinear":

                gam = p["model"]
                edf = gam.statistics_['edof']
                k_node = gam.statistics_['edof'] + 1
                X_train = train_data[parents].values
                y_train = train_data[node].values
                mu_train = gam.predict(X_train)
                rss = np.sum((y_train - mu_train)**2)
                sigma2 = rss / (len(y_train)) # MLE variance
                # sigma2 = rss / (len(y_train) - edf)  # corrected variance
                mu_test = gam.predict(train_data[parents].values)
                y_test = train_data[node].values
                ll_test = -0.5*np.log(2*np.pi*sigma2) - 0.5*((y_test - mu_test)**2)/sigma2
                node_ll = ll_test.sum()

        ll_cache_temp[node] = node_ll
        k_cache_temp[node] = k_node

        # if nodes is None:
            # print(f" Node {node}: LL = {node_ll}")
    
    total_ll = sum(ll_cache_temp.values())
    total_params = sum(k_cache_temp.values())

    if method == 'LL':
        
        return total_ll, ll_cache_temp, k_cache_temp
    
    elif method == 'EBIC':

        total_ll -= 0.5 * total_params * np.log(n_samples) + 2 * spar * len(G.edges) * np.log(len(G.nodes))

        return total_ll, ll_cache_temp, k_cache_temp

    elif method == 'BIC':

        total_ll -= 0.5 * total_params * np.log(n_samples)

        return total_ll, ll_cache_temp, k_cache_temp
    
    else:

        print('PLEASE SELECT A VALID SCORE')


def evaluate_event(event, G, node_types, params, n_bases, penalization, data_train, k_nodos, ll_nodos, score, score_cache):
    
    """
    Evaluate a candidate modification (event) to the current model.

    Supported events:
    - change_type: change node type (linear ↔ nonlinear)
    - add_edge: add directed edge
    - remove_edge: remove edge
    - reverse_edge: reverse edge direction

    Key ideas:
    - Applies the modification to a copy of the graph.
    - Re-estimates only affected nodes.
    - Computes updated score.
    - Returns all necessary information for selection.

    Returns:
        tuple: (score, event, updated_types, updated_params, k_cache, ll_cache, tabu_move, updated_graph)
    """

    tipo = event[0]
    
    if tipo == "change_type":
        G_copy = G.copy()
        node_types_copy = node_types.copy()
        node = event[1]
        new_type = event[2]
        node_types_copy[node] = new_type
        params_prov = estimate_parameters(data_train, G_copy, node_types_copy, n_bases, penalization, params, [node], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[node])
        tabu = ("change_type", node)
        return ll, event, node_types_copy, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy
    
    elif tipo == "add_edge":
        node, target = event[1], event[2]
        if nx.has_path(G, target, node):
            return -1e10, None, node_types, params, k_nodos, ll_nodos, None, G
        G_copy = G.copy()
        G_copy.add_edge(node, target)
        # if not nx.is_directed_acyclic_graph(G_copy):
            # return -1e10, None, node_types, params, ll_nodos
        params_prov = estimate_parameters(data_train, G_copy, node_types, n_bases, penalization, params, [target], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[target])
        tabu = ("remove_edge", node, target)
        return ll, event, node_types, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy
    
    elif tipo == "remove_edge":
        G_copy = G.copy()
        node, target = event[1], event[2]
        G_copy.remove_edge(node, target)
        params_prov = estimate_parameters(data_train, G_copy, node_types, n_bases, penalization, params, [target], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[target])
        tabu = ("add_edge", node, target)
        return ll, event, node_types, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy
    
    elif tipo == "reverse_edge":
        G_copy = G.copy()
        node, target = event[1], event[2]
        G_copy.remove_edge(node, target)
        if nx.has_path(G_copy, node, target):
            return -1e10, None, node_types, params, k_nodos, ll_nodos, None, G
        G_copy.add_edge(target, node)
        # if not nx.is_directed_acyclic_graph(G_copy):
            # return -1e10, None, node_types, params, ll_nodos
        params_prov = estimate_parameters(data_train, G_copy, node_types, n_bases, penalization, params, [node, target], score_cache=score_cache)
        ll, ll_nodos_prov, k_nodos_prov = compute_total_ll_bic(data_train, params_prov, G_copy, method=score, k_cache=k_nodos, ll_cache=ll_nodos, nodes=[node, target])
        tabu = ("reverse_edge", target, node)
        return ll, event, node_types, params_prov, k_nodos_prov, ll_nodos_prov, tabu, G_copy


def can_change_type(node, tabu, type_change, n_parents):

    """
    Check whether a node type change is allowed.

    Conditions:
    - Not in tabu list.
    - Type change is enabled.
    - Nonlinear type requires at least one parent.
    """

    can_try_type_change = True

    if ("change_type", node) in tabu:

        can_try_type_change = False

    if type_change == False:
                    
        can_try_type_change = False

    if type_change == 'nonlinear' and n_parents == 0:

        can_try_type_change = False

    return can_try_type_change


def can_add_edge(node, target, G, tabu, max_parents):

    """
    Check whether adding an edge is valid.

    Conditions:
    - Edge does not already exist.
    - Not forbidden by tabu list.
    - Target does not exceed max number of parents.
    - Does not introduce cycles.
    """

    if G.has_edge(node, target): 
                    
        return False
                    
    if ("add_edge", node, target) in tabu:
            
        return False

    if G.in_degree(target) >= max_parents:

        return False    

    # if not nx.is_directed_acyclic_graph(G_copy):
    if nx.has_path(G, target, node):

        return False 

    return True


def can_remove_edge(node, target, G, tabu):

    """
    Check whether removing an edge is allowed.

    Conditions:
    - Edge exists.
    - Not forbidden by tabu list.
    """

    if not G.has_edge(node, target): 
                    
        return False
                    
    if ("remove_edge", node, target) in tabu:
            
        return False 

    return True


def can_reverse_edge(node, target, G, tabu, max_parents):

    """
    Check whether reversing an edge is valid.

    Conditions:
    - Edge exists.
    - Not in tabu list.
    - Parent constraints satisfied.
    - Does not introduce cycles.
    """

    if not G.has_edge(node, target): 
                    
        return False
      
    if ("reverse_edge", node, target) in tabu:
            
        return False

    if G.in_degree(node) >= max_parents:

        return False    

    G_copy = G.copy()
    
    G_copy.remove_edge(node, target)

    if nx.has_path(G_copy, node, target):

        return False 

    # if not nx.is_directed_acyclic_graph(G_copy):

    #     can_try_reverse_edge = False 

    return True


def hc_oneloop(data_train, data_val, initial_type='linear', type_change=False, max_iter = 1000, tabu_size = 5, patience = 5, n_bases = 6, penalization = 0.5, score = 'BIC', cores = 1):

    """
    Full hill-climbing algorithm for structure and type learning.

    Learns:
    - Graph structure (edges)
    - Node types (linear/nonlinear)

    Key features:
    - Tabu search to avoid cycling
    - Parallel evaluation of candidate moves
    - Early stopping via patience
    - Incremental parameter updates

    Process:
    1. Start from empty graph.
    2. Generate all valid candidate moves.
    3. Evaluate them in parallel.
    4. Apply the best move.
    5. Repeat until convergence.

    Returns:
        tuple: (best_graph, best_node_types, best_params, final_log_likelihood)
    """

    nodes = list(data_train.columns)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    iteration = 0
    max_iter = max_iter
    count = 0
    tabu_moves = []
    tabu_size = tabu_size
    score_cache = {}
    max_parents = 5
    n_bases = n_bases
    penalization = penalization
    min_delta = 0
    
    node_types = {node: initial_type for node in nodes}

    params = estimate_parameters(data_train, G, node_types, n_bases, penalization, score_cache=score_cache)

    total_ll, ll_nodos, k_nodos = compute_total_ll_bic(data_train, params, G, method = score)

    best_G = G.copy()
    best_node_types = node_types.copy()
    best_params = params.copy()

    best_ll = total_ll

    while iteration < max_iter:

        best_event = None
        best_score = -1e10

        events_to_try = []

        for i, node in enumerate(nodes):

            n_parents = G.in_degree(node)

            if can_change_type(node, tabu_moves, type_change, n_parents):

                current_type = node_types[node]

                new_type = type_change if current_type == 'linear' else 'linear'

                events_to_try.append(("change_type", node, new_type))


            for j, target in enumerate(nodes):

                if i == j:
                    continue

                if can_add_edge(node, target, G, tabu_moves, max_parents):
                    events_to_try.append(("add_edge", node, target))

                if can_remove_edge(node, target, G, tabu_moves):
                    events_to_try.append(("remove_edge", node, target))

                if can_reverse_edge(node, target, G, tabu_moves, max_parents):
                    events_to_try.append(("reverse_edge", node, target))

        results = Parallel(n_jobs=cores, prefer="threads")( 
            delayed(evaluate_event)(event, G, node_types, params, n_bases, penalization, data_train, k_nodos, ll_nodos, score, score_cache)
            for event in events_to_try
        )

        best_result = max(results, key=lambda x: x[0])

        ll_select, best_event, node_types_select, params_select, k_nodos_select, ll_nodos_select, tabu, G_prov = best_result

        
        tabu_moves.append(tabu)

        if len(tabu_moves) > tabu_size:
            tabu_moves.pop(0)


        params = params_select
        G = G_prov.copy()
        ll_nodos = ll_nodos_select
        k_nodos = k_nodos_select
        node_types = node_types_select

        iteration += 1

        if ll_select > best_ll:
            count = 0
            best_params = params
            best_node_types = node_types_select
            best_G = G.copy()
            best_ll = ll_select

        else:
            count += 1

        if best_event[0] == 'change_type':

            print(f"Iteration {iteration}. {best_event[0]}, {best_event[1]}. LL = {ll_select}. Count = {count}")

        else:

            print(f"Iteration {iteration}. {best_event[0]}, {best_event[1]}, {best_event[2]}: LL = {ll_select}. Count = {count}")

        if count >= patience:


            final_G = best_G
            final_node_types = best_node_types
            final_params = best_params

            # La verosimilitud final se calcula sin penalización BIC.

            final_ll, ll_nodos, k_nodos = compute_total_ll_bic(data_val, final_params, final_G, method = 'LL')

            print("No improvement for 5 iterations. Returning best network found. Total LL (no BIC) = ", final_ll)

            break

    return final_G, final_node_types, final_params, final_ll




def hc_type_change(G, data_train, data_val, initial_type='linear', type_change=False, max_iter = 1000, tabu_size = 5, patience = 5, n_bases = 6, penalization = 0.5, score = 'BIC', cores = 1):

    """
    Hill-climbing algorithm for node type optimization with fixed structure.

    Learns:
    - Node types only (structure is fixed)

    Key features:
    - Same optimization strategy as hc_oneloop
    - Restricted to type-change moves
    - Efficient for scenarios with known graph structure

    Returns:
        tuple: (best_graph, best_node_types, best_params, final_log_likelihood)
    """

    nodes = list(data_train.columns)

    G_0 = G.copy()

    iteration = 0
    max_iter = max_iter
    count = 0
    tabu_moves = []
    tabu_size = tabu_size
    n_bases = n_bases
    penalization = penalization
    score_cache = {}
    
    node_types = {node: initial_type for node in nodes}

    params = estimate_parameters(data_train, G_0, node_types, n_bases, penalization, score_cache)

    total_ll, ll_nodos, k_nodos = compute_total_ll_bic(data_train, params, G_0, method = score)

    best_G = G_0.copy()
    best_node_types = node_types.copy()
    best_params = params.copy()
    best_ll = total_ll

    while iteration < max_iter:

        best_event = None

        events_to_try = []

        for i, node in enumerate(nodes):

            n_parents = G_0.in_degree(node)

            if can_change_type(node, tabu_moves, type_change, n_parents):

                current_type = node_types[node]

                new_type = type_change if current_type == 'linear' else 'linear'

                events_to_try.append(("change_type", node, new_type))

        if not events_to_try:

            print("No events to try. Stopping algorithm.")
            
            final_G = best_G
            final_node_types = best_node_types
            final_params = best_params
            
            final_ll, ll_nodos, k_nodos = compute_total_ll_bic(data_val, final_params, final_G, method='LL')
    
            break

        results = Parallel(n_jobs=cores)( 
            delayed(evaluate_event)(event, G_0, node_types, params, n_bases, penalization, data_train, k_nodos, ll_nodos, score, score_cache)
            for event in events_to_try
        )

        best_result = max(results, key=lambda x: x[0])

        ll_select, best_event, node_types_select, params_select, k_nodos_select, ll_nodos_select, tabu, G_prov = best_result

        
        tabu_moves.append(tabu)

        if len(tabu_moves) > tabu_size:
            tabu_moves.pop(0)


        params = params_select
        G_0 = G_prov.copy()
        ll_nodos = ll_nodos_select
        k_nodos = k_nodos_select
        node_types = node_types_select

        iteration += 1

        if ll_select > best_ll:
            count = 0
            best_params = params
            best_node_types = node_types_select
            best_G = G_0.copy()
            best_ll = ll_select

        else:
            count += 1

        if best_event[0] == 'change_type':
            print(f"Iteration {iteration}. {best_event[0]}, {best_event[1]}. LL = {ll_select}. Count = {count}")
        else:
            print(f"Iteration {iteration}. {best_event[0]}, {best_event[1]}, {best_event[2]}: LL = {ll_select}. Count = {count}")

        if count >= patience:


            final_G = best_G
            final_node_types = best_node_types
            final_params = best_params

            final_ll, ll_nodos, k_nodos = compute_total_ll_bic(data_val, final_params, final_G, method = 'LL')

            print("No improvement for 5 iterations. Returning best network found. Total LL (no BIC) = ", final_ll)

            break

    return final_G, final_node_types, final_params, final_ll