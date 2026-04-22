import time
import numpy as np
import pandas as pd
import networkx as nx
from greedy_hc_slgbn import hc_oneloop, hc_type_change
from synthetic_data.bn_5 import generate_bn_5, ground_truth_5
from synthetic_data.bn_8 import generate_bn_8, ground_truth_8
from synthetic_data.bn_insurance import generate_bn_insurance, ground_truth_insurance
from auxiliar import dag_to_cpdag, cpdag_to_networkx, dag_metrics, cpdag_metrics, skeleton_metrics, type_distance
from pybnesian import (SemiparametricBN, GreedyHillClimbing, 
                       OperatorPool, ChangeNodeTypeSet,
                       ValidatedLikelihood, LinearGaussianCPDType, ArcOperatorSet)

"""
Main experimental script of the paper.

This script runs the core experiments for structure learning and data fitting
across the three considered models (LG, SLGBN, and SPBN) on multiple datasets.

It includes:
- Fixed-structure experiments (parameter and type learning).
- Full structure learning experiments.
- Evaluation of learned graphs using multiple structural metrics (DAG, CPDAG, skeleton).
- Comparison of predicted node types.

All results are aggregated and saved into a CSV file for subsequent analysis
and visualization (used later to generate the figures in the paper).
"""

train_sizes = [200, 1000, 5000, 10000]
seeds = np.arange(7, 207, 10)
np.random.seed(123)
data_val8 = generate_bn_8(1000)
data_val5 = generate_bn_5(1000)
data_val_insu = generate_bn_insurance(1000)

# TYPE CHANGE

G5 = nx.DiGraph()
nodes5 = ['A', 'B', 'C', 'D', 'E']
G5.add_nodes_from(nodes5)
G5.add_edges_from([
    ('A', 'C'),
    ('B', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E')
])

G5_types = pd.read_csv("synthetic_data/node_types_5.txt", sep=",", skipinitialspace=True, header=None)
G5_types = dict(zip(G5_types[0], G5_types[1]))

cpdag_cl5 = dag_to_cpdag(G5)
cpdag_G5 = cpdag_to_networkx(cpdag_cl5)

G8 = nx.DiGraph()
nodes8 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
G8.add_nodes_from(nodes8)
G8.add_edges_from([
    ('A', 'D'),
    ('A', 'E'),
    ('B', 'D'),
    ('B', 'E'),
    ('C', 'D'),
    ('C', 'E'),
    ('D', 'E'),
    ('D', 'F'),
    ('E', 'F'),
    ('F', 'G'),
    ('F', 'H'),
    ('G', 'H'),
])

G8_types = pd.read_csv("synthetic_data/node_types_8.txt", sep=",", skipinitialspace=True, header=None)
G8_types = dict(zip(G8_types[0], G8_types[1]))

cpdag_cl8 = dag_to_cpdag(G8)
cpdag_G8 = cpdag_to_networkx(cpdag_cl8)

df_insu = pd.read_csv("synthetic_data/insurance.csv")
G_insu = nx.DiGraph()
G_insu.add_edges_from(zip(df_insu["from"], df_insu["to"]))

G_insu_types = pd.read_csv("synthetic_data/node_types_insurance.txt", sep=",", skipinitialspace=True, header=None)
G_insu_types = dict(zip(G_insu_types[0], G_insu_types[1]))

cpdag_cl_insu = dag_to_cpdag(G_insu)
cpdag_G_insu = cpdag_to_networkx(cpdag_cl_insu)

results_typechange = []

models_graph = [[G5, "G5", cpdag_G5, G5_types], [G8, "G8", cpdag_G8, G8_types], [G_insu, "G_insu", cpdag_G_insu, G_insu_types]]

model_types = [
    ("LG", False),
    ("SLGBN", "nonlinear"),
    ("SPBN", False)
]

# FIXED STRUCTURE

for graph, data_name, cpdag, types in models_graph:
    for model_label, type_change_value in model_types:
        for n in train_sizes:
            for seed in seeds:

                start = time.time()
            
                np.random.seed(seed)

                if data_name == "G5":

                    data_train = generate_bn_5(n)
                    data_val = data_val5

                elif data_name == "G8":

                    data_train = generate_bn_8(n)
                    data_val = data_val8

                elif data_name == "G_insu":

                    data_train = generate_bn_insurance(n)
                    data_val = data_val_insu

                if model_label == 'SPBN':

                    vl = ValidatedLikelihood(data_train, k=10, seed=0)
                    pool = OperatorPool([ChangeNodeTypeSet()])
                    node_types = [(name, LinearGaussianCPDType()) for name in data_train.columns]
                    bn0 = SemiparametricBN(list(data_train.columns), node_types)

                    if data_name == "G5":

                        bn0.add_arc("A", "C")
                        bn0.add_arc("B", "C")
                        bn0.add_arc("B", "D")
                        bn0.add_arc("C", "D")
                        bn0.add_arc("D", "E")

                    elif data_name == "G8":

                        bn0.add_arc("A", "D")
                        bn0.add_arc("A", "E")
                        bn0.add_arc("B", "D")
                        bn0.add_arc("B", "E")
                        bn0.add_arc("C", "D")
                        bn0.add_arc("C", "E")
                        bn0.add_arc("D", "E")
                        bn0.add_arc("D", "F")
                        bn0.add_arc("E", "F")
                        bn0.add_arc("F", "G")
                        bn0.add_arc("F", "H")
                        bn0.add_arc("G", "H")

                    elif data_name == "G_insu":

                        for parent, child in G_insu.edges():

                            bn0.add_arc(parent, child)
                    
                    hc = GreedyHillClimbing()

                    start_time = time.time()

                    final_model = hc.estimate(pool, vl, bn0, patience=5)

                    final_model.fit(data_train)

                    best_ll = final_model.slogl(data_val)

                    exec_time = time.time() - start_time

                    node_types = {
                                    node: (
                                        'nonlinear' if str(final_model.node_type(node)) == 'CKDEFactor'
                                        else 'linear' if str(final_model.node_type(node)) == 'LinearGaussianFactor'
                                        else None
                                    )
                                    for node in final_model.nodes()
                                }
                    
                    thd = type_distance(types, node_types)[0]

                    results_typechange.append({
                        "Data": data_name,
                        "Model": model_label,
                        "Type": "Fixed",
                        "train_size": n,
                        "seed": seed,
                        "execution_time": exec_time,
                        "best_ll": best_ll,
                        "node_types": node_types,
                        "edges": final_model.arcs(),
                        "THD": thd
                    })

                else:
                
                    start_time = time.time()
                    
                    G, node_types, params, best_ll = hc_type_change(
                        graph,
                        data_train,
                        data_val,
                        initial_type='linear',
                        type_change=type_change_value,
                        max_iter=1000,
                        tabu_size=5,
                        patience=5,
                        n_bases = 6,
                        penalization=0.5,
                        score='BIC',
                        cores=20
                    )
                    
                    exec_time = time.time() - start_time

                    thd = type_distance(types, node_types)[0]
                    
                    results_typechange.append({
                        "Data": data_name,
                        "Model": model_label,
                        "Type": "Fixed",
                        "train_size": n,
                        "seed": seed,
                        "execution_time": exec_time,
                        "best_ll": best_ll,
                        "node_types": node_types,
                        "edges": G.edges(),
                        "THD": thd
                    })

                end = time.time()

                print('Execution time: ', start-end)

# STRUCTURE LEARNING

for graph, data_name, cpdag, types in models_graph:
    for model_label, type_change_value in model_types:
        for n in train_sizes:
            for seed in seeds:

                start = time.time()

                np.random.seed(seed)

                if data_name == "G5":

                    data_train = generate_bn_5(n)
                    data_val = data_val5
                    ground_ll = ground_truth_5(data_val)

                elif data_name == "G8":

                    data_train = generate_bn_8(n)
                    data_val = data_val8
                    ground_ll = ground_truth_8(data_val)

                elif data_name == "G_insu":

                    data_train = generate_bn_insurance(n)
                    data_val = data_val_insu
                    ground_ll = ground_truth_insurance(data_val)

                if model_label == 'SPBN':
                    
                    hc = GreedyHillClimbing()
                    pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])
                    vl = ValidatedLikelihood(data_train, k=10, seed=0)
                    node_types = [(name, LinearGaussianCPDType()) for name in data_train.columns]
                    start_model = SemiparametricBN(list(data_train.columns), node_types)                   

                    start_time = time.time()

                    final_model = hc.estimate(pool, vl, start_model, patience=5)

                    final_model.fit(data_train)

                    best_ll = final_model.slogl(data_val)

                    exec_time = time.time() - start_time

                    node_types = {
                        node: (
                            'nonlinear' if str(final_model.node_type(node)) == 'CKDEFactor'
                            else 'linear' if str(final_model.node_type(node)) == 'LinearGaussianFactor'
                            else None
                        )
                        for node in final_model.nodes()
                    }

                    G_spbn = nx.DiGraph()
                    G_spbn.add_nodes_from(final_model.nodes())
                    G_spbn.add_edges_from(final_model.arcs())

                    metrics_dag_spbn = dag_metrics(graph, G_spbn)
                    metrics_cpdag_spbn = cpdag_metrics(cpdag, G_spbn)
                    skeleton_original = graph.to_undirected() 
                    skeleton_learned = G_spbn.to_undirected()
                    metrics_skeleton_spbn = skeleton_metrics(skeleton_original, skeleton_learned)
                    thd = type_distance(types, node_types)[0]

                    results_typechange.append({
                        "Data": data_name,
                        "Model": model_label,
                        "Type": "Free",
                        "train_size": n,
                        "seed": seed,
                        "execution_time": exec_time,
                        "best_ll": best_ll,
                        "ground_ll": ground_ll,
                        "dag_K": metrics_dag_spbn["K"],
                        "dag_Recall": metrics_dag_spbn["Recall"],
                        "dag_Precision": metrics_dag_spbn["Precision"],
                        "dag_Precision_gl": metrics_dag_spbn["GlobalPrecision"],
                        "dag_FN": metrics_dag_spbn["FN"],
                        "dag_FP": metrics_dag_spbn["FP"],
                        "dag_SHD": metrics_dag_spbn["SHD"],
                        "cpdag_K": metrics_cpdag_spbn["K"],
                        "cpdag_Recall": metrics_cpdag_spbn["Recall"],
                        "cpdag_Precision": metrics_cpdag_spbn["Precision"],
                        "cpdag_Precision_gl": metrics_cpdag_spbn["GlobalPrecision"],
                        "cpdag_FN": metrics_cpdag_spbn["FN"],
                        "cpdag_FP": metrics_cpdag_spbn["FP"],
                        "cpdag_SHD": metrics_cpdag_spbn["SHD"],
                        "skeleton_K": metrics_skeleton_spbn["K"],
                        "skeleton_Recall": metrics_skeleton_spbn["Recall"],
                        "skeleton_Precision": metrics_skeleton_spbn["Precision"],
                        "skeleton_Precision_gl": metrics_skeleton_spbn["GlobalPrecision"],
                        "skeleton_FN": metrics_skeleton_spbn["FN"],
                        "skeleton_FP": metrics_skeleton_spbn["FP"],
                        "skeleton_SHD": metrics_skeleton_spbn["SHD"],
                        "THD": thd,
                        "node_types": node_types,
                        "edges": final_model.arcs()
                    })

                else:
                
                    start_time = time.time()
                    
                    G, node_types, params, best_ll = hc_oneloop(
                        data_train,
                        data_val,
                        initial_type='linear',
                        type_change=type_change_value,
                        max_iter=1000,
                        tabu_size=5,
                        patience=5,
                        n_bases = 6,
                        penalization=0.5,
                        score='BIC',
                        cores=40
                    )

                    exec_time = time.time() - start_time

                    metrics_dag_slgbn = dag_metrics(graph, G)
                    metrics_cpdag_slgbn = cpdag_metrics(cpdag, G)
                    skeleton_original = graph.to_undirected() 
                    skeleton_learned = G.to_undirected()
                    metrics_skeleton_slgbn = skeleton_metrics(skeleton_original, skeleton_learned)
                    thd = type_distance(types, node_types)[0]

                    results_typechange.append({
                        "Data": data_name,
                        "Model": model_label,
                        "Type": "Free",
                        "train_size": n,
                        "seed": seed,
                        "execution_time": exec_time,
                        "best_ll": best_ll,
                        "ground_ll": ground_ll,
                        "dag_K": metrics_dag_slgbn["K"],
                        "dag_Recall": metrics_dag_slgbn["Recall"],
                        "dag_Precision": metrics_dag_slgbn["Precision"],
                        "dag_Precision_gl": metrics_dag_slgbn["GlobalPrecision"],
                        "dag_FN": metrics_dag_slgbn["FN"],
                        "dag_FP": metrics_dag_slgbn["FP"],
                        "dag_SHD": metrics_dag_slgbn["SHD"],
                        "cpdag_K": metrics_cpdag_slgbn["K"],
                        "cpdag_Recall": metrics_cpdag_slgbn["Recall"],
                        "cpdag_Precision": metrics_cpdag_slgbn["Precision"],
                        "cpdag_Precision_gl": metrics_cpdag_slgbn["GlobalPrecision"],
                        "cpdag_FN": metrics_cpdag_slgbn["FN"],
                        "cpdag_FP": metrics_cpdag_slgbn["FP"],
                        "cpdag_SHD": metrics_cpdag_slgbn["SHD"],
                        "skeleton_K": metrics_skeleton_slgbn["K"],
                        "skeleton_Recall": metrics_skeleton_slgbn["Recall"],
                        "skeleton_Precision": metrics_skeleton_slgbn["Precision"],
                        "skeleton_Precision_gl": metrics_skeleton_slgbn["GlobalPrecision"],
                        "skeleton_FN": metrics_skeleton_slgbn["FN"],
                        "skeleton_FP": metrics_skeleton_slgbn["FP"],
                        "skeleton_SHD": metrics_skeleton_slgbn["SHD"],
                        "THD": thd,
                        "node_types": node_types,
                        "edges": G.edges()
                    })
                
                end = time.time()

                print('Execution time: ', start-end)

# Convertimos a DataFrame
results_df = pd.DataFrame(results_typechange)

# Guardar a CSV para análisis posterior
results_df.to_csv("results/results_compro.csv", index=False)

print("Experiments finished.")