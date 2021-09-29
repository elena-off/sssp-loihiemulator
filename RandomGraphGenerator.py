import matplotlib.pyplot as plt
import numpy as np
import random
import time
import networkx, random
import math


# random seed from current time
random.seed(time.time())


#generate a random graph dictionary given number of nodes, probability of connection between the nodes and maximal cost
def random_graph_dict(num_nodes,prob_connection,max_cost):
    graphinfo = {}
    graphinfo["num_nodes"] = num_nodes
    #generate random graph via networkx
    random_graph = networkx.binomial_graph(num_nodes, prob_connection)

    edges = []
    costs_edges = []
    for edge in random_graph.edges():
        edges.append(edge)
        #assign random costs to edges
        costs_edges.append(random.randint(1,max_cost))

    graphinfo["edge_tuples"] = edges
    graphinfo["costs_edges"] = np.array(costs_edges,dtype = np.uint8)
    #calculate number of necessary to display sum of all costs as binary number
    cost_sum = np.sum(costs_edges)
    try: 
        graphinfo["num_bits"] = int(math.ceil(math.log(cost_sum,2)))
    except: 
        return
    return graphinfo

#calculate SSSP solution using Dijkstra given an adjacency matrix of a graph
def dijkstra(adjacency_matrix):
    #all distances at beginning are infinity except distance to source node, which is 0
    distances = [float("inf") for _ in range(len(adjacency_matrix))]
    distances[0] = 0

    # set of visited nodes, False if node was not yet visited
    visited = [False for _ in range(len(adjacency_matrix))]

    # While not all nodes have been visited yet:
    while True:

        #find note with the shortest distance to start node from set of unvisited nodes
        shortest_distance = float("inf")
        shortest_index = -1
        for i in range(len(adjacency_matrix)):
            if distances[i] < shortest_distance and not visited[i]:
                shortest_distance = distances[i]
                shortest_index = i

        if shortest_index == -1:
            # if all nodes have been visited, the algorithm is finished
            return distances

        # for all neighboring nodes that haven't been visited yet
        for i in range(len(adjacency_matrix[shortest_index])):
            #replace current path with shorter path (if there is one) 
            if adjacency_matrix[shortest_index][i] != 0 and distances[i] > distances[shortest_index] + adjacency_matrix[shortest_index][i]:
                distances[i] = distances[shortest_index] + adjacency_matrix[shortest_index][i]

        visited[shortest_index] = True

#list of graph dicts
graph_dicts = []

#create 10 random graphs with random number of nodes between 2 and 10, probabilty of connection between 0.5 and 1 
#and max cost of 10 for each edge
for i in np.arange(10):
    num_nodes = random.randint(2,10)
    prob_connection = random.uniform(0.5,1)
    max_cost = 10
    graphinfo = random_graph_dict(num_nodes,prob_connection,max_cost)
    if graphinfo:
        graph_dicts.append(graphinfo)

# make dijkstra distance dict with solutions from dijkstra
dijkstra_distances = []
i = 0
for graphinfo in graph_dicts:
    edge_tuples = graphinfo["edge_tuples"]
    costs_edges = graphinfo["costs_edges"]
    num_nodes = graphinfo["num_nodes"]
    #calculate adjacency matrix from graph dictionary
    adjacency_matrix = np.zeros((num_nodes,num_nodes))
    for ((nodeA,nodeB),cost) in zip(edge_tuples,costs_edges):
        adjacency_matrix[nodeA][nodeB] = cost

   #if dijkstra solution exists, append it to dijkstra list; else, delete graph dict because it will not have a solution either
    try: 
        dijkstra_distances.append([int(x) for x in dijkstra(adjacency_matrix)])
    except: 
        graph_dicts.remove(graphinfo)

#write graph dicts and dijkstra solutions into file, to input into calculation network and compare results
f = open("ExampleDicts.txt", "a")
f.write(str(graph_dicts))
f.write(str(dijkstra_distances))
f.close()
