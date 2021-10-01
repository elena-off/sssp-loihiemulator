from EmulatorNetwork import PathFinding
from time import *
import numpy as np

#example graph as input
graph_info = {"num_nodes": 4, "num_bits":4, "edge_tuples":  [(0,1),(0,2),(0,3),(1,3),(2,3)], "costs_edges": np.array([1, 1, 5, 3, 2],dtype=np.uint8) }
network = PathFinding(graph_info)

#measure time necessary to simulate network
start = time()
#calculate number of timesteps necessary to finish calculation via num_nodes*t_cyc
sim_timesteps = graph_info["num_nodes"]*(4*graph_info["num_bits"]+11)
#simulate network
network.net.run(sim_timesteps)
end = time()
totaltime = end-start
print("simulation time: " + str(totaltime))


#calculate distances of shortest paths from source node to each node using outputs from corresponding minimum components
network_distances = []
for i in range(len(network.output_monitors)):
    times, index= np.unique(network.output_monitors[i].t,return_index=True)
    if times.size == 0:
        network_distances.append(0)
    valuelist = np.split(network.output_monitors[i].i,index)
    k = -1
    for values in valuelist:
        if values.size >0:
            sum = 0
            for value in values:
                sum += 2**value
            #print values for last simulated timestep; all other outputs can be printed here with print(value), if wanted
            if times[k] == (sim_timesteps - 1):
                network_distances.append(sum)
        k+=1
print("Result of shortest paths calculated by network: " + str(network_distances))
