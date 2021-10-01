# SSSP Loihi Emulator
Solution for the SSSP as proposed by Aimone et al, implemented on the Brian2Loihi Emulator (https://github.com/sagacitysite/brian2_loihi) 

## Usage
Install Brian2Loihi as described on the github page (https://github.com/sagacitysite/brian2_loihi) as well as brian2 (pip install brian2)

Replace input graph in ExampleUsage with own dictionary describing a graph, examples see below.

Solution to SSSP in form of a list of shortest paths from node 0 to each node will be the output.

## Example Graph Dictionaries
1. {'num_nodes': 5, 'edge_tuples': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3)], 'costs_edges': np.array([3, 7, 5, 7, 4, 2],dtype=np.uint8), 'num_bits': 5}
2. {'num_nodes': 3, 'edge_tuples': [(0, 1), (0, 2), (1, 2)], 'costs_edges': array([5, 7, 2], dtype=uint8), 'num_bits': 4}
3. {'num_nodes': 5, 'edge_tuples': [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], 'costs_edges': array([ 9, 10,  1,  2,  3,  9,  6,  5,  9,  2], dtype=uint8), 'num_bits': 6}
4. {'num_nodes': 10, 'edge_tuples': [(0, 1), (0, 2), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (2, 5), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 5), (4, 7), (4, 9), (5, 6), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9), (8, 9)], 'costs_edges': array([ 6,  1,  2,  5, 10,  3,  5,  8,  3,  3,  9,  1,  4,  6,  5,  9,  2, 9,  7,  4,  1,  8,  2,  3,  1,  3, 10,  9,  6, 10,  9,  9,  5], dtype=uint8), 'num_bits': 8},
5. {'num_nodes': 2, 'edge_tuples': [(0, 1)], 'costs_edges': array([3], dtype=uint8), 'num_bits': 2}
6. {'num_nodes': 9, 'edge_tuples': [(0, 1), (0, 2), (0, 3), (0, 5), (0, 6), (0, 7), (0, 8), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)], 'costs_edges': array([ 7,  9,  8,  7,  6,  7,  3,  2, 10,  9,  2,  1,  8,  7, 10,  5,  5, 9,  3, 10,  2,  2,  1,  2,  5,  3, 10,  3,  2,  6,  7,  6,  8,  1],
      dtype=uint8), 'num_bits': 8}
7. {'num_nodes': 4, 'edge_tuples': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 'costs_edges': array([10,  4,  7,  7,  5,  8], dtype=uint8), 'num_bits': 6}
8. {'num_nodes': 7, 'edge_tuples': [(0, 1), (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (1, 6), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6), (4, 5)], 'costs_edges': array([3, 9, 2, 2, 3, 9, 3, 8, 5, 7, 6, 8, 4, 1], dtype=uint8), 'num_bits': 7}
9. {'num_nodes': 6, 'edge_tuples': [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 3), (4, 5)], 'costs_edges': array([10,  8,  8,  8,  2,  7,  3], dtype=uint8), 'num_bits': 6}
10. {'num_nodes': 7, 'edge_tuples': [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)], 'costs_edges': array([ 4,  7,  2, 10,  9,  7, 10,  8,  6,  6,  1,  7,  7,  8,  3,  6,  2,
        5,  6,  3,  5], dtype=uint8), 'num_bits': 7}


## Random Graph Generator
The random graph generator can be used to produce more random graph dictionaries such as the ones above. It also calculates and outputs the solutions to the SSSP as calculated using the Dijkstra algorithm. For it to run, networkx (pip install networkx) needs to be installed.
