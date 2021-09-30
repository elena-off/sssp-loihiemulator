from brian2 import *
prefs.codegen.target = 'numpy'  # use the Python fallback
import numpy as np
from brian2_loihi import *
from time import *

class PathFinding():

    def __init__(self, graph_info):
        # Define variables/parameters
        self.net = LoihiNetwork()
        self.num_bits = graph_info["num_bits"]
        self.edge_tuples = graph_info["edge_tuples"]
        self.num_nodes = graph_info["num_nodes"]
        self.costs_edges = graph_info["costs_edges"].reshape(len(graph_info["costs_edges"]),1)
        self.costs_edges = np.unpackbits(self.costs_edges, axis=1)
        self.costs_edges = np.flip(self.costs_edges,axis=1)
        self.num_words = np.zeros(self.num_nodes)
        #dictionaries for easy access to edges and synapses
        self.edges = {}
        self.synapses = {}
        # count number of input to each node
        self.count_num_words()
        # list for Spike monitors necessary for reading output
        self.output_monitors = []
        # make nodes
        self.nodes = self.make_nodes()
        # Connect path by making edges
        self.connect()

    def connect(self):
        #list for how many ports are needed at each minimum component
        ports = np.zeros(self.num_nodes,dtype = int)
        #time for one cycle to be completed
        t_cyc = 4*self.num_bits + 11
        #external neuron for cost input, fires at beginning of each addition phase of each cycle
        times = np.arange(self.num_nodes)*(t_cyc)
        indices = np.zeros(len(times))
        ar = np.arange(len(self.costs_edges[0]))
        EXT = LoihiSpikeGeneratorGroup(1, indices, times)
        self.net.add(EXT)

        #make edges and connect them to nodes
        k = 0
        for (fr, to) in self.edge_tuples:
            # make edge and append to network
            edge = AddComponent(self.num_bits, fr, to, ports[to])
            self.net.add(edge.objects)

            #name convention for edges: edge_fr_to_port
            edge_name = "edge"+"_"+str(fr)+"_"+str(to)+"_"+str(ports[to])
            self.edges[edge_name] = edge

            #connect node output port to edge input port and edge output port to node input port
            syn_name_input = edge_name + "_" + "input"
            self.synapses[syn_name_input] = LoihiSynapses(self.nodes[fr].objects[0][1],self.edges[edge_name].objects[0][0])
            self.synapses[syn_name_input].connect(i=np.arange(self.num_bits), j = np.arange(self.num_bits))
            self.synapses[syn_name_input].w = 4

            syn_name_output = edge_name + "_" + "output"
            self.synapses[syn_name_output] = LoihiSynapses(self.edges[edge_name].objects[0][0],self.nodes[to].objects[0][0])
            self.synapses[syn_name_output].connect(i=np.arange(6*self.num_bits,7*self.num_bits),
                                                   j = np.arange(self.num_bits*int(ports[to]), self.num_bits*(int(ports[to])+1)))
            self.synapses[syn_name_output].w = 2

            #count number of ports up for next iteration
            ports[to] +=1

            #connect external cost neuron to edge input
            syn_name_cost = edge_name + "_" + "cost"
            self.synapses[syn_name_cost] = LoihiSynapses(EXT,self.edges[edge_name].objects[0][0])
            self.synapses[syn_name_cost].connect(i = np.zeros(np.count_nonzero(self.costs_edges[k]),dtype=int),
                                                 j = ar[np.where(self.costs_edges[k])]+self.num_bits)
            self.synapses[syn_name_cost].w = 4

            k += 1
            self.net.add([self.synapses[syn_name_input],self.synapses[syn_name_output],self.synapses[syn_name_cost]])


    def get_node(self,node_id):
        return nodes[node_id]

    def count_num_words(self):
        #count number of inputs to each node
        self.num_words[0] = 1
        for (fr,to) in self.edge_tuples:
            self.num_words[to]+=1


    def make_nodes(self):
        #make all nodes by initializing Minimum Components with appropriate numbers of inputs
        nodes = []
        for i in range(self.num_nodes):
            node = MinimumComponent(i, self.num_bits, int(self.num_words[i]),self.num_nodes)
            nodes.append(node)
            self.net.add(node.objects)
            self.output_monitors.append(node.objects[0][-1])
            #print number of edge to see progress during initialization
            print(i)
        return nodes


class AddComponent():

    def __init__(self, num_bits, source_id, target_id, port_id):
        #initialize important parameters
        self.num_bits = num_bits
        self.objects=[]
        self.source_id = source_id
        self.target_id = target_id
        self.port_id = port_id
        self.make_edge()


    def make_edge(self):
        """
        These are the weight matrices necessary for connecting the different types of neurons within the addition component.
        There are n_bits each of  COST,MIN,CARRY, AND, XOR, AND2 and SUM-neurons each, with one of each type being necessary for
        each layer and n_bits layers making up the addition component.
        Delay: 0
                                  TO
        [          COST MIN CARRY AND XOR AND2 SUM
            COST  [0    0    R    2   0    0   0   ]
            MIN   [0    0    R    2   0    0   0   ]
            CARRY [0    0    R    0   0    2   0   ]
       FROM AND   [0    0    0    0   -8   0   0   ]
            XOR   [0    0    0    0   0    2   0   ]
            AND2  [0    0    0    0   0    0   -8  ]
            SUM   [0    0    0    0   0    0   0   ]
        R is a matrix with 1s along the first off-diagonal because the COST, MIN and CARRY neurons of one layer need to be
        connected to the CARRY neurons of the next layer rather than the same layer

        Weight matrix for synapses with a delay of 2 timesteps
        Delay: 2
        [          COST MIN CARRY AND XOR AND2 SUM
            COST  [0    0    0    0   4    0   0   ]
            MIN   [0    0    0    0   4    0   0   ]
            CARRY [0    0    0    0   0    0   0   ]
       FROM AND   [0    0    0    0   0    0   0   ]
            XOR   [0    0    0    0   0    0   0   ]
            AND2  [0    0    0    0   0    0   0   ]
            SUM   [0    0    0    0   0    0   0   ]
        ]

        Weight matrix for synapses with a delay of 3 timesteps
        Delay: 3
        [          COST MIN CARRY AND XOR AND2 SUM
            COST  [0    0    0    0   0    0   0   ]
            MIN   [0    0    0    0   0    0   0   ]
            CARRY [0    0    0    0   0    0   4   ]
       FROM AND   [0    0    0    0   0    0   0   ]
            XOR   [0    0    0    0   0    0   4   ]
            AND2  [0    0    0    0   0    0   0   ]
            SUM   [0    0    0    0   0    0   0   ]
        ]
        """

        #edge name for unique synapse names, format "edge"_source_target_port
        edge_name = "edge" + str(self.source_id) + "_" + str(self.target_id) + "_" + str(self.port_id)

        #initialize the ADD-neurons
        ADD = LoihiNeuronGroup(7*self.num_bits, threshold_v_mant=2, decay_v=1024
                               , decay_I=4096)

        #define weight matrices
        base_del0 = np.array([
                   [0,0,0,2,0,0,0],
                   [0,0,0,2,0,0,0],
                   [0,0,0,0,0,2,0],
                   [0,0,0,0,-8,0,0],
                   [0,0,0,0,0,2,0],
                   [0,0,0,0,0,0,-8],
                   [0,0,0,0,0,0,0]],dtype=int)

        base_del1 = np.array([
                   [0,0,0,0,4,0,0],
                   [0,0,0,0,4,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0]],dtype=int)

        base_del2 = np.array([
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,4],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,4],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0]],dtype=int)
        I = np.eye(self.num_bits,dtype=int)
        weight_matrix_del0 = np.kron(base_del0,I)
        weight_matrix_del1 = np.kron(base_del1,I)
        for i in range(self.num_bits-1):
            weight_matrix_del0[i][2*self.num_bits+i+1] = 2
            weight_matrix_del0[self.num_bits+i][2*self.num_bits+i+1] = 2
            weight_matrix_del0[2*self.num_bits+i][2*self.num_bits+i+1] = 2
        weight_matrix_del2 = np.kron(base_del2,I)

        #connections without any delay
        DEL0 = LoihiSynapses(ADD, ADD, name=edge_name + 'DEL_0',sign_mode=synapse_sign_mode.MIXED)
        (sources,targets) = np.where(weight_matrix_del0)
        DEL0.connect(i=sources, j=targets)
        DEL0.w = weight_matrix_del0[sources, targets]

        #connections with a delay of 2 timesteps
        DEL1 = LoihiSynapses(ADD,ADD, delay = 2, name=edge_name + 'DEL1')
        (sources,targets) = np.where(weight_matrix_del1)
        DEL1.connect(i=sources, j=targets)
        DEL1.w = weight_matrix_del1[sources, targets]

        #connections with a delay of 3 timesteps
        DEL2 = LoihiSynapses(ADD,ADD,delay=3, name=edge_name + 'DEL2')
        (sources,targets) = np.where(weight_matrix_del2)
        DEL2.connect(i=sources, j=targets)
        DEL2.w = weight_matrix_del2[sources, targets]

        self.objects.append([ADD,DEL0,DEL1,DEL2])


class MinimumComponent():
    """
        Define a Minimum Component with num_words input words, each having num_bits bits"""
    def __init__(self, node_id, num_bits, num_words,num_nodes):
        self.node_id = node_id
        self.num_bits = num_bits
        self.num_words = num_words
        self.num_nodes = num_nodes
        self.objects = []
        self.make_node()

    def make_node(self):
        #parameters: L = number of bits, d = number of words, i if True or j if False

        # returns arrays for connecting IN to ZER neurons correctly, returns i (presynaptic) range for  i = True,
        # and j (postsynaptic) range for False
        def InZer(i):
            if(i):
                ar= []
                for i in range(self.num_bits)[::-1]:
                    ar = np.concatenate((ar, np.arange(self.num_words)*self.num_bits+i), axis=None )
                return ar.astype(np.int64)
            else:
                return np.arange(self.num_bits*self.num_words)

        #returns array for postsynaptic range (j) for connecting IN and COP neurons
        def InCop():
            i = self.num_bits-1
            ar = []
            for j in arange(self.num_bits*self.num_words):
                ar.append(i)
                if i%self.num_bits == 0:
                    i +=2*self.num_bits
                i-=1
            return ar

        #time for one cycle to be completed
        t_cyc = 4*self.num_bits + 11
        #define external neuron for cost input, fires at beginning of each minimum phase of each cycle
        times = np.arange(self.num_nodes)*(t_cyc) + 8
        indices = np.zeros(len(times))
        EXT2 = LoihiSpikeGeneratorGroup(1, indices, times)

        #decays for different num_bits
        decays = [0,2700,1500,1050,800,600,500,410,350,300,260,225,200,175]
        decay = decays[self.num_bits]

        IN = LoihiNeuronGroup(self.num_bits*self.num_words,threshold_v_mant=2, decay_v=decay, decay_I=4096)

        #Neurons CAN- Candidates for Maximum after each bit, from most significant to least
        CAN = LoihiNeuronGroup(self.num_words*(self.num_bits+1),threshold_v_mant=2, decay_v=decay, decay_I=4096)

        #Layer Neurons Or
        OR =  LoihiNeuronGroup(self.num_bits,threshold_v_mant=2, decay_v=decay, decay_I=4096)

        #Layer Neurons INH- Inhibitory Neurons for Candidate Neurons
        INH =  LoihiNeuronGroup(self.num_words*self.num_bits,threshold_v_mant=2, decay_v=decay, decay_I=4096)

        #ZER Neurons, check whether input bit is 0 and number is candidate
        ZER =  LoihiNeuronGroup(self.num_words*self.num_bits,threshold_v_mant=2, decay_v=300, decay_I=4096)

        #Copy Layer for output
        COP = LoihiNeuronGroup(self.num_words*self.num_bits,threshold_v_mant=2, decay_v=decay, decay_I=4096)

        #Output Neurons, copy minimum
        OUT = LoihiNeuronGroup(self.num_bits,threshold_v_mant=2, decay_v=decay, decay_I=4096)

        #setup layer: candidate neurons for first bit of each word should be active
        EXTCAN = LoihiSynapses(EXT2,CAN)
        EXTCAN.connect(i=np.zeros(self.num_words).astype(int),j=np.arange(self.num_words))
        EXTCAN.w = 4

        #external to input for synchronization
        EXTIN = LoihiSynapses(EXT2, IN)
        EXTIN.connect(i=np.zeros(self.num_bits*self.num_words).astype(int),j=np.arange(self.num_bits*self.num_words))
        EXTIN.w = 2

        #main layers: check whether inputs still candidates after next significant bits
        #for explanation of connections and weights, see thesis
        INZER= LoihiSynapses(IN,ZER,sign_mode=synapse_sign_mode.INHIBITORY)
        INZER.connect(i=InZer(True), j =InZer(False))
        INZER.w = -16
        CANZER = LoihiSynapses(CAN,ZER)
        CANZER.connect(i=np.arange(self.num_words*self.num_bits),j=np.arange(self.num_words*self.num_bits))
        CANZER.w = 4
        ZERINH = LoihiSynapses(ZER,INH,sign_mode=synapse_sign_mode.INHIBITORY)
        ZERINH.connect(i=np.arange(self.num_words*self.num_bits),j=np.arange(self.num_words*self.num_bits))
        ZERINH.w = -8
        ZEROR = LoihiSynapses(ZER,OR)
        ZEROR.connect(i=np.arange(self.num_bits*self.num_words), j =np.repeat(np.arange(self.num_bits),self.num_words))
        ZEROR.w = 4
        ORINH = LoihiSynapses(OR,INH)
        ORINH.connect(i=np.repeat(np.arange(self.num_bits),self.num_words),j=np.arange(self.num_words*self.num_bits))
        ORINH.w = 4
        INHCAN = LoihiSynapses(INH,CAN,sign_mode=synapse_sign_mode.INHIBITORY)
        INHCAN.connect(i=np.arange(self.num_words*self.num_bits),j=np.arange(self.num_words,self.num_words*self.num_bits+self.num_words))
        INHCAN.w = -8
        CANCAN = LoihiSynapses(CAN,CAN,delay= 3)
        CANCAN.connect(i=np.arange(self.num_words*self.num_bits),j=np.arange(self.num_words,self.num_words*self.num_bits+self.num_words))
        CANCAN.w = 4
        INCOP = LoihiSynapses(IN,COP,delay= 3*self.num_bits)
        INCOP.connect(i=np.arange(self.num_words*self.num_bits),j=InCop())
        INCOP.w = 2
        CANCOP = LoihiSynapses(CAN,COP)
        CANCOP.connect(i=np.repeat(np.arange(self.num_bits*self.num_words, self.num_bits*self.num_words+self.num_words),self.num_bits), j = np.arange(self.num_words*self.num_bits))
        CANCOP.w = 2
        COPOUT = LoihiSynapses(COP,OUT)
        COPOUT.connect(i=np.arange(self.num_bits*self.num_words),j=np.tile(np.arange(self.num_bits)[::-1],self.num_words))
        COPOUT.w = 4

        OUT_MON = LoihiSpikeMonitor(OUT, name = "out_monitor_node_" + str(self.node_id))

        self.objects.append([IN, OUT, EXT2, CAN, OR, INH, ZER, COP, EXTCAN, EXTIN, INZER, CANZER, ZERINH, ZEROR, ORINH, INHCAN, CANCAN, INCOP,
                            CANCOP, COPOUT, OUT_MON])


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
    valuelist = split(network.output_monitors[i].i,index)
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
    valuelist = split(network.output_monitors[i].i,index)
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
