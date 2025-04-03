import numpy as np
import networkx as nx

# get pdb coordinates
def get_contact_map(pdb_path, threshold= 12.0, max_L= 1000):
    
    with open(pdb_path, 'r') as file:
        pdb_content = file.read()
        
    cb_coordinates = []
    for line in pdb_content.splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            if atom_name == "CA":
                cb_coordinates.append((x, y, z))
    
    num_residues = len(cb_coordinates)
    contact_map = np.zeros((max_L, max_L), dtype=int)
    
    for i in range(int(num_residues)):
        for j in range(i + 1, int(num_residues)):
            residue_i = cb_coordinates[i]
            residue_j = cb_coordinates[j]
            
            distance = np.sqrt(
                (residue_i[0] - residue_j[0]) ** 2 +
                (residue_i[1] - residue_j[1]) ** 2 +
                (residue_i[2] - residue_j[2]) ** 2
            )
            
            if distance <= threshold:
                contact_map[i][j] = 1
                contact_map[j][i] = 1
    
    return contact_map, num_residues

def compute_SPD_centrality(contact_map):
    shape = contact_map.shape
    graph = nx.from_numpy_array(contact_map)
    centrality = np.array(list(nx.centrality.betweenness_centrality(graph).values()))
    SPD = np.empty(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            try:
                SPD[i][j] = nx.shortest_path_length(graph, source=i, target=j)
            except:
                SPD[i][j] = 0.0
    
    return SPD, centrality

    
    
    
    