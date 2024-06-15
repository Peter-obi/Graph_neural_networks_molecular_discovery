import torch
from torch_geometric.data import Data
from biopandas.pdb import PandasPdb
import glob

def compute_edge_index(x, threshold, min_distance, node_labels):
    """
    Compute edge indices for the graph based on distance and residue number difference.

    Parameters:
    - x (Tensor): Coordinates of the nodes.
    - threshold (float): Maximum distance between nodes to create an edge.
    - min_distance (int): Minimum difference in residue numbers to create an edge.
    - node_labels (Tensor): Residue numbers of the nodes.

    Returns:
    - edge_index (Tensor): Edge indices for the graph.
    """
    edge_index = []
    for i in range(x.size(0)):
        for j in range(i + 1, x.size(0)):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            residue_number_i = node_labels[i]
            residue_number_j = node_labels[j]
            residue_number_difference = abs(residue_number_i - residue_number_j)
            if distance <= threshold and residue_number_difference >= min_distance:
                edge_index.append([i, j])
    return torch.tensor(edge_index).t().contiguous()

def load_pdb_coords(pdb_file):
    """
    Load coordinates of C-alpha atoms from a PDB file.

    Parameters:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - coords (ndarray): Coordinates of C-alpha atoms.
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)
    ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    return ca_atoms[['x_coord', 'y_coord', 'z_coord']].values

def load_pdb_residue_numbers(pdb_file, x):
    """
    Load residue numbers of C-alpha atoms from a PDB file.

    Parameters:
    - pdb_file (str): Path to the PDB file.
    - x (Tensor): Coordinates of the nodes.

    Returns:
    - residue_numbers (list): Residue numbers for each node.
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)
    ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].values
    coord_to_residue = {tuple(coord): ca_atoms.iloc[i]['residue_number'] for i, coord in enumerate(coords)}
    return [coord_to_residue[tuple(x[i, :].tolist())] for i in range(x.size(0))]

# List of PDB files in the current directory
pdb_files = glob.glob("*.pdb")

for pdb_file in pdb_files:
    coords = load_pdb_coords(pdb_file)
    x = torch.tensor(coords)
    residue_numbers = load_pdb_residue_numbers(pdb_file, x)
    node_labels = torch.tensor(residue_numbers)
    edge_index = compute_edge_index(x, 8.0, 3, node_labels)
    y = torch.tensor([0])
    data = Data(x=x, edge_index=edge_index, y=y, node_labels=node_labels)
    torch.save(data, f"{pdb_file}.pt")
