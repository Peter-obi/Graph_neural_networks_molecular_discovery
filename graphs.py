#!/usr/bin/env python
import csv
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils.convert import to_networkx, from_networkx
import glob
from biopandas.pdb import PandasPdb


df = pd.read_csv('pdb_labels.csv')

data_list = []

def compute_edge_index(x, threshold, min_distance,node_labels):
    # create an empty list to store the edge indices
    edge_index = []

    # iterate over the coordinates and compute the distances
    # between all pairs of C-alpha atoms
    for i in range(x.size(0)):
        for j in range(i + 1, x.size(0)):
            # compute the distance between the points
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

             # get the residue numbers for the two C-alpha atoms
            residue_number_i = node_labels[i]
            residue_number_j = node_labels[j]

            # compute the difference in residue numbers
            residue_number_difference = abs(residue_number_i - residue_number_j)

            # add the pair of points to the edge index if the distance is within the specified threshold and the difference in residue numbers is at least min_distance
            if distance <= threshold and residue_number_difference >= min_distance:
                edge_index.append([i, j])

    # convert the edge index list to a tensor
    edge_index = torch.tensor(edge_index).t().contiguous()

    return edge_index
   


def load_pdb_coords(pdb_file):
    # create a PandasPdb object and read the PDB file
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)

    # get the coordinates of the C-alpha atoms
    ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].values

    return coords


def load_pdb_residue_numbers(pdb_file, x):
    # create a PandasPdb object and read the PDB file
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)

    # get the coordinates of the C-alpha atoms
    ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].values

    # create a dictionary mapping the coordinates to the
    # residue numbers of the C-alpha atoms
    coord_to_residue = {}
    for i, coord in enumerate(coords):
        coord_tuple = tuple(coord)
        coord_to_residue[coord_tuple] = ca_atoms.iloc[i]['residue_number']

    # create a list to store the residue numbers for each node
    residue_numbers = []

    # iterate over the nodes in the graph and map the
    # coordinates to the corresponding residue numbers
    for i in range(x.size(0)):
        coord = tuple(x[i, :].tolist())
        residue_numbers.append(coord_to_residue[coord])

    return residue_numbers

for index, row in df.iterrows():
    # get the PDB file name and label
    pdb_file = row['PDB']
    label = row['label']

    # load the coordinates of the C-alpha atoms from the PDB file
    coords = load_pdb_coords(pdb_file)
    x = torch.tensor(coords)
    residue_numbers = load_pdb_residue_numbers(pdb_file, x)
    node_labels = torch.tensor(residue_numbers)
    edge_index = compute_edge_index(x, 8.0, 3, node_labels)
    y = torch.tensor([label])
    data = Data(x=x, edge_index=edge_index, y=y, node_labels=node_labels)
    torch.save(data, f"{pdb_file}.pt")





