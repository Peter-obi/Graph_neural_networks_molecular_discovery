# GCN for classification of protein conformations + interpretability

This repository contains code for training and explaining a Graph Convolutional Network (GCN) model on protein structures from different classes. Example: Active and inactive conformations of the protein. We add an interpretability layer for understanding the edges important for classification.

### Installation

Clone the repository:
```
git clone https://github.com/your-username/gcn-protein-data-processing.git
cd gcn-protein-data-processing
```
Install the required dependencies:
```
pip install -r requirements.txt
```

### Usage
To train and evaluate the GCN model and generate explanations, run the following command:
```
python main.py --data-dirs path/to/data1 path/to/data2 --num-epochs 20 --patience 10
```

--data-dirs: Specify one or more directories containing the .pt files. Separate multiple directories with spaces.

--num-epochs: Set the number of training epochs (default: 20).

--patience: Set the number of epochs to wait for improvement before early stopping (default: 10).

### Data
You can create the graphs using the create_graphs.py. This generates edges between c-alpha atoms within a certain threshold and stores them in a .pt file. Usage could be implemented by taking breaking up a trajectory into individual pdb files and then create graphs from that. Creativity with this is encouraged!
The code assumes that the protein data is stored in .pt files located in one or more directories. Each .pt file should contain a torch_geometric.data.Data object with the following attributes:
```
x: Node feature matrix of shape [num_nodes, num_features].
edge_index: Edge index tensor of shape [2, num_edges].
y: Graph label tensor of shape [1].
```
The loader.py file contains functions to load the data from multiple directories, split it into train, validation, and test sets, and create data loaders for each set.

### Model

The GCN model is defined in the gcn.py file. It consists of three graph convolutional layers followed by a linear layer for classification. The model takes the node features, edge indices, and batch information as input and outputs the log-softmax probabilities for each class.

### Explanations
The explain.py file contains functions to generate explanations for the GCN model using the Integrated Gradients and Saliency methods from the Captum library.

### Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for more information.
