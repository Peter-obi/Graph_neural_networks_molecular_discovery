# GCN Protein Data Processing
This repository contains code for training and explaining a Graph Convolutional Network (GCN) model on protein data stored in .pt files.

### Installation

Clone the repository:
'''
git clone https://github.com/your-username/gcn-protein-data-processing.git
cd gcn-protein-data-processing
'''
Install the required dependencies:
'''
pip install -r requirements.txt
'''

### Usage
To train and evaluate the GCN model and generate explanations, run the following command:
'''
python main.py --data-dirs path/to/data1 path/to/data2 --num-epochs 20 --patience 10
'''

--data-dirs: Specify one or more directories containing the .pt files. Separate multiple directories with spaces.
--num-epochs: Set the number of training epochs (default: 20).
--patience: Set the number of epochs to wait for improvement before early stopping (default: 10).

### Data
The code assumes that the protein data is stored in .pt files located in one or more directories. Each .pt file should contain a torch_geometric.data.Data object with the following attributes:
'''
x: Node feature matrix of shape [num_nodes, num_features].
edge_index: Edge index tensor of shape [2, num_edges].
y: Graph label tensor of shape [1].
'''
The loader.py file contains functions to load the data from multiple directories, split it into train, validation, and test sets, and create data loaders for each set.

### Model

The GCN model is defined in the gcn.py file. It consists of three graph convolutional layers followed by a linear layer for classification. The model takes the node features, edge indices, and batch information as input and outputs the log-softmax probabilities for each class.
The model is trained using the cross-entropy loss and the Adam optimizer. Early stopping is implemented based on the validation accuracy.
Explanations
The explain.py file contains functions to generate explanations for the GCN model using the Integrated Gradients and Saliency methods from the Captum library.
The explain function takes the explanation method, the trained model, the input graph data, and the target class as arguments and returns an edge mask indicating the importance of each edge.
The aggregate_edge_directions function aggregates the edge importance scores for each edge direction in the graph.
The code generates explanations for all the test graphs and prints the top 20 most important edges and their corresponding scores for each explanation method.

### Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for more information.
