import torch
import numpy as np
from captum.attr import IntegratedGradients, Saliency

def explain(method, model, data, target):
    """
    Generate explanations for the GCN model.

    Args:
        method (str): Explanation method to use ('ig' for Integrated Gradients or 'saliency' for Saliency).
        model (nn.Module): GCN model.
        data (torch_geometric.data.Data): Input graph data.
        target (int): Target class for explanation.

    Returns:
        numpy.ndarray: Edge mask indicating the importance of each edge.
    """
    model.eval()
    input_mask = data.x.clone().requires_grad_(True).to(data.x.device)

    if method == 'ig':
        ig = IntegratedGradients(model)
        mask = ig.attribute(input_mask, target=target, additional_forward_args=(data.edge_index, data.batch))
    elif method == 'saliency':
        saliency = Saliency(model)
        mask = saliency.attribute(input_mask, target=target, additional_forward_args=(data.edge_index, data.batch))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask

def aggregate_edge_directions(edge_mask, data):
    """
    Aggregate the edge importance scores for each edge direction.

    Args:
        edge_mask (numpy.ndarray): Edge mask indicating the importance of each edge.
        data (torch_geometric.data.Data): Input graph data.

    Returns:
        dict: Dictionary mapping edge tuples to their aggregated importance scores.
    """
    from collections import defaultdict
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict
