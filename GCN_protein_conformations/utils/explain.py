import torch
import numpy as np
from captum.attr import IntegratedGradients, Saliency

def explain(method, model, data, target):
    model.eval()
    input_mask = data.x.clone().requires_grad_(True).to(data.x.device)

    if method == 'ig':
        ig = IntegratedGradients(model)
        mask = ig.attribute(input_mask, target=target, additional_forward_args=(data,))
    elif method == 'saliency':
        saliency = Saliency(model)
        mask = saliency.attribute(input_mask, target=target, additional_forward_args=(data,))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask

def aggregate_edge_directions(edge_mask, data):
    from collections import defaultdict
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict
