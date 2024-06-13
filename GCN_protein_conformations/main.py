import torch
import torch.nn.functional as F
import argparse
from data.loader import get_dataloaders
from models.gcn import GCN
from utils.explain import explain, aggregate_edge_directions

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
            total += int(data.test_mask.sum())
    return correct / total

def main(data_dir, num_epochs=20, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, valid_loader, test_loader = get_dataloaders(data_dir)

    # Initialize model
    sample_data = next(iter(train_loader))
    model = GCN(input_dim = 3, hidden_channels=256)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, valid_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        if patience_counter >= patience:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Test the best model
    test_acc = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')

    # Explanation on test data
    for graph in test_loader:
        graph = graph.to(device)

        for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
            edge_mask = explain(method, model, graph, target=0)
            edge_mask_dict = aggregate_edge_directions(edge_mask, graph)
            print(f"Method: {title}")
            print("Edge\tScore")
            for edge, score in sorted(edge_mask_dict.items(), key=lambda x: x[1], reverse=True)[:20]:
                print(f"{edge}\t{score}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN Protein Data Processing")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing .pt files')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()
    main(args.data_dir, num_epochs=args.num_epochs, patience=args.patience)
