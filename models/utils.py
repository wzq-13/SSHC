from models.neural_networks import MLP

def create_model(config, model_type='MLP', device='cpu'):
    """Creates and returns a neural network model."""
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    model = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
    return model.to(device)