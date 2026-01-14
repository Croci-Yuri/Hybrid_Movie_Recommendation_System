######################################################################
##                         Hybrid Model                             ##
##          Hybrid Neural Collaborative Filtering Model             ##
##                    (Movie Content Features Only)                 ##
######################################################################

# Import necessary libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error




##                       Hybrid Model Class                         ##
######################################################################

class HybridNN(nn.Module):

    """
    Hybrid recommendation model combining:
    - Pre-trained Funk SVD embeddings and biases that are frozen
    - Movie content features that are frozen
    - A feedforward neural network learning a residual correction over SVD and content features

    Prediction follows:
        r_hat_ui = mu + b_u + b_i + f(p_u, q_i, x_i)


    where:        
        - mu: global mean rating
        - b_u, b_i: user and item biases
        - p_u, q_i: user and item latent factors from pre-trained SVD
        - x_i: standardized movie content features
        - f(.,.,.): feedforward neural network producing residual correction


    Parameters:
        - user_factors: Tensor (n_users, k)
        - item_factors: Tensor (n_items, k)
        - user_bias: Tensor (n_users,)
        - item_bias: Tensor (n_items,)
        - global_mean: float
        - content_dim: int, dimension of movie content features
        - hidden_dims: tuple, dimensions of hidden layers in the feedforward NN

    """

    def __init__(self, user_factors, item_factors,
                user_bias, item_bias, global_mean,
                content_dim, hidden_dims=(128, 64)):

        super().__init__()

        ## Store pre-trained SVD components ##
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_bias = user_bias
        self.item_bias = item_bias

        # Global mean rating saved as buffer
        self.register_buffer(
            "global_mean",
            torch.tensor(global_mean, dtype=torch.float32)
        )

        # Latent embedding dimension 
        emb_dim = user_factors.shape[1]

        
        ## Feedforward neural network f(p_u, q_i, x_i) ## 

        layers = []
        input_dim = 2 * emb_dim + content_dim

        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        # Final linear layer producing the residual correction
        layers.append(nn.Linear(input_dim, 1))

        # Combine layers into a sequential module
        self.ffn = nn.Sequential(*layers)

    ## Forward pass ##
    ##################

    def forward(self, user_idx, item_idx, content_vec):


        # Retrieve frozen SVD embeddings
        p_u = self.user_factors[user_idx]     # (B, k)
        q_i = self.item_factors[item_idx]     # (B, k)

        # Retrieve SVD biases
        b_u = self.user_bias[user_idx]        # (B,)
        b_i = self.item_bias[item_idx]        # (B,)

        # Concatenate latent factors with content features
        nn_input = torch.cat([p_u, q_i, content_vec], dim=1)

        # Neural network residual
        residual = self.ffn(nn_input).squeeze(1)

        # Final prediction
        return self.global_mean + b_u + b_i + residual




##                       Training function                          ##
######################################################################

def train_hybrid_model(model, train_df, val_df, 
                       content_features, user2idx, item2idx, 
                       n_epochs=50, batch_size=1024, lr=1e-3, weight_decay=1e-5,
                       device="cpu",val_epochs=None, verbose=True):
    """

    Training loop for the HybridNN model.
    
    Parameters:

    model : nn.Module -> HybridNN instance
    train_df : pd.DataFrame -> Training dataframe with columns ['userId', 'movieId', 'rating']
    val_df : pd.DataFrame -> Validation dataframe
    content_features : pd.DataFrame or np.array -> Standardized movie content features indexed by movieId
    user2idx : dict -> Mapping from raw userId to model index
    item2idx : dict -> Mapping from raw movieId to model index
    n_epochs : int -> Number of training epochs
    batch_size : int -> Mini-batch size
    lr : float -> Learning rate for optimizer
    weight_decay : float -> L2 regularization (weight decay)
    device : str -> Device to run training on ("cpu" or "cuda")
    val_epochs : None or tuple/list -> If None, validate every epoch. If tuple/list, validate only on the given epoch numbers.
    verbose : bool -> If True, print training and validation loss per epoch
    
    Returns:
    model : nn.Module --> Trained HybridSVDNN
    history : dict --> Training and validation loss history

    """

    # Model to device
    model.to(device)

    # Validation epochs 
    if val_epochs is None:
        val_epochs = tuple(range(1, n_epochs + 1))

    # Loss function and optimizer 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.ffn.parameters(), lr=lr, weight_decay=weight_decay)

    # Convert content features to tensor #
    if isinstance(content_features, pd.DataFrame):
        content_tensor = torch.tensor(content_features.values, dtype=torch.float32, device=device)
    else:
        content_tensor = torch.tensor(content_features, dtype=torch.float32, device=device)


    ## Training loop ## 

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):

        model.train()
        epoch_loss = 0.0

        # Shuffle training data
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        # Mini-batch iteration
        for start in range(0, len(train_df), batch_size):
            batch = train_df.iloc[start:start+batch_size]

  
            # Map raw IDs to indices (revert to 0 if not found)
            user_idx = torch.tensor([user2idx.get(u, 0) for u in batch['userId']], dtype=torch.long, device=device)
            item_idx = torch.tensor([item2idx.get(i, 0) for i in batch['movieId']], dtype=torch.long, device=device)


            # Gather content features for items in the batch
            batch_content = content_tensor[item_idx]

            # Ratings
            ratings = torch.tensor(batch['rating'].values, dtype=torch.float32, device=device)

            # Forward pass
            preds = model(user_idx, item_idx, batch_content)

            # Compute loss
            loss = criterion(preds, ratings)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)

        # Average training loss #
        train_loss = epoch_loss / len(train_df)
        history["train_loss"].append(train_loss)

        ## Validation step  over the specified epochs ##
        if epoch in val_epochs:
            model.eval()
            with torch.no_grad():
                user_idx_val = torch.tensor([user2idx.get(u, 0) for u in val_df['userId']], dtype=torch.long, device=device)
                item_idx_val = torch.tensor([item2idx.get(i, 0) for i in val_df['movieId']], dtype=torch.long, device=device)
                batch_content_val = content_tensor[item_idx_val]
                ratings_val = torch.tensor(val_df['rating'].values, dtype=torch.float32, device=device)
                val_preds = model(user_idx_val, item_idx_val, batch_content_val)
                val_loss = criterion(val_preds, ratings_val).item()
                history["val_loss"].append(val_loss)
            if verbose:
                print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, history




##                        Evaluation Function                       ##
######################################################################

def evaluate_hybrid_model(model, eval_df, content_features, user2idx, item2idx, device="cpu"):
    """
    Evaluate HybridSVDNN model on a dataset.
    
    Parameters:
    model : nn.Module -> Trained HybridSVDNN instance
    eval_df : pd.DataFrame -> Evaluation dataframe ['userId', 'movieId', 'rating']
    content_features : pd.DataFrame or np.array -> Standardized movie content features indexed by movieId
    user2idx : dict -> Mapping from raw userId to model index
    item2idx : dict -> Mapping from raw movieId to model index
    device : str -> Device ("cpu" or "cuda")
    
    Returns:
    preds : np.array -> Predicted ratings
    rmse : float -> Root mean squared error over eval_df
    """
    
    # Move model to device and set to eval mode 
    model.to(device)
    model.eval()

    # Convert content features to tensor
    if isinstance(content_features, pd.DataFrame):
        content_tensor = torch.tensor(content_features.values, dtype=torch.float32, device=device)
    else:
        content_tensor = torch.tensor(content_features, dtype=torch.float32, device=device)

    # Map raw IDs to indices #
    user_idx = torch.tensor([user2idx.get(u, 0) for u in eval_df['userId']], dtype=torch.long, device=device)
    item_idx = torch.tensor([item2idx.get(i, 0) for i in eval_df['movieId']], dtype=torch.long, device=device)

    # Gather content features for items in batch
    batch_content = content_tensor[item_idx]

    # Ratings
    ratings = eval_df['rating'].values

    # Forward pass #
    with torch.no_grad():
        preds = model(user_idx, item_idx, batch_content).cpu().numpy()

    # Compute RMSE #
    rmse = np.sqrt(mean_squared_error(ratings, preds))

    return preds, rmse
