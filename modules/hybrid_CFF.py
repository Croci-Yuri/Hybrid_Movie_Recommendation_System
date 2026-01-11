######################################################################
##                         Hybrid Model                             ##
##          Hybrid Collaborative Filtering + Content-based Model    ##
######################################################################

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

######################################################################
##                       Hybrid Model Class                         ##
######################################################################

class Hybrid_CCF(nn.Module):

    """
    Hybrid Collaborative Filtering + Content-based Model

    Prediction formula:
        rating = global_bias + user_bias + item_bias + user_embed^T @ item_embed
                 + w_u^T @ user_features + w_i^T @ item_features

    Components:
        - user_embed / item_embed: latent factor embeddings for users and items
        - w_u / w_i: linear weights for user/item content features
        - global_bias / user_bias / item_bias: biases
    """

    def __init__(self, n_users, n_items, n_factors=20,
                 user_feature_dim=0, item_feature_dim=0):
        super().__init__()

        
        # Collaborative Filtering #
        self.user_embed = nn.Embedding(n_users, n_factors)  # user latent factors
        self.item_embed = nn.Embedding(n_items, n_factors)  # item latent factors

        self.user_bias = nn.Embedding(n_users, 1)  # user bias
        self.item_bias = nn.Embedding(n_items, 1)  # item bias
        self.global_bias = nn.Parameter(torch.zeros(1))  # overall bias

        # Content-based Linear Terms #
        self.use_user_features = user_feature_dim > 0
        self.use_item_features = item_feature_dim > 0
        if self.use_user_features:
            self.w_u = nn.Linear(user_feature_dim, 1, bias=False) # user-content feature 
        if self.use_item_features:
            self.w_i = nn.Linear(item_feature_dim, 1, bias=False) # item-content feature 

        # Initialize embeddings and weights
        self._init_weights()

    
    # Weight Initialization function #

    def _init_weights(self):
        """Initialize embeddings and linear weights"""
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.item_bias.weight, 0)
        if self.use_user_features:
            nn.init.normal_(self.w_u.weight, std=0.01)
        if self.use_item_features:
            nn.init.normal_(self.w_i.weight, std=0.01)

    
    # Forward Pass #
    ################

    def forward(self, user_ids, item_ids, user_features=None, item_features=None):

        """
        Compute predictions for given user/item IDs and optional features
        """

        # Collaborative filtering  terms via embedding dot product
        u_emb = self.user_embed(user_ids)
        i_emb = self.item_embed(item_ids)
        interaction = (u_emb * i_emb).sum(dim=1)

        # Add bias terms
        pred = self.global_bias + self.user_bias(user_ids).squeeze() + self.item_bias(item_ids).squeeze() + interaction

        # Add content-based linear terms if provided
        if self.use_user_features and user_features is not None:
            pred += self.w_u(user_features).squeeze()
        if self.use_item_features and item_features is not None:
            pred += self.w_i(item_features).squeeze()

        return pred

######################################################################
##                    Prepare Tensors for PyTorch                   ##
######################################################################

def prepare_tensors(df, user_feat_df=None, item_feat_df=None, device='cpu'):

    """
    Prepares tensors for model.
    Automatically maps userId/movieId to consecutive indices.
    """

    # Build mappings
    user2idx = {uid: i for i, uid in enumerate(df['userId'].unique())}
    item2idx = {iid: i for i, iid in enumerate(df['movieId'].unique())}

    df = df.copy()
    df['user_idx'] = df['userId'].map(user2idx)
    df['item_idx'] = df['movieId'].map(item2idx)

    # Convert to tensors
    user_ids = torch.LongTensor(df['user_idx'].values).to(device)
    item_ids = torch.LongTensor(df['item_idx'].values).to(device)
    ratings = torch.FloatTensor(df['rating'].values).to(device) if 'rating' in df.columns else None

    # Align features row-wise
    user_features = torch.FloatTensor(user_feat_df.values[df['user_idx'].values]).to(device) if user_feat_df is not None else None
    item_features = torch.FloatTensor(item_feat_df.values[df['item_idx'].values]).to(device) if item_feat_df is not None else None

    return user_ids, item_ids, ratings, user_features, item_features


######################################################################
##                     Train Model Function                         ##
######################################################################

def train_model(model, train_df, user_feat_df=None, item_feat_df=None,
                n_epochs=20, lr=0.001, device='cpu'):
    """
    External training loop
    - Allows epoch-level control for validation tracking
    - Returns trained model and ID mappings
    """
    # Prepare tensors for training
    user_ids, item_ids, ratings, user_features, item_features = prepare_tensors(
        train_df, user_feat_df, item_feat_df, device
    )

    # Set model to training mode
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(user_ids, item_ids, user_features, item_features)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}")

    return model

######################################################################
##                  Predict + Validation Function                   ##
######################################################################

def predict_val(model, df, user_feat_df=None, item_feat_df=None,
                device='cpu'):
    
    """
    Predict ratings and compute RMSE in one unified function
    - Returns predictions (numpy array) and RMSE
    """

    # min_max ratings for clipping
    rating_min, rating_max=0.5, 5.0

    # Set model to evaluation mode
    model.eval()
    user_ids, item_ids, ratings, user_features, item_features = prepare_tensors(
        df, user_feat_df, item_feat_df, device
    )

    with torch.no_grad():
        preds = model(user_ids, item_ids, user_features, item_features).cpu().numpy()
    
    preds = np.clip(preds, rating_min, rating_max)

    rmse = None
    if ratings is not None:
        rmse = np.sqrt(np.mean((preds - ratings.cpu().numpy()) ** 2))

    return preds, rmse