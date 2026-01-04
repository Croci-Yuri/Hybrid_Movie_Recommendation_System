######################################################################
##                         DATA CLEANING                            ##          
######################################################################


# Sampling users from rating dataframe #
########################################

def sample_users(rating_df, n_users=40000, random_state=42):

    """Sample n random users from ratings dataframe and return filtered dataframe."""
    
    selected_users = rating_df['userId'].drop_duplicates().sample(
        n=n_users, random_state=random_state
    )
    return rating_df[rating_df['userId'].isin(selected_users)]




# Iterative filtering to ensure minimum rating number  for movies and users #
#############################################################################

def iterative_filter(rating_df, min_user_ratings=20, min_movie_ratings=25, max_iter=20):
    
    """
    Iteratively filter users and movies to meet minimum rating thresholds.
    
    Parameters:
    -----------
    rating_df : pd.DataFrame
        Ratings dataframe with userId and movieId columns
    min_user_ratings : int
        Minimum number of ratings per user
    min_movie_ratings : int
        Minimum number of ratings per movie
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    pd.DataFrame : Filtered ratings dataframe
    dict : Statistics about the filtering process
    """

    stats = []
    
    for iteration in range(max_iter):
        n_before = len(rating_df)
        
        # Filter movies
        movie_counts = rating_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        rating_df = rating_df[rating_df['movieId'].isin(valid_movies)]
        
        # Filter users
        user_counts = rating_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        rating_df = rating_df[rating_df['userId'].isin(valid_users)]
        
        n_after = len(rating_df)
        
        stats.append({
            'iteration': iteration + 1,
            'ratings': n_after,
            'users': len(valid_users),
            'movies': len(valid_movies)
        })
        
        if n_before == n_after:
            break
            
    return rating_df, stats