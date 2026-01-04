
####################################
#           Data Analysis          #
####################################


# Load libraries required #
###########################
import pandas as pd






# Dataframe quality check #
###########################

def data_quality_summary(df, name="DataFrame"):
    """
    Comprehensive data quality report for any dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    name : str
        Name of the dataframe for display purposes
    """
    print(f"\n{'='*70}")
    print(f"DATA QUALITY REPORT: {name}")
    print(f"{'='*70}")
    
    # Basic info
    print(f"\nBASIC INFO:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column-level summary
    print(f"\n COLUMN SUMMARY:")
    summary = pd.DataFrame({
        'Type': df.dtypes,
        'Non-Null': df.count(),
        'Null': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique': df.nunique(),
        'Unique %': (df.nunique() / len(df) * 100).round(2)
    })
    print(summary.to_string())
    
    # Duplicates
    n_duplicates = df.duplicated().sum()
    print(f"\n DUPLICATES:")
    print(f"  Total duplicate rows: {n_duplicates:,} ({n_duplicates/len(df)*100:.2f}%)")
    
    print(f"\n{'='*70}\n")
    
    return summary