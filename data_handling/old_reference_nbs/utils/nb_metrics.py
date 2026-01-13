"""
'nb_metrics' module for notebook metrics
"""

class Metrics:
    """
    Metrics class contains helper functions to extract measurable data from our 
    digital collections metadata
    """

    def df_type_counts(df):
        """
        Prints a string of the various different types that
        USC collections metadata contains.

        Parameters:
            df (pandas.DataFrame): Digital collections metadata DataFrame.
        """
        still = (df['Type'].str.lower() == "still image").sum()
        text = (df['Type'].str.lower() == "text").sum()
        still_text = (df['Type'].str.lower() == "still image; text").sum()
        other = (((df['Type'].str.lower() != "text") & (df['Type'].str.lower() != "still image") & (df['Type'].str.lower() != "still image; text"))).sum()
        print(f'{still} rows Type = "still image", {text} rows Type = "text", {still_text} rows "still image; text", {other} rows of other types')
        sum = still+text+still_text+other
        print(f"Sum = {sum}\n")
        return sum
    
    