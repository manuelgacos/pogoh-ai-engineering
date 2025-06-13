import pandas as pd
import string


def clean_station_text(series: pd.Series) -> pd.Series:
    """Cleans a column of string, assumed to be locations of POGOH stations, 
    and returns a cleaned version of the column.

    Parameters
    ----------
    series : pd.Series
        A column from a POGOH dataset that contains names of stations.

    Returns
    -------
    pd.Series
        A cleaned version of the input column.
    """
    # Strip whitespace at the start and end of string
    series = series.str.strip()
    # Make all characters lowercase
    series = series.str.lower()
    # Normalize inner spaces to just one space
    series = series.str.replace(r"\s+", " ", regex=True)
    # Normalize common abbreviations
    replace_dict = {
        r"\bst\b": "street",
        r"\bave\b": "avenue",
        r"\bblvd\b": "boulevard",
        r"\bdr\b": "drive",
        r"\bext\b": "extension",
        r"\bn\b": "north",
        r"\bs\b": "south",
        r"&": "and"
    }
    for key, val in replace_dict.items():
        series = series.str.replace(key, val, regex=True)
    # Remove punctuation symbols in the strings
    # NOTE: If done earlier, might erase symbols like & form the names
    series = series.str.translate(str.maketrans("","",string.punctuation))
    # Making sure that no extra spaces were introduced
    series = series.str.replace(r"\s+", " ", regex=True).str.strip()

    return series


def clean_station_names(df: pd.DataFrame,
                        col_name="start_station_name") -> pd.DataFrame:
    """Cleans and standardizes station name columns in a POGOH dataset.
    Adds a new column: '{col_name}_clean'.

    Parameters
    ----------
    df : pd.DataFrame
        A dataset from the POGOH database where the 'col_name' column
        includes station names.
    col_name : str, optional
        The name of the column to be cleaned, by default "start_station_name"

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the new cleaned column.
    """

    df = df.copy()
    df[f"{col_name}_clean"] = clean_station_text(df[col_name])

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the name of the columns of the dataset, erasing leading/trailing
    spaces, converting strings to lowercase and replacing symbols with
    underscores.

    Parameters
    ----------
    df : pd.DataFrame
        A dataset from the POGOH database.

    Returns
    -------
    pd.DataFrame
        A dataset with the column names in a standard format.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()  # remove leading/trailing spaces
                  .str.lower()  # convert to lowercase
                  .str.replace(r"[ \-\.]", "_", regex=True)  # replace spaces/symbols with underscores
                  .str.replace(r"__+", "_", regex=True)
    )

    return df
