import pandas as pd
import string


def _clean_station_text(series: pd.Series) -> pd.Series:
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
    df[f"{col_name}_clean"] = _clean_station_text(df[col_name])

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


def filter_valid_station_rows(
    df: pd.DataFrame,
    cast_ids_to_int: bool = True,
    start_id_col: str = "start_station_id",
    start_name_col: str = "start_station_name",
    end_id_col: str = "end_station_id",
    end_name_col: str = "end_station_name"
) -> pd.DataFrame:
    """
    Filters out rows with missing station ID or name fields.
    Optionally casts station ID columns to integers.

    Parameters
    ----------
    df : pd.DataFrame
        The input POGOH dataset.

    cast_ids_to_int : bool, default=True
        Whether to cast ID columns to integers after filtering.

    Returns
    -------
    pd.DataFrame
        A filtered and optionally type-converted DataFrame.
    """
    df = df.copy()

    # Drop rows with any missing station ID or name
    df = df.dropna(subset=[start_id_col, start_name_col, end_id_col,
                           end_name_col])

    if cast_ids_to_int:
        df[start_id_col] = df[start_id_col].astype(int)
        df[end_id_col] = df[end_id_col].astype(int)

    return df


def select_invalid_station_rows(
    df: pd.DataFrame,
    start_id_col: str = "start_station_id",
    start_name_col: str = "start_station_name",
    end_id_col: str = "end_station_id",
    end_name_col: str = "end_station_name"
) -> pd.DataFrame:
    """
    Returns all rows where any station ID or name is missing.

    Parameters
    ----------
    df : pd.DataFrame
        The input POGOH dataset.

    start_id_col : str, default="start_station_id"
    start_name_col : str, default="start_station_name"
    end_id_col : str, default="end_station_id"
    end_name_col : str, default="end_station_name"

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only rows with missing values
        in any of the station ID or name columns.
    """
    mask = df[[start_id_col, start_name_col, end_id_col,
               end_name_col]].isnull().any(axis=1)
    return df[mask].copy()


def _check_station_mapping(df: pd.DataFrame, id_col: str,
                           name_col: str) -> pd.DataFrame:
    """
    Returns (id, name) pairs where a single station ID maps to more than one
    name.

    Parameters
    ----------
    df : pd.DataFrame
        The input POGOH dataset, containing station ID and name columns.
    id_col : str
        The name of the column with station IDs.
    name_col : str
        The name of the column with station names.

    Returns
    -------
    pd.DataFrame
        A DataFrame with duplicated ID-to-name mappings (if any).
        Empty if clean.
    """
    mapping_counts = df.groupby(id_col)[name_col].nunique()
    conflicted_ids = mapping_counts[mapping_counts > 1].index

    return (
        df[df[id_col].isin(conflicted_ids)]
        [[id_col, name_col]]
        .drop_duplicates()
        .sort_values(id_col)
        .reset_index(drop=True)
    )


def _check_cross_direction_id_name_consistency(
    df: pd.DataFrame,
    start_id_col: str = "start_station_id",
    start_name_col: str = "start_station_name",
    end_id_col: str = "end_station_id",
    end_name_col: str = "end_station_name"
) -> dict:
    """
    Checks that the set of (station_id, station_name) pairs is consistent
    between start and end stations.

    Returns a dictionary with any mismatched pairs.

    Returns
    -------
    dict with keys:
        'start_only_pairs': DataFrame of (id, name) pairs only found in start
        'end_only_pairs': DataFrame of (id, name) pairs only found in end
    """
    start_pairs = df[[start_id_col, start_name_col]].drop_duplicates().rename(
        columns={start_id_col: "station_id", start_name_col: "station_name"}
    )
    end_pairs = df[[end_id_col, end_name_col]].drop_duplicates().rename(
        columns={end_id_col: "station_id", end_name_col: "station_name"}
    )

    start_only = pd.merge(start_pairs, end_pairs, how="left", indicator=True).query('_merge == "left_only"').drop(columns="_merge")
    end_only = pd.merge(end_pairs, start_pairs, how="left", indicator=True).query('_merge == "left_only"').drop(columns="_merge")

    return {
        "start_only_pairs": start_only.reset_index(drop=True),
        "end_only_pairs": end_only.reset_index(drop=True)
    }


def check_station_id_name_consistency(
    df: pd.DataFrame,
    start_id_col: str = "start_station_id",
    start_name_col: str = "start_station_name",
    end_id_col: str = "end_station_id",
    end_name_col: str = "end_station_name"
) -> dict:
    """
    Runs consistency checks on station ID/name pairs.
    Returns a summary indicating whether inconsistencies exist.

    Returns
    -------
    dict
        {
            "status": bool,  # True = clean, False = has issues
            "issues": {
                "start_conflicts": DataFrame,
                "end_conflicts": DataFrame,
                "start_only_pairs": DataFrame,
                "end_only_pairs": DataFrame
            }
        }
    """
    start_conflicts = _check_station_mapping(df, start_id_col, start_name_col)
    end_conflicts = _check_station_mapping(df, end_id_col, end_name_col)
    cross_conflicts = _check_cross_direction_id_name_consistency(
        df, start_id_col, start_name_col, end_id_col, end_name_col
    )

    has_issues = (
        not start_conflicts.empty
        or not end_conflicts.empty
        or not cross_conflicts["start_only_pairs"].empty
        or not cross_conflicts["end_only_pairs"].empty
    )

    return {
        "status": not has_issues,
        "issues": {
            "start_conflicts": start_conflicts,
            "end_conflicts": end_conflicts,
            "start_only_pairs": cross_conflicts["start_only_pairs"],
            "end_only_pairs": cross_conflicts["end_only_pairs"]
        } if has_issues else {}
    }


def print_station_consistency_summary(result: dict) -> None:
    """
    Prints a summary of the results from a station ID-name consistency check.

    This function is designed to be used alongside the output of
    `check_station_id_name_consistency()`. It prints a user-friendly message
    indicating whether the dataset passes all ID-name mapping checks, and if
    not, provides counts of the different types of mismatches.

    Parameters
    ----------
    result : dict
        A dictionary returned by `check_station_id_name_consistency()`, with
        keys:
        - 'status' (bool): True if data is clean; False otherwise.
        - 'issues' (dict): Contains DataFrames for each category of conflict:
            - 'start_conflicts'
            - 'end_conflicts'
            - 'start_only_pairs'
            - 'end_only_pairs'

    Returns
    -------
    None
        This function prints output to the console and does not return
        anything.
    """
    if result["status"]:
        print(" Station ID-name mapping is consistent across start and end stations.")
    else:
        issues = result["issues"]
        print(" Inconsistencies detected in station ID-name pairs:\n")
        print(f"- {len(issues['start_conflicts'])} conflicting ID-name mappings in START stations")
        print(f"- {len(issues['end_conflicts'])} conflicting ID-name mappings in END stations")
        print(f"- {len(issues['start_only_pairs'])} ID-name pairs appear only in START")
        print(f"- {len(issues['end_only_pairs'])} ID-name pairs appear only in END")
        print("\nPlease inspect the 'issues' dictionary for more details.")


def check_duration_mismatch(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    duration_col: str = "duration",
    tolerance_sec: float = 1.0
) -> pd.DataFrame:
    """Checks whether the 'duration' column matches the computed duration 
    from timestamps. Always prints a message summarizing the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    start_col : str, optional
        Name of start datetime column, by default "start_date"
    end_col : str, optional
        Name of end datetime column, by default "end_date"
    duration_col : str, optional
        Name of existing duration column (in seconds), by default "duration"
    tolerance_sec : float, optional
        Allowed absolute difference (in seconds), by default 1.0

    Returns
    -------
    pd.DataFrame
        DataFrame with missmatching information
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

    df["computed_duration"] = (df[end_col] - df[start_col]).dt.total_seconds()
    mismatch = (df[duration_col] - df["computed_duration"]).abs() > tolerance_sec
    num_mismatches = mismatch.sum()

    if num_mismatches == 0:
        print("No duration mismatches found.")
    else:
        print(f"Duration mismatch found in {num_mismatches} row(s).")

    return df[mismatch].reset_index(drop=True)


def _check_negative_or_zero_duration(
    df: pd.DataFrame,
    duration_col: str = "duration"
) -> pd.Series:
    """Checks for rows with zero or negative duration values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    duration_col : str, optional
        Name of the column representing trip duration (in seconds),
        by default "duration"

    Returns
    -------
    pd.Series
        Boolean Series indicating rows with non-positive durations.
    """
    return df[duration_col] <= 0


def _check_unrealistic_duration(
    df: pd.DataFrame,
    duration_col: str = "duration",
    max_seconds: int = 86400
) -> pd.Series:
    """Checks for rows with particularly long durations, threshold can be
    set by the user. The default limit is 86400 seconds or 1 day.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    duration_col : str, optional
        Name of the duration column (in seconds), by default "duration"
    max_seconds : int, optional
        Maximum allowable duration in seconds, by default 86400

    Returns
    -------
    pd.Series
        Boolean Series indicating rows with overly long durations.
    """
    return df[duration_col] > max_seconds


def _check_missing_timestamps(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date"
) -> pd.Series:
    """Checks for rows with missing start or end timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    start_col : str, optional
        Name of the start timestamp column, by default "start_date"
    end_col : str, optional
        Name of the end timestamp column, by default "end_date"

    Returns
    -------
    pd.Series
        Boolean Series indicating rows with missing timestamps.
    """
    return df[start_col].isna() | df[end_col].isna()
