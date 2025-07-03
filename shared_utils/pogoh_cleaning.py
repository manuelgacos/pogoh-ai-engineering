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


def _check_start_after_end(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date"
) -> pd.Series:
    """Checks for rows where the start time is after the end time.

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
        Boolean Series indicating rows where start is after end.
    """
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")
    return start > end


def _check_datetime_out_of_range(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    min_year: int = 2020,
    max_year: int = 2025,
    min_month: int = 1,
    max_month: int = 12
) -> pd.Series:
    """Checks whether start or end timestamps fall outside of the allowed year
    and month range. Months are labeled as integers 1-12, 1 = January and
    12 = December.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    start_col : str, optional
        Name of the start timestamp column, by default "start_date"
    end_col : str, optional
        Name of the end timestamp column, by default "end_date"
    min_year : int, optional
        Minimum allowed year (inclusive), by default 2020
    max_year : int, optional
        _description_, by default 2025
    min_month : int, optional
        Minimum allowed month (1 = January), by default 1
    max_month : int, optional
        Maximum allowed month (12 = December), by default 12

    Returns
    -------
    pd.Series
        Boolean Series indicating rows with start or end dates outside the
        allowed range.
    """
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")

    year_range_ok = (start.dt.year.between(min_year, max_year) &
                     end.dt.year.between(min_year, max_year))
    month_range_ok = (start.dt.month.between(min_month, max_month) &
                      end.dt.month.between(min_month, max_month))

    return ~(year_range_ok & month_range_ok)


def _check_duration_mismatch(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    duration_col: str = "duration",
    tolerance_sec: float = 1.0
) -> pd.Series:
    """Checks for rows where the reported duration doesn't match the computed
    duration (end - start), within a specified tolerance -in seconds.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    start_col : str, optional
        Name of the start timestamp column, by default "start_date", by
        default "start_date"
    end_col : str, optional
        Name of the end timestamp column, by default "end_date", by default
        "end_date"
    duration_col : str, optional
        Name of the duration column (in seconds), by default "duration", by
        default "duration"
    tolerance_sec : float, optional
        Allowed absolute difference in seconds between recorded and computed
        duration, by default 1.0

    Returns
    -------
    pd.Series
        Boolean Series indicating mismatched durations.
    """
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")
    computed = (end - start).dt.total_seconds()
    return (df[duration_col] - computed).abs() > tolerance_sec


def validate_trip_data(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    duration_col: str = "duration",
    max_duration_sec: int = 86400,
    tolerance_sec: float = 1.0,
    min_year: int = 2020,
    max_year: int = 2025,
    min_month: int = 1,
    max_month: int = 12
) -> dict:
    """Runs a series of validation checks on trip data columns and returns a
    summary of issues found.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    start_col : str, optional
        Name of the start timestamp column, by default "start_date"
    end_col : str, optional
        Name of the end timestamp column, by default "end_date"
    duration_col : str, optional
        Name of the trip duration column (in seconds), by default "duration"
    max_duration_sec : int, optional
        Maximum allowable trip duration in seconds (default is
        86400 seconds = 1 day), by default 86400
    tolerance_sec : float, optional
        Tolerance when comparing computed and recorded duration (in seconds),
        by default 1.0
    min_year : int, optional
        Minimum valid year for timestamps, by default 2020
    max_year : int, optional
        Maximum valid year for timestamps, by default 2025
    min_month : int, optional
        Minimum valid month for timestamps, by default 1 (January)
    max_month : int, optional
        Maximum valid month for timestamps, by default 12 (December)

    Returns
    -------
    dict
        Dictionary mapping check names to the number of rows that failed.
    """
    summary = {}

    summary["negative_or_zero_duration"] = _check_negative_or_zero_duration(
        df, duration_col).sum()
    summary["unrealistic_duration"] = _check_unrealistic_duration(
        df, duration_col, max_seconds=max_duration_sec).sum()
    summary["missing_timestamps"] = _check_missing_timestamps(
        df, start_col, end_col).sum()
    summary["start_after_end"] = _check_start_after_end(
        df, start_col, end_col).sum()
    summary["datetime_out_of_range"] = _check_datetime_out_of_range(
        df, start_col, end_col, min_year, max_year, min_month, max_month).sum()
    summary["duration_mismatch"] = _check_duration_mismatch(
        df, start_col, end_col, duration_col, tolerance_sec).sum()

    return summary


def summarize_trip_validation(summary_dict: dict) -> None:
    """
    Prints a plain-text summary of trip data validation results done with the
    validate_trip_data() function.

    Parameters
    ----------
    summary_dict : dict
        Dictionary from validate_trip_data() mapping issue names to row counts.
    """
    print("Trip Data Validation Summary:")
    print("-" * 35)

    for check, count in summary_dict.items():
        if count == 0:
            print(f"[OK]    {check.replace('_', ' ').capitalize()}: no issues found.")
        else:
            print(f"[FAIL]  {check.replace('_', ' ').capitalize()}: {count} issue(s) found.")

    print("-" * 35)


def get_trip_data_issues(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    duration_col: str = "duration",
    max_duration_sec: int = 86400,
    tolerance_sec: float = 1.0,
    min_year: int = 2020,
    max_year: int = 2025,
    min_month: int = 1,
    max_month: int = 12,
    include_full_data: bool = True
) -> pd.DataFrame:
    """Returns a DataFrame of rows with validation issues, optionally
    including the full original data alongside boolean flag columns
    indicating failed checks.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    start_col : str, optional
        Name of the start timestamp column, by default "start_date"
    end_col : str, optional
        Name of the end timestamp column, by default "end_date"
    duration_col : str, optional
        Name of the trip duration column (in seconds), by default "duration"
    max_duration_sec : int, optional
        Maximum allowable trip duration in seconds, by default 86400
    tolerance_sec : float, optional
        Tolerance when comparing computed and recorded duration (in seconds),
        by default 1.0
    min_year : int, optional
        Minimum valid year for timestamps, by default 2020
    max_year : int, optional
        Maximum valid year for timestamps, by default 2025
    min_month : int, optional
        Minimum valid month for timestamps, by default 1
    max_month : int, optional
        Maximum valid month for timestamps, by default 12
    include_full_data : bool, optional
        If True, return original data and flags; else return only flags and
        index, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the rows with one or more validation issues.
        Includes either the full original data or only the invalid flag
        columns.
    """
    flags = pd.DataFrame(index=df.index)

    flags["invalid_negative_or_zero_duration"] = _check_negative_or_zero_duration(df, duration_col)
    flags["invalid_unrealistic_duration"] = _check_unrealistic_duration(df, duration_col, max_seconds=max_duration_sec)
    flags["invalid_missing_timestamps"] = _check_missing_timestamps(df, start_col, end_col)
    flags["invalid_start_after_end"] = _check_start_after_end(df, start_col, end_col)
    flags["invalid_datetime_out_of_range"] = _check_datetime_out_of_range(df, start_col, end_col, min_year, max_year, min_month, max_month)
    flags["invalid_duration_mismatch"] = _check_duration_mismatch(df, start_col, end_col, duration_col, tolerance_sec)

    # Identify rows with at least one failure
    rows_with_issues = flags.any(axis=1)

    if include_full_data:
        result = df.copy()
        result["original_index"] = result.index
        for col in flags.columns:
            result[col] = flags[col]
        result = result[rows_with_issues]
    else:
        result = flags[rows_with_issues].copy()
        result["original_index"] = result.index

    # Reset index for usability (preserving original_index for traceability)
    return result.reset_index(drop=True)


def clean_trip_data(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    duration_col: str = "duration",
    fix_duration: bool = True,
    drop_other_issues: bool = True,
    max_duration_sec: int = 86400,
    tolerance_sec: float = 1.0,
    min_year: int = 2020,
    max_year: int = 2025,
    min_month: int = 1,
    max_month: int = 12
) -> pd.DataFrame:
    """Cleans trip data by fixing or dropping rows based on validation checks.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    start_col : str, optional
        Start timestamp column name, by default "start_date"
    end_col : str, optional
        End timestamp column name, by default "end_date"
    duration_col : str, optional
        Trip duration column name (in seconds), by default "duration"
    fix_duration : bool, optional
        If True, recompute duration where mismatched, by default True
    drop_other_issues : bool, optional
        If True, drop rows with other validation issues (negative or zero
        duration, unrealistic duration, missing timestamps, start after end
        date, dates out of range), by default True
    max_duration_sec : int, optional
        Maximum valid duration (in seconds), by default 86400
    tolerance_sec : float, optional
        Allowed difference between recorded and computed duration (in
        seconds), by default 1.0
    min_year : int, optional
        Minimum valid year for timestamps, by default 2020
    max_year : int, optional
        Maximum valid year for timestamps, by default 2025
    min_month : int, optional
        Minimum valid month for timestamps, by default 1
    max_month : int, optional
        Maximum valid month for timestamps, by default 12

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df_clean = df.copy()
    df_clean["original_index"] = df_clean.index

    # Compute duration mismatch mask
    mismatch_mask = _check_duration_mismatch(
        df_clean, start_col, end_col, duration_col, tolerance_sec
    )

    if fix_duration:
        # Recompute where mismatched
        start = pd.to_datetime(df_clean[start_col], errors="coerce")
        end = pd.to_datetime(df_clean[end_col], errors="coerce")
        new_duration = (end - start).dt.total_seconds()
        df_clean.loc[mismatch_mask, duration_col] = new_duration[mismatch_mask]
        print(f"Fixed duration in {mismatch_mask.sum()} row(s).")

    if drop_other_issues:
        # Build combined mask for other issues
        mask_drop = (
            _check_negative_or_zero_duration(df_clean, duration_col)
            | _check_unrealistic_duration(df_clean, duration_col,
                                          max_seconds=max_duration_sec)
            | _check_missing_timestamps(df_clean, start_col, end_col)
            | _check_start_after_end(df_clean, start_col, end_col)
            | _check_datetime_out_of_range(df_clean, start_col, end_col,
                                           min_year, max_year, min_month,
                                           max_month)
        )
        n_drop = mask_drop.sum()
        df_clean = df_clean[~mask_drop]
        print(f"Dropped {n_drop} row(s) with unrecoverable issues.")

    return df_clean.reset_index(drop=True)


def _flag_duration_outliers(
    df: pd.DataFrame,
    duration_col: str = "duration",
    min_duration_sec: int = 60,
    max_duration_sec: int = 86400
) -> pd.DataFrame:
    """Adds flags for short and long duration outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    duration_col : str, optional
        Name of duration column, by default "duration"
    min_duration_sec : int, optional
        Minimum duration threshold, by default 60
    max_duration_sec : int, optional
        Maximum duration threshold, by default 86400

    Returns
    -------
    pd.DataFrame
        DataFrame with added outlier flags:
        - is_short_duration_outlier
        - is_long_duration_outlier
    """
    df["is_short_duration_outlier"] = df[duration_col] < min_duration_sec
    df["is_long_duration_outlier"] = df[duration_col] > max_duration_sec
    return df


def flag_trip_outliers(
    df: pd.DataFrame,
    duration_col: str = "duration",
    min_duration_sec: int = 60,
    max_duration_sec: int = 86400
) -> pd.DataFrame:
    """Adds outlier flags for bikeshare trips. Currently supports:
    - Short duration outliers
    - Long duration outliers

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    duration_col : str, optional
        Name of duration column in seconds, by default "duration"
    min_duration_sec : int, optional
        Threshold below which a trip is flagged as short duration outlier, by
        default 60
    max_duration_sec : int, optional
        Threshold above which a trip is flagged as long duration outlier, by
        default 86400

    Returns
    -------
    pd.DataFrame
        DataFrame with added outlier flags.
    """
    df_out = df.copy()

    df_out = _flag_duration_outliers(
        df_out,
        duration_col=duration_col,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec
    )

    # In the future, other outlier flaggers could be added here

    return df_out


def summarize_trip_outliers(df: pd.DataFrame) -> None:
    """
    Prints a plain-text summary of outlier flags in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by `flag_trip_outliers()`, containing outlier flags.
    """
    print("Trip Outlier Summary:")
    print("-" * 30)

    if "is_short_duration_outlier" in df.columns:
        n_short = df["is_short_duration_outlier"].sum()
        print(f"[{'OK' if n_short == 0 else 'FLAG'}] Short duration outliers: {n_short}")

    if "is_long_duration_outlier" in df.columns:
        n_long = df["is_long_duration_outlier"].sum()
        print(f"[{'OK' if n_long == 0 else 'FLAG'}] Long duration outliers: {n_long}")

    # Future: other outlier types

    print("-" * 30)


def add_time_features(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    duration_col: str = "duration"
) -> pd.DataFrame:
    """Adds time-derived features to the DataFrame:
    - trip_duration_min
    - start_hour
    - end_hour
    - start_weekday
    - is_weekend

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    start_col : str, optional
        Column with start timestamps, by default "start_date"
    end_col : str, optional
        Column with end timestamps, by default "end_date"
    duration_col : str, optional
        Column with trip duration in seconds, by default "duration"

    Returns
    -------
    pd.DataFrame
        DataFrame with added features.
    """
    df_out = df.copy()

    start = pd.to_datetime(df_out[start_col], errors="coerce")
    end = pd.to_datetime(df_out[end_col], errors="coerce")

    df_out["trip_duration_min"] = df_out[duration_col] / 60
    df_out["start_hour"] = start.dt.hour
    df_out["end_hour"] = end.dt.hour
    df_out["start_weekday"] = start.dt.weekday
    df_out["is_weekend"] = df_out["start_weekday"] >= 5  # 5=Saturday, 6=Sunday

    return df_out
