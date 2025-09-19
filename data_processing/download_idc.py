import idc_index.index as idx
import argparse
from inference.utils.save_outputs import ensure_empty_dir
from utils.path_obtainer import get_paths
from data_processing.process_dcm import process_dcm
import pandas as pd
import os


# Studies with non-conforming name formats or missing phases
EXCLUDED_STUDY_UIDS = [
    "1.3.6.1.4.1.14519.5.2.1.6834.5010.170605434729890793667175785576",
    "1.3.6.1.4.1.14519.5.2.1.6834.5010.195368562719946042143948478411",
    "1.3.6.1.4.1.14519.5.2.1.6834.5010.983119680047871775036619601861",
]


def create_index(
    query: str, printing: bool = False
) -> tuple[idx.IDCClient, pd.DataFrame]:
    """
    Creates an index of data based on a SQL query.

    Args:
        query (str): The SQL query to execute.
        printing (bool): If True, prints the DataFrame. Defaults to False.

    Returns:
        tuple[idc_index.IDCClient, pd.DataFrame]: A tuple containing the client and the DataFrame.
    """
    client = idx.IDCClient()

    df = client.sql_query(query)
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())

    if printing:
        print(df)

    return client, df


def create_manifest(
    df: pd.DataFrame, manifest_path: str, printing: bool = False
) -> None:
    """
    Creates a manifest file from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        manifest_path (str): The path where the manifest file will be saved.
        printing (bool): If True, prints the manifest path and file size. Defaults to False.
    """
    with open(manifest_path, "w") as f:
        f.write("\n".join(df["cp_command"].tolist()))

    if printing:
        print(f"Writing manifest to: {manifest_path}")
        print(f"Manifest file size: {os.path.getsize(manifest_path)} bytes")


def download_files(client: idx.IDCClient, manifest_path: str) -> None:
    """
    Efficiently downloads files corresponding to a manifest.

    Args:
        client (idc_index.IDCClient): The IDC client.
        manifest_path (str): The path to the manifest file.
    """
    # efficiently download the files corresponding to the manifest
    s5cmd_binary = client.s5cmdPath
    os.system(
        f"{s5cmd_binary} --no-sign-request --endpoint-url https://s3.amazonaws.com run {manifest_path}"
    )


def print_sorted_unique_values(df: pd.DataFrame, column_name: str) -> None:
    """
    Prints the sorted unique values of a column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_name (str): The name of the column.
    """
    sorted_unique_values = df[column_name].value_counts().sort_values(ascending=False)
    print(sorted_unique_values.head())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data range from IDC.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive)")
    parser.add_argument(
        "--target_vol_dimensions",
        type=tuple,
        default=(50, 256, 256),
        help="Target volume dimensions",
    )
    args = parser.parse_args()
    start = args.start
    end = args.end
    target_vol_dimensions = args.target_vol_dimensions

    dp_paths = get_paths("data_processing")
    raw_data_save_path = dp_paths["idc_downloads"]["raw"]
    processed_data_save_path = dp_paths["idc_downloads"]["processed"]
    manifest_path = os.path.join(raw_data_save_path, "download_manifest.txt")
    ensure_empty_dir(raw_data_save_path)

    # This query selects all CT series from the 4D-Lung set
    excluded_uid_rows = " UNION ALL\n          ".join(
        f"SELECT '{uid}' AS StudyInstanceUID" for uid in EXCLUDED_STUDY_UIDS
    )

    lung_4d_query = f"""
    WITH excluded AS (
      {excluded_uid_rows}
    )
    SELECT
      SeriesInstanceUID,
      ANY_VALUE(PatientID)        as PatientID,
      ANY_VALUE(StudyInstanceUID) as StudyInstanceUID,
      ANY_VALUE(CONCAT('cp ', series_aws_url, ' {raw_data_save_path}')) as cp_command,
      ANY_VALUE(series_size_MB)   as series_size_MB,
      ANY_VALUE(instanceCount)    as instance_count
    FROM index
    WHERE collection_id = '4d_lung'
      AND Modality       = 'CT'
      AND instanceCount > 49
      AND StudyInstanceUID NOT IN (SELECT StudyInstanceUID FROM excluded)
    GROUP BY SeriesInstanceUID
    ORDER BY PatientID, StudyInstanceUID, SeriesInstanceUID
    """
    # Select a subset of the data
    client, df = create_index(lung_4d_query, printing=False)

    df = df[start:end]

    print(df.head())
    total_data_size = df["series_size_MB"].sum()
    print(f"\nTotal data disk size: {total_data_size} MB")

    # Create a manifest file
    create_manifest(df, manifest_path, printing=True)

    proceed = input(
        f"Are you sure you want to proceed downloading {total_data_size} MB of data? (enter 'y' for yes): "
    )
    if proceed == "y":
        # Download the files
        print("Removing existing data...")
        ensure_empty_dir(processed_data_save_path)

        print("Downloading data...")
        download_files(client, manifest_path)

        # Organize data
        process_dcm(
            raw_data_save_path,
            processed_data_save_path,
            target_vol_dimensions=target_vol_dimensions,
        )

    else:
        print("Download sequence cancelled.")


if __name__ == "__main__":
    main()
