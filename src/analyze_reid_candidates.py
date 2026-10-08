from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_FILES = {
    "day": ROOT / "data" / "reid" / "candidates" / "day" / "candidate_sequences.csv",
    "night": ROOT / "data" / "reid" / "candidates" / "night" / "candidate_sequences.csv",
}

OUTPUT_DIR = ROOT / "data" / "reid" / "metadata"


def load_candidates():

    frames = []

    for dataset, path in CANDIDATE_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Candidate file not found: {path}"
            )

        df = pd.read_csv(path)

        df["dataset"] = dataset

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main():

    print("=" * 70)
    print("VanDrishti Re-ID Candidate Analysis")
    print("=" * 70)

    df = load_candidates()

    print()
    print(f"Total candidate sequences: {len(df)}")

    print()
    print("Sequences by dataset")
    print("-" * 40)

    print(
        df["dataset"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Inspect available columns
    # -----------------------------------------------------

    print()
    print("Available columns")
    print("-" * 40)

    for column in df.columns:
        print(column)

    # -----------------------------------------------------
    # Basic sequence statistics
    # -----------------------------------------------------

    possible_length_columns = [
        "length",
        "sequence_length",
        "num_frames",
        "frame_count",
    ]

    length_column = None

    for column in possible_length_columns:
        if column in df.columns:
            length_column = column
            break

    if length_column is None:

        print()
        print(
            "WARNING: No recognized sequence-length column found."
        )

    else:

        print()
        print(
            f"Sequence length column: {length_column}"
        )

        print()
        print("Sequence length statistics")
        print("-" * 40)

        print(
            df[length_column]
            .describe()
            .to_string()
        )

        print()
        print("Longest sequences")
        print("-" * 40)

        display_columns = [
            column
            for column in [
                "dataset",
                "species",
                "start_frame",
                "end_frame",
                length_column,
            ]
            if column in df.columns
        ]

        print(
            df.sort_values(
                length_column,
                ascending=False,
            )
            .head(30)[display_columns]
            .to_string(index=False)
        )

    # -----------------------------------------------------
    # Species distribution
    # -----------------------------------------------------

    possible_species_columns = [
        "species",
        "class",
        "category",
        "label",
    ]

    species_column = None

    for column in possible_species_columns:
        if column in df.columns:
            species_column = column
            break

    if species_column:

        print()
        print("Candidate sequences by species")
        print("-" * 40)

        print(
            df[species_column]
            .value_counts()
            .to_string()
        )

    # -----------------------------------------------------
    # Save combined candidate table
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "reid_candidate_analysis.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"Combined candidate table saved:"
    )
    print(output_file)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()