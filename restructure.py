import argparse
import os
import shutil
from pathlib import Path


def restructure_folders(src: Path, dst: Path):
    """
    Restructure source folders containing MRI, CT, and mask images into a new folder structure.

    For example,
    The input folders should have the following structure:
    1BAxxx:
    - mr.nii.gz
    - mask.nii.gz
    2BAxxx:
    - cbct.nii.gz
    - mask.nii.gz

    The output folder structure will be as follows:
    images:
    - mri:
        - 1BAxxx.nii.gz
    - body:
        - 1BAxxx.nii.gz
        - 2BAxxx.nii.gz
    - cbct:
        - 2BAxxx.nii.gz

    :param src: Source base directory Path object
    :param dst: Destination base directory Path object
    """
    # Check if the destination base directories exist; if not, create them
    (dst / "images" / "mri").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "body").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "ct").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "cbct").mkdir(parents=True, exist_ok=True)

    # Iterate through the directories in the source base directory
    for subdir in src.iterdir():
        if subdir.is_dir():
            prefix = (
                subdir.name
            )  # Set prefix to the current subdirectory (1BAxxx or 2BAxxx, ...)

            # Iterate through the files in the current subdirectory
            for filename in subdir.glob("*"):
                new_filename = f"{prefix}.nii.gz"  # Create a new filename based on the subdirectory, e.g., "1BAxxx.nii.gz"
                src_path = filename
                subfolder = ""

                # Determine the target subfolder based on the file name
                if filename.name == "mr.nii.gz":
                    subfolder = "mri"
                elif filename.name == "ct.nii.gz":
                    subfolder = "ct"
                elif filename.name == "cbct.nii.gz":
                    subfolder = "cbct"
                elif filename.name == "mask.nii.gz":
                    subfolder = "body"

                # Move and rename the file
                if subfolder:
                    dst_path = dst / "images" / subfolder / new_filename
                    shutil.copy(str(src_path), str(dst_path))


def main():
    """
    Parse command-line arguments and call the restructure_folders function.
    """
    parser = argparse.ArgumentParser(
        description="Restructure folders containing MRI, CT, and mask images."
    )
    parser.add_argument("src", type=Path, help="Source base directory")
    parser.add_argument("dst", type=Path, help="Destination base directory")
    args = parser.parse_args()

    restructure_folders(args.src, args.dst)


if __name__ == "__main__":
    main()
