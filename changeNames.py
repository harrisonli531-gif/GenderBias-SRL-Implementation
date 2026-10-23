from pathlib import Path

def rename_images(folder_path):
    folder = Path(folder_path)

    if not folder.is_dir():
        print(f"Error: Folder does not exist: {folder}")
        return

    # Supported image extensions
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".webp", ".tiff", ".tif", ".heic"
    }

    # Collect images only; skip files already renamed
    images = sorted(
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
        and not (
            file.stem.startswith("male")
            and file.stem.endswith("_original")
        )
    )

    if not images:
        print("No images to rename.")
        return

    # Use temporary names first to avoid filename collisions
    temp_files = []

    for i, image in enumerate(images):
        temp_path = folder / f"__temp_rename_{i}{image.suffix.lower()}"
        while temp_path.exists():
            temp_path = folder / f"_{temp_path.name}"

        image.rename(temp_path)
        temp_files.append((temp_path, image.suffix.lower()))

    # Rename sequentially, starting from female6_original
    for index, (temp_path, extension) in enumerate(temp_files, start=6):
        new_path = folder / f"male{index}_original{extension}"

        if new_path.exists():
            print(f"Warning: Target already exists: {new_path}")
            continue

        temp_path.rename(new_path)
        print(f"{temp_path.name} -> {new_path.name}")

    print(f"\nFinished! Renamed {len(temp_files)} images.")


if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ").strip().strip('"')
    rename_images(folder_path)