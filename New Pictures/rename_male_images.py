import os
import re

ROOT_FOLDER = "male"
START_NUMBER = 6
END_NUMBER = 53

def rename_images():
    for subfolder in sorted(os.listdir(ROOT_FOLDER)):

        # Leave this folder untouched
        if subfolder == "male_original":
            print(f"Skipping folder: {subfolder}")
            continue

        subfolder_path = os.path.join(ROOT_FOLDER, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nProcessing folder: {subfolder}")

        # Only process image files
        files = sorted([
            f for f in os.listdir(subfolder_path)
            if os.path.isfile(os.path.join(subfolder_path, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(files) > END_NUMBER - START_NUMBER + 1:
            print(f"WARNING: Too many images in {subfolder}. Skipping folder.")
            continue

        # First rename to temporary names to avoid filename collisions
        temp_files = []

        for index, filename in enumerate(files):
            old_path = os.path.join(subfolder_path, filename)

            name, extension = os.path.splitext(filename)

            # Remove the entire original numeric prefix
            match = re.match(r"^\d+(.*)$", name)

            if not match:
                print(f"WARNING: No numeric prefix found: {filename}")
                continue

            suffix = match.group(1)

            new_number = START_NUMBER + index
            new_filename = f"male{new_number}{suffix}{extension}"

            temp_path = os.path.join(
                subfolder_path,
                f"__temp_rename_{index}{extension}"
            )

            os.rename(old_path, temp_path)
            temp_files.append((temp_path, new_filename))

        # Rename temporary files to final names
        for temp_path, new_filename in temp_files:
            new_path = os.path.join(subfolder_path, new_filename)

            os.rename(temp_path, new_path)
            print(f"{os.path.basename(temp_path)} -> {new_filename}")

    print("\nRenaming completed!")


if __name__ == "__main__":
    rename_images()