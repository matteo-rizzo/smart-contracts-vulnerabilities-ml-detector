import os


def modify_solidity_files(directory):
    spdx_line = "// SPDX-License-Identifier: MIT"
    pragma_line = "pragma solidity ^0.8.0;"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sol"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Check if lines are present and count duplicates
                spdx_present = any(spdx_line in line for line in lines)
                pragma_present = any(pragma_line in line for line in lines)
                cleaned_lines = []

                added_spdx = False
                added_pragma = False

                for line in lines:
                    if spdx_line in line:
                        if not added_spdx:
                            cleaned_lines.append(spdx_line + '\n')
                            added_spdx = True
                    elif pragma_line in line:
                        if not added_pragma:
                            cleaned_lines.append(pragma_line + '\n')
                            added_pragma = True
                    else:
                        cleaned_lines.append(line)

                # Prepend missing lines if not added
                if not spdx_present:
                    cleaned_lines.insert(0, spdx_line + '\n')
                if not pragma_present:
                    # Find where to insert pragma based on spdx line placement
                    pragma_insert_index = 1 if spdx_present else 0
                    cleaned_lines.insert(pragma_insert_index, pragma_line + '\n')

                # Write the cleaned/modified lines back to the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                print(f"Processed {filepath}")


# Usage example
directory_path = 'dataset/aisc/source'
modify_solidity_files(directory_path)
