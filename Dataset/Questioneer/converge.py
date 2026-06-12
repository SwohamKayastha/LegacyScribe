import os

def merge_jsonl_files():
    # Define the output file name
    output_filename = "combined_dataset.jsonl"
    
    # Generate the list of files to merge based on image_b1e467.png
    input_files = [f"batch{i}.jsonl" for i in range(1, 14)]
    
    print(f"🚀 Starting merge process into '{output_filename}'...")
    
    # Track total lines written for a quick sanity check
    total_lines = 0
    
    # Open the consolidated file in write mode
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for file_name in input_files:
            if os.path.exists(file_name):
                print(f"  -> Reading {file_name}...")
                with open(file_name, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # Ensure we don't copy over accidental blank lines
                        if line.strip():
                            outfile.write(line.strip() + '\n')
                            total_lines += 1
            else:
                print(f"  ⚠️ Warning: {file_name} was not found. Skipping.")
                
    print(f"\n🎉 Done! Successfully merged files into '{output_filename}' ({total_lines} total rows).")

if __name__ == "__main__":
    merge_jsonl_files()