import os
import re
import csv

def extract_and_save_to_csv(names_list, output_filename="training_results_1.csv", base_path="./logs"):
    metric_pattern = re.compile(r"^(\d+):\s*([\d\.]+),\s*([\d\.]+)")
    headers = ["Name", "File_Index", "Epoch", "MSE", "R2"]
    rows_to_write = []

    for name in names_list:
        for i in range(8):
            filename = f"1.0-Water-5-True-Projected-OPTIMIZED-{name}-1-1-qsvr-{i}.out"
            file_path = os.path.join(base_path, filename)
            
            if not os.path.exists(file_path):
                print(f"[-] Missing: {filename} ... Skipping.")
                continue
                
            recording = False
            file_rows = []
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        
                        if "Starting training of model" in line:
                            recording = True
                            continue
                        
                        if "Processing dictionary:" in line:
                            #file is valid
                            recording = False
                            rows_to_write.extend(file_rows)
                            print(f"[+] Processed: {filename}")
                            break
                        
                        if recording:
                            match = metric_pattern.search(line)
                            if match:
                                epoch, mse, r2 = match.groups()
                                file_rows.append([name, i, epoch, mse, r2])
            except Exception as e:
                print(f"[!] Error reading {filename}: {e}")

    # Write data to CSV
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows_to_write)
    
    print(f"\nDone! Saved {len(rows_to_write)} total data points to {output_filename}")

my_names = ["Hubregtsen","Chebyshev", "MultiControl", "CPMap", "YZ_CX", "ParamZ"]
extract_and_save_to_csv(my_names)
