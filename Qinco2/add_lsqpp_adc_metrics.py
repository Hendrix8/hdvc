import pandas as pd
import glob
import os

lsqpp_dir = "/mnthdd/cpanourg/2-hdvc/results/lsqpp/lsqpp_grid_M2-7-15_nbits4-8-12_train10k-100k_20260501_232731"
csv_files = glob.glob(os.path.join(lsqpp_dir, "*_LSQpp_adc_vs_exact_eval.csv"))

out_dir = "/home/cpanourg/projects/2-hdvc/Qinco2"

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # Calculate per_pair_adc_time_ns
    # Assuming distance_table_time_s and adc_time_s are the total times for nq queries over nb_sample database vectors
    if "per_pair_adc_time_ns" not in df.columns:
        df["per_pair_adc_time_ns"] = ((df["distance_table_time_s"] + df["adc_time_s"]) / (df["nq"] * df["nb_sample"])) * 1e9
    
    basename = os.path.basename(csv_file)
    out_path = os.path.join(out_dir, basename)
    df.to_csv(out_path, index=False)
    print(f"Processed {basename} and saved to {out_path}")
