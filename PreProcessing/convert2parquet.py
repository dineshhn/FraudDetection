import pandas as pd
import pyarrow


data = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\Financial_datasets_log.csv")
data.to_parquet(r'C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\output_data\Financial_datasets_log1.parquet', index=False)
print("Converted the files to Parquet successfully!")

#When reading both actual file and synthetic file

# import pandas as pd
# import glob
# import os
#
# csv_directory = r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive"
# csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
# df_list = [pd.read_csv(file) for file in csv_files]
# combined_df = pd.concat(df_list, ignore_index=True)
# combined_df.to_parquet('sample_findata.parquet', index=False)
