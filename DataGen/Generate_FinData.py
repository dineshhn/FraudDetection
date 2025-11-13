import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer , CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from tabulate import tabulate

data = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\sample_csv.txt")
metadata = Metadata.detect_from_dataframe(data)

print("Original data:")
print(tabulate(data.head(5), headers='keys', tablefmt='psql'))

synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=5)

diagnostic_report = run_diagnostic(
    real_data=data,
    synthetic_data=synthetic_data,
    metadata=metadata)

quality_report = evaluate_quality(
    real_data=data,
    synthetic_data=synthetic_data,
    metadata=metadata)

print("Synthetic Data")
print(tabulate(synthetic_data, headers='keys', tablefmt='psql'))

#synthetic_data.to_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\synthetic_data1.txt")