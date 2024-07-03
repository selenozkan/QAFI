
# Specify the path to your ZIP file
zip_file_path = #'path/to/your/file_uniprot_id.csv.zip'

# Read the CSV file inside the ZIP archive
df = pd.read_csv(zip_file_path, compression='zip')
