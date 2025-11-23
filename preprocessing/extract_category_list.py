import pyarrow.parquet as pq

table = pq.read_table("processed/venues_master.parquet")

print("Schema:")
print(table.schema)

categories_column = table.column("categories")

print("\nFirst 5 categories (Arrow ListArray):")
for i in range(5):
    print(i, categories_column[i].as_py())
