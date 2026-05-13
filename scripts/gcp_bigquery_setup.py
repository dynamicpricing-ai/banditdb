import os
import argparse
from google.cloud import bigquery
from google.cloud import storage

def setup_bigquery_external_table(project_id, bucket_name, dataset_id, table_id):
    """
    Sets up a BigQuery external table mapped to a GCS bucket containing 
    BanditDB Parquet exports.
    """
    client = bigquery.Client(project=project_id)

    # 1. Create Dataset if it doesn't exist
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}.")

    # 2. Configure External Table
    table = bigquery.Table(dataset_ref.table(table_id))
    
    external_config = bigquery.ExternalConfig("PARQUET")
    # Path pattern for BanditDB shards: exports/{campaign_id}/*.parquet
    uri = f"gs://{bucket_name}/exports/*/*.parquet"
    external_config.source_uris = [uri]
    external_config.autodetect = True  # BanditDB Parquet schema is stable
    
    table.external_data_configuration = external_config

    # 3. Create or Update Table
    try:
        client.delete_table(table)  # Refresh mapping
    except Exception:
        pass
        
    client.create_table(table)
    print(f"Created external table {dataset_id}.{table_id} pointing to {uri}")
    print("\nBanditDB Zero-ETL Analytics is ready. Try this query in BigQuery console:")
    print(f"SELECT arm_id, AVG(reward) as avg_reward, COUNT(*) as interactions")
    print(f"FROM `{project_id}.{dataset_id}.{table_id}`")
    print(f"GROUP BY 1 ORDER BY 2 DESC")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BanditDB BigQuery Setup")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--bucket", required=True, help="GCS Bucket Name where Parquet files live")
    parser.add_argument("--dataset", default="banditdb", help="BigQuery Dataset ID")
    parser.add_argument("--table", default="interactions_all", help="BigQuery Table ID")
    
    args = parser.parse_args()
    setup_bigquery_external_table(args.project, args.bucket, args.dataset, args.table)
