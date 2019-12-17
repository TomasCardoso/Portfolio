from google.cloud import storage
import os
import pandas as pd
import datetime


df = pd.DataFrame(data=[{1,2,3},{4,5,6}],columns=['a','b','c'])
test_str = "wsws"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\pasilvacorista\\Documents\\json\\knime-project-automl.json"
client = storage.Client()

#bucket_name = "deploydata12345"

bucket_name = "deploy_predictions"
temp_file_location = 'C:\\Users\\pasilvacorista\\Documents\\json\\tmp.csv'
#model_bucket = client.blob.Blob(client, name=bucket_name)

#if not model_bucket.exists():
#	bucket = model_bucket
#else:
#	bucket = client.get_bucket(bucket_name)

bucket = client.get_bucket(bucket_name)


csv_prefix="predicted_"
blob_prefix = "processed_data_blob_"
process_date = datetime.datetime.now().now().strftime('%Y-%m-%d-%H:%M:%S')
csv_name = csv_prefix


blob_name = blob_prefix + process_date

print(blob_name)

blob = bucket.blob(blob_name)
df.to_csv(path_or_buf=temp_file_location)
blob.upload_from_filename(temp_file_location)


print(blob)