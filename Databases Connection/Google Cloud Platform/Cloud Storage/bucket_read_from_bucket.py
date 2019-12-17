import os
from google.cloud import storage
import pandas as pd
import pickle
import datetime



json_path = "C:\\Users\\pasilvacorista\\Documents\\json\\knime-project-automl.json"
bucket_name = "bucket_knime"


def unix_time_millis(dt):
    dt = dt.replace(tzinfo=None)
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000.0)


def concat_df(df, blobs):
    for blob in blobs[1:]:
        file_uri = 'gs://' + bucket_name + '/' + blob.name
        df2 = pd.read_csv(file_uri)
        df = pd.concat([df, df2], ignore_index=True)
    return df

def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    try:
        last_blob_timestamp = pickle.load(open("last_blob_timestamp.p", "rb"))
    except:
        last_blob_timestamp = 0

    blobs = [blob for blob in bucket.list_blobs() if unix_time_millis(blob.time_created) > last_blob_timestamp]

    if (len(blobs) > 0):
        file_uri = 'gs://' + bucket_name + '/' + blobs[0].name
        df = pd.read_csv(file_uri)
        df = concat_df(df,blobs)
        last_blob_timestamp = unix_time_millis(blobs[-1].time_created)
        pickle.dump(last_blob_timestamp, open("last_blob_timestamp.p", "wb"))
        print(df)
    else:
        print("No new data")
main()
