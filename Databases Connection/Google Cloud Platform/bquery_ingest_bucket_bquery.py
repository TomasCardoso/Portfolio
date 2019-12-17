from google.cloud import storage
from google.cloud import bigquery as bq
from datetime import datetime
import os
import pandas as pd
import gcsfs

def load_file_to_bigquery():

    ### get data from file:
    #file = event
    #file_name=file['name']
    file_name='iris.csv'

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\pasilvacorista\\Documents\\json\\knime-project-automl.json"
    print('################ Processing '+str(file_name)+": ")

    ingest_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('# Ingest_time: '+str(ingest_time))

    ### set storage client:
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('testetesteteste')

    file_uri = 'gs://testetesteteste/'+file_name

    df = pd.read_csv(file_uri)

    df.insert(0, 'TimeStamp', pd.datetime.now().replace(microsecond=0))

    bigquery_client = bq.Client()
    table = bq.Table('knime-project-automl.knime_dataset' + '.' + 'knime_table')
    job_config = bq.LoadJobConfig()

    job = bigquery_client.load_table_from_dataframe(df, table,job_config=job_config)
    print('Starting job {}'.format(job.job_id))
    try:
        result = job.result()
    except:
        print(job.errors)



load_file_to_bigquery()