###########################################################################################
# Script that reads from a Google Cloud storage bucket and writes it to a bigQuery table. #
# Next it reades from that table and stores it to a dataframe															#
###########################################################################################

### import packages:
import pandas as pd
import os
import gcsfs
from google.cloud import storage
from google.cloud.storage import blob
from google.cloud import bigquery
from datetime import datetime
import pickle

json_path = "C:\\Users\\pasilvacorista\\Documents\\json\\knime-project-automl.json"


def bq_create_dataset(bigquery_client, dataset_name):
    # CREATE DATASET IF NOT EXISTS
    dataset_ref = bigquery_client.dataset(dataset_name)
    try:
        bigquery_client.get_dataset(dataset_ref)
    except:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = 'europe-north1'
        dataset = bigquery_client.create_dataset(dataset)
        print('Dataset {} created.'.format(dataset.dataset_id))


def load_file_to_bigquery():
    ### get data from file:
    # file = event
    # file_name=file['name']
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path

    file_name = 'iris.csv'
    print('################ Processing ' + str(file_name) + ": ")
    bigquery_client = bigquery.Client()
    ingest_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    ### set storage client:
    storage_client = storage.Client()
    bucket_name = 'testetesteteste'
    bucket = storage_client.get_bucket(bucket_name)

    file_uri = 'gs://' + bucket_name + '/' + file_name

    df = pd.read_csv(file_uri)

    df.insert(0, 'TimeStamp', pd.datetime.now().replace(microsecond=0))

    ## Write to BigQuery
    # Get bigQuery client
    bigquery_client = bigquery.Client()

    # Define Dataset and Table names
    bigquery_dataset_name = 'knime_dataset'
    bigquery_table_name = 'test_table'

    # Try to get dataset, if it does not exist, create one
    try:
        dataset_ref = bigquery_client.dataset(bigquery_dataset_name)
        dataset = bigquery_client.get_dataset(dataset_ref)
    except:
        bq_create_dataset(bigquery_client, bigquery_dataset_name)

    # Try to get table, if it does not exist, create one
    try:
        table_ref = dataset_ref.table(bigquery_table_name)
        table = bigquery_client.get_table(table_ref)
    except:
        # Define schema for the new table
        schema = [
            bigquery.SchemaField('TimeStamp', 'TIMESTAMP'),
            bigquery.SchemaField('sepal_length', 'FLOAT'),
            bigquery.SchemaField('sepal_width', 'FLOAT'),
            bigquery.SchemaField('petal_length', 'FLOAT'),
            bigquery.SchemaField('petal_width', 'FLOAT'),
            bigquery.SchemaField('species', 'STRING')
        ]

        # Create a nre table
        table = bigquery.Table(table_ref, schema=schema)
        table = bigquery_client.create_table(table)

    # Create a job to load data to specified table from a pandas dataframe
    # job = bigquery_client.load_table_from_dataframe(df, table)

    # Start job
    # print('Starting job {}'.format(job.job_id))
    # try:
    #    result = job.result()
    # except:
    #    print('Job failed:')
    #    print(job.errors)

    try:
        max_time = pickle.load(open("last_timestamp.p", "rb"))
    except:
        max_time = pd.Timestamp(0, unit='s')

    # Read from bigQuery table
    sqlquery1 = """
        SELECT *
        FROM `knime-project-automl.knime_dataset.test_table`
        WHERE TimeStamp > @last_time
    """
    query_parameters = [
        bigquery.ScalarQueryParameter("last_time", "TIMESTAMP", max_time)
    ]
    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = query_parameters
    df = bigquery_client.query(sqlquery1, job_config=job_config).to_dataframe()
    print(df)
    max_time = str(list(df['TimeStamp'])[0])
    print("max time is " + max_time)

    pickle.dump(max_time, open("last_timestamp.p", "wb"))

    print(len(df))


load_file_to_bigquery()