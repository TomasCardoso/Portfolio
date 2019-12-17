#This script aims at making a connection with ElasticSearch, retrieve some data, create a DataFrame
#using Pandas and write it back to ElasticSearch. It is a specific example using a specific index.
#To apply to another index with different type of keys and data, modifications need to be made.

#Author: Tomás Cardoso

from pandas import DataFrame
from espandas import Espandas
from elasticsearch import Elasticsearch
import json
import pandas as pd
import elasticsearch.helpers # helpers

### custom elasticsearch queries
import queries.template
import queries.documents

#Recursive function that searches for index keys that are not dictionaries and stores them in result
def getKeys(data):
	result = []
	for key in data.keys():
		if type(data[key]) != dict:
			result.append(key)
		else:
			result += getKeys(data[key])
	return cleanKeys(result)

#Removes keys that start with '_'. These keys are created by Kibana when queries are processed and
#we only want the keys corresponding to data fields
def cleanKeys(keys):
	result = []
	for key in keys:
		if not key.startswith('_'):
			result.append(key)

	return result
#Function that receives a document, the set of keys and an iterator that indicates a specific key.
#Tries to retrieve data from the specific key of the document and returns it, if the key does not
#exist in the document, returns None.
def getContentOfDoc(doc, keys, iterator):
	try:
		data = json.dumps(doc['_source'][keys[iterator]])
		data = data.replace('"','')
	except:
		data = None

	return data

#Functions that processes a scroll of documents of size defined in the query and updates a dictionary with it.
#For each document, singles out the actual data camps with ['hits']['hits'] and retrieves the data
#from each key. The dictionary passed through input is updated with this data
def processDocBatch(scroll, dic, keys):
	for doc in scroll['hits']['hits']:
		for i in range(0, len(keys)):
			data = getContentOfDoc(doc, keys, i)
			dic[keys[i]].append(data)
	return dic

def main():
	#connect to ElasticSearch
	es = Elasticsearch(['http://10.0.0.55:9207'])

	#Query to retrieve data from ElasticSearch. Excludes the key message and returns all data
	#from 2018-06-17T23:00:00 to 2018-06-18T23:00:00 (actually it returns from 
	#2018-06-18T00:00:00 to 2018-06-19T00:00:00). This query will retrieve data in smaller 
	#scrolls since there is a limit of 10k documents per single retrieval.
	response = es.search(
		index='tomas-knime',	#Name of index
		scroll = '2m',
  		size = 1000,	#Size of each scroll
		body={
			"_source": {
			    "excludes": [
			      "message"
			    ]
			  },
			"query": {
				"range": {
					"@timestamp": {
						"gte": "2018-06-17T23:00:00",
						 "lt": "2018-06-18T23:00:00"
					}
				}
			}
		},
	)
	sid = response['_scroll_id'],
	scroll_size = response['hits']['total']

	#Gets all keys from the first document and uses those to create the table. Works if all documents have the same keys
	#keys = getKeys(response['hits']['hits'][0])

	#use User defined keys
	keys = ['UserAgentName', '@timestamp', 'CallTime', 'Status', 'FirstInviteProcessingTime']
	#creates empty dictionary with the set of keys
	dic = {key: [] for key in keys}
	#process the first scroll
	dic = processDocBatch(response, dic, keys)

	# Start scrolling
	print("Scrolling...")
	while (scroll_size > 0):
		#gets scroll
		response = es.scroll(scroll_id = sid, scroll = '2m')
		# Update the scroll ID
		sid = response['_scroll_id']
		# Get the number of results that we returned in the last scroll
		scroll_size = len(response['hits']['hits'])
		#processes scroll
		dic = processDocBatch(response, dic, keys)
	#creates Pandas dataframe with resulting dictionary
	df = DataFrame(data = dic)

	#convert specific columns to numbers (change at will)
	df['CallTime'] = pd.to_numeric(df['CallTime'])
	df['FirstInviteProcessingTime'] = pd.to_numeric(df['FirstInviteProcessingTime'])

	#Alright, ElasticSearch -> Dataframe completed
	#Now onto Dataframe -> ElasticSearch

	#convert dataframe to list of documents in json format
	data = []
	destination_index = 'test_index'

	for index, row in df.iterrows():
		data.append({
                        "_index": destination_index,
                        "_id": str(row['@timestamp']) + str(row['UserAgentName']) + str(row['CallTime']) + str(row['Status']),
                        "_type": "aggregations",
                        "_source": {
                            '@timestamp': row['@timestamp'],
                            'UserAgentName': row['UserAgentName'],
                            'CallTime': row['CallTime'],
                            'Status': row['Status'],
                            'FirstInviteProcessingTime': row['FirstInviteProcessingTime']
                        }
                    })

	# get template
	template = queries.template.get_template()
    # create index
	result = es.indices.create(index=destination_index, body=template, ignore=400)

     # insert into elasticsearch
	bulk_result = elasticsearch.helpers.bulk(es, data)
main()