import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import openai
import boto3


########### Define Helper Functions###################

def get_s3_session():

    session = boto3.Session(
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    )
    return session.resource('s3')

def download_s3_object(object_name, bucket_name, session):

    s3_obj = session.Object(
        bucket_name=bucket_name,
        key=object_name
    )

    with open(object_name, 'wb') as file:
        s3_obj.download_fileobj(
        Fileobj=file
    )

    return True


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   
   return np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'], dtype='float32').reshape(1, -1)


def search_with_original_query(df, faiss_index, query_embedding, num_of_records=100):
    # we need to normalize the question embedding in order to use cosine similarity to search 
    faiss.normalize_L2(query_embedding)

    # distance is the correspnding distance
    # result_idx is the index of the input array, hence the index of the dataframe (if the dataframe index is reset which starts with 0)
    distance, result_idx = faiss_index.search(query_embedding, k=num_of_records)

    # use the return index to create the result dataframe
    df_result = df.iloc[result_idx.squeeze().tolist()]
    # add Distance to the result dataframe
    df_result['distance'] = distance.T

    df_result = df_result.sort_values(by='distance', ascending=True)
    
    return df_result, result_idx

######## Main Starts Here ####################################

s3 = get_s3_session()

df_file = 'apparel_10to14.tsv.gz'
bucket_name = 'nescapstone'
faiss_file = 'apparel_10to14_review_cosine.faissindex'

data_load_state = st.text('Loading data...')

if (download_s3_object(object_name=df_file, bucket_name=bucket_name, session=s3)):
    df_apparel = pd.read_csv(df_file, sep='\t', compression='gzip')

if (download_s3_object(object_name=faiss_file, bucket_name=bucket_name, session=s3)):
    faiss_index = faiss.read_index(faiss_file)

data_load_state.text("Done Loading Data")





query = "Long thin cotton socks for men, need to be breathable, even feeling cool for summer time."
st.write(query)

openai.api_key = st.secrets["OPENAI_KEY"]
query_embedding = get_embedding(query)
df_result_original, result_idx = search_with_original_query(df_apparel, faiss_index, query_embedding, num_of_records=100)

st.write(df_result_original)