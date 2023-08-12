import os
import faiss
import openai
import pandas as pd
import numpy as np
import streamlit as st
import boto3

import warnings
warnings.filterwarnings('ignore')

########################################################################
###  Set the style of the page
########################################################################
css='''
<style>
    section.main > div {max-width:70rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)



#######################################################################
### Helper functions for download the data file and index from S3
#######################################################################

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

s3 = get_s3_session()

@st.cache_resource 
def download_data(df_file, bucket_name):
    cols = ['product_id', 'review_id', 'star_rating', 'product_title', 'review_body']
    if (download_s3_object(object_name=df_file, bucket_name=bucket_name, session=s3)):
        df_apparel = pd.read_csv(df_file, sep='\t', compression='gzip')
        df_apparel = df_apparel[cols]
    return df_apparel

@st.cache_resource 
def download_index(faiss_file, bucket_name):
    if (download_s3_object(object_name=faiss_file, bucket_name=bucket_name, session=s3)):
        faiss_index = faiss.read_index(faiss_file)
    return faiss_index


#######################################################################
### Helper functions for generating search results
#######################################################################


# Helper function to create the query embedding. Make sure to use the same model as what we used to created the product embedding
@st.cache_data
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


def generate_opposite_query(orignal_query=''):

    prompt="""
        You are an English teacher. You need to find every single ADJECTIVE from the sentences delimited by triple backquotes below.
        Then, you transform every adjective into its antonym.
        Finally, give the dictionary meaning for each antonym.
        Below are two examples. You need to comlete the third one. 
        

        Text 1: Kids flip flops for girl, cute, good fit, comfortable and durable, low price
        Output 1: Artless means without guile or deception. Unsuited means not proper or fitting for something. Uncomfortable means causing discomfort.  Fragile means easily broken. Costly means expensive.
        ## 
        Text 2: Long sleeve shirts for men. Wrinkle-free, thick but breathable and slim fit
        Output 2: Short means having little length. Crinkle means to form many short bends or ripples. Thin means measuring little in cross section or diameter. Airtight means impermeable to air or nearly so. Wide means having a greater than usual measure across
        ##
        Text 3:  ```{}```
        Output 3:
    """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt.format(orignal_query),
        temperature=0,
        max_tokens=1000,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response['choices'][0]['text']


def search_with_opposite_query(df, faiss_index, opposite_query_embedding, original_query_result_index, num_of_records=100):

    faiss.normalize_L2(opposite_query_embedding)

    # we want to make sure the opposite query only compare against the texts found by the original query 
    id_selector = faiss.IDSelectorArray(original_query_result_index.shape[1], faiss.swig_ptr(original_query_result_index))
    filtered_distances, filtered_indices = faiss_index.search(opposite_query_embedding, k=num_of_records, params=faiss.SearchParametersIVF(sel=id_selector))

    df_opposite_result = df.iloc[filtered_indices.squeeze().tolist()]
    df_opposite_result['distance'] = filtered_distances.T

    df_opposite_result = df_opposite_result.sort_values(by='distance', ascending=False)

    return df_opposite_result


def get_reconcile_result(df_result_original, df_result_opposite):

    df_reconcile_result = df_result_original.merge(df_result_opposite[['review_id', 'distance']], 
                            left_on='review_id', right_on='review_id', how='left', suffixes=('_original', '_opposite'))
    
    # Using Dot Product FAISS Index with L2 normaliztion, the returning result is Cosine Similiarty, rather than Distance.
    # There will turn the Cosine Similarity to Distance 
    df_reconcile_result['distance_original'] = 1 - df_reconcile_result['distance_original']
    df_reconcile_result['distance_opposite'] = 1 - df_reconcile_result['distance_opposite']

    return df_reconcile_result

# Helper function to clip the distance_opposite

def clip_distance_opposite(df, clipping=0.5):
    df = df.sort_values(by='distance_opposite', ascending=False).reset_index(drop=True)
    # Flatten the first n% with distance_opposite sorted in descending order  
    quantile_value = df['distance_opposite'].quantile(q=(1-clipping))
    df.loc[0:(clipping * len(df)), ['distance_opposite']] = quantile_value

    return df

# Helper function to calculate adjsuted distance using the distance_oppsite as a penalty term

def cal_adjusted_distance(df, k=0.5):
    df['distance_adjusted'] = df.apply(lambda row: row['distance_original'] + (k * 1/row['distance_opposite']), axis='columns')
    
    return df 


# Helper function to calculate review level similarity scores

def cal_review_similarity_score(df):
    # find max of distance_adjusted  
    max_distance_adjusted  = df['distance_adjusted'].max()
    # normalized adjusted distance then subtract from 1 to calculate the similarity score 
    df['similarity_score'] = df['distance_adjusted'].apply(lambda x: 1 - x / max_distance_adjusted)

    return df 


# Helper function to calculate the product level similarity scores 

def cal_product_similarity_score(df, method='discount_reward'):

    if method == 'average':
        df_temp = df.groupby('product_id')['similarity_score'].mean()
        df_temp = df_temp.to_frame(name='product_similarity_score').reset_index()
        df = pd.merge(df, df_temp, left_on='product_id', right_on='product_id')
    else:
        df_grouped = df.groupby(by='product_id')

        for name, data in df_grouped:
            data = data.sort_values('similarity_score', ascending=False)
            scores = []
            for cnt, (index, row) in enumerate(data.iterrows()):
                discounted_score = row['similarity_score'] / pow(2, cnt)
                scores.append(discounted_score)
            df.loc[data.index, 'product_similarity_score'] = sum(scores)
        
    df = df.sort_values(by=['product_similarity_score', 'similarity_score'], ascending=[False, False])

    return df


def get_final_search_result(df_reconcile_result, clipping=0, weight=0, method='discount_reward'):
    """
    Method 1:   Calculate product level similarity score by Average
                clipping=0, weight=0, method='average'

    Method 2:   Calculate product level similarity score by Discount Reward
                clipping=0, weight=0, method='discount_reward' 

    Method 3:   Calcuate product level similarity score by Discount Reward with Adjustment by Opposite Query 
                clipping=0.1, weight=0.5, method='discount_reward
    """

    df_copy = df_reconcile_result.copy()

    df_copy = clip_distance_opposite(df_copy, clipping=clipping)
    df_copy = cal_adjusted_distance(df_copy, k=weight)
    df_copy = cal_review_similarity_score(df_copy)
    df_copy = cal_product_similarity_score(df_copy, method=method)

    # re-arrange columns for output
    cols = ['product_id', 'review_id', 'star_rating', 'distance_original', 'distance_opposite',
            'distance_adjusted', 'similarity_score', 'product_similarity_score', 'product_title', 'review_body']

    df_copy = df_copy[cols]

    return df_copy



#####################################################
### Main Body Execution 
#####################################################


openai.api_key = st.secrets["OPENAI_KEY"]
df_file = 'apparel_10to14.tsv.gz'
bucket_name = 'nescapstone'
faiss_file = 'apparel_10to14_review_cosine.faissindex'


df_apparel = download_data(df_file, bucket_name)
faiss_index = download_index(faiss_file, bucket_name)


# Pre-defined search queries
queries = {
            "Q1" : "Long thin cotton socks for men, need to be breathable, even feeling cool for summer time.",
            "Q2" : "Wrinkle free chiffon blouse, sleek style, long sleeve, slim fit, with comfortable inside layer",
            "Q3" : "DC or Marvel superhero costumes for adults. Comes with a mask.  Deluxe and high quality. Comfortable to wear. Washable and durable.",
            "Q4" : "Baby footed bodysuit, made with soft and comfortable material, keeps the baby snug and warm, washed and dried well without shrinking", 
            "Q5" : "Women's knee-length A-line dress, looks cutie, made with high quality and relatively thick material, fits slim body",
            "Q6" : "Men's lightweight packable down jacket, wind-Resistant, quilted design, snug fit, warm but breathable",
            "Q7" : "Women's high-waisted leggings, tummy control for slimmer look, black color, moisture-wicking fabric, non-see-through",
            "Q8" : "Men's skinny fit stretch jeans, tight on legs but fits good on hips, soft, comfortable and durable",
            "Q9" : "Kid's Rain jacket with hood. Made from waterproof material that keeps dry even in heavy rain. Lightweight but able to keep body warm."
        }




st.title('Customer Review Driven Product Search Engine')


def change_selectbox():
    #st.session_state.changed = True
    st.session_state.user_query = ''


with st.sidebar:
    st.write("This is a Capstone project for the Master of Applied Data Science program under School of Information at the University of Michigan")
    st.write("For details, please vist https://github.com/gen-exody/nes")


col1, col2 = st.columns([0.7, 0.3])
with col1:
    selected_query = st.selectbox("Select a predefined query:", list(queries.values()), on_change=change_selectbox)
    user_query = st.text_area("Or enter your query:", height=30, key='user_query')
with col2:
    ranking_method = st.radio("Select a ranking algorithm:", ["Average", "Discounted Reward Only", "Discounted Reward with Adjustment"], index=2)
    search = st.button("Search")



df_final_result = None
        
if search:
    if user_query:
        query = user_query
    else:
        query = selected_query

    # Implement the search logic
    query_embedding = get_embedding(query)
    df_result_original, result_idx = search_with_original_query(df_apparel, faiss_index, query_embedding, num_of_records=100)
    opposite_query = generate_opposite_query(query)
    opposite_query_embedding = get_embedding(opposite_query)
    df_result_opposite = search_with_opposite_query(df_apparel, faiss_index, opposite_query_embedding, result_idx, num_of_records=100)
    df_reconcile_result = get_reconcile_result(df_result_original, df_result_opposite)

    # Choose the appropriate method based on selected ranking algorithm
    if ranking_method == "Average":
        df_final_result = get_final_search_result(df_reconcile_result, clipping=0, weight=0, method='average')
    elif ranking_method == "Discounted Reward Only":
        df_final_result = get_final_search_result(df_reconcile_result, clipping=0, weight=0, method='discount_reward')
    else:
        df_final_result = get_final_search_result(df_reconcile_result, clipping=0.1, weight=0.5, method='discount_reward')


if df_final_result is not None:
    #st.write(df_final_result)
    cnt = 1
    prev_product_id = ''
    for _, row in df_final_result.iterrows():
        product_id = row['product_id']
        product_title = row['product_title']
        review_body = row['review_body']
        similarity_score = round(row['product_similarity_score'], 4)

        if product_id != prev_product_id:
            st.write("-" * 30)
            st.write(f"<h4>{cnt}. ({product_id}) {product_title}</h4>", unsafe_allow_html=True)
            st.write(":green[Product Similarity Score:]", similarity_score)
            st.write("_Review Body_:", review_body)
            cnt += 1
        else:
            st.write("_Review Body_:", review_body)
        
        prev_product_id = product_id

