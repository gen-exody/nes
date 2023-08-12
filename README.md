# Customer Review Driven Product Search Engine

<img src="https://github.com/gen-exody/nes/blob/master/resources/img/umsi.png?raw=true" align="left"/>

*This is a Capstone project for the Master of Applied Data Science program under School of Information at the University of Michigan*  

*Elaine Chen, Gen Ho, Varshini Rana (8-14-2023)*


## Motivation

When customers go online to purchase from e-commerce websites, product reviews are a vital source of information to reference as the target products cannot be physically touched. However, reading through all the reviews and comparing across different products could be time consuming and challenging. 

We propose to build a product search application which allows customers to search with natural language queries. Unlike traditional product search engines which are mainly based on predefined categories and keywords in product descriptions, our proposed solution returns results with consideration of customer reviews. We believe this search engine can greatly improve customer experience over product searching on e-commerce websites.


## Data Source and Scope

We used the **Amazon Customer Review** dataset for our application. This dataset contains information of Amazon products and corresponding reviews from 1995 to 2015. There are 37 product categories where we have chosen the Apparel product group. We have also limited our scope to 2015 and further divided the data into two sub-groups. 
- The first group is products with 15-25 reviews. This becomes our development dataset
- The second subgroup is products with 11-14 reviews. This serves as our evaluation set. 

The source data for this projet can be downloaded at [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Apparel_v1_00.tsv). More detailed information including the license model can be found [here](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). 

## Solution Architecture

Our solution is an embedding-based retrieval (EBR) system where the embeddings are created by the Large Language Model (LLM), OpenAI GPT-3 specifically.   
EBR is widely accepted nowadays as a better approach than traditional term-based retrieval methods for information retrieval as term-based methods only focus on exact matching, thus lack the consideration for semantically related words <sup>[1]</sup>.  
On the other hand, we have also formulated our own ranking algorithm with the aid of a mechanism which we called the opposite query.  

Here is the logical architecture of our solution. We will go over the implementation details in the next section.  

<figure>
  <img src="https://github.com/gen-exody/nes/blob/master/resources/img/architecture.png?raw=true" alt="Logical Architecture"/>
  <figcaption>Figure 1: Application Logical Architecture.</figcaption>
</figure>  
  
1. Use OpenAI service to transform the documents of `product_tile + review_body` into document embeddings
2. The document embeddings are indexed by Faiss 
3. Use OpenAI service to transform the product search query into query embeddings
4. Conduct a similarity search over  the Faiss index with the query embeddings
5. The returned result is sent to our ranking algorithm to produce the finalized, ranked result. 

## Getting Started

### GitHub project file structure

    .
    |____notebooks/                        # research jupyter notebooks 
    |____resources/                        # 
    | |____binary/                         # stores the Faiss index after created
    | |____data/                           # stores the source data
    | |____eval/                           # stores project evaluation results by raters
    | |____img/                            # stores image files
    | |____output/                         # stores search engine outputs for evaluation
    |____streamlit_app.py                  # Streamlit application
    |____README.md                         # this file

### Clone this repo

```
git clone https://github.com/gen-exody/nes.git
```

In order to run the Jupyter notebooks and the Streamlit application in this project, you need to have your own [OpenAI subscription](https://openai.com/) and your [AWS S3 storage](https://aws.amazon.com/s3/?nc2=h_ql_prod_fs_s3). 

### Run the Streamlit application

To run the Streamlit application in your development environment, you need to firstly run the `source_data_analysis_preprocessing.ipynb`, `create_embedding.ipynb`, and `create_index.ipynb` notebooks, then store the outputted `apparel_10to14.tsv.gz` and `apparel_10to14_review_cosine.faissindex` files in your AWS S3 storage.

You also need to plug those values into the secrets.toml file in the /.streamlit directory. For details, please refer to the [Streamlit documentation]( 
https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management). 



```
AWS_ACCESS_KEY_ID = "your_aws_access_key_id"
AWS_SECRET_ACCESS_KEY = "your_aws_access_key"
OPENAI_KEY = "your_open_ai_key"

```
Issue below commond in your OS shell to run the Streamlit app
```
streamlit run streamlit_app.py
```

## Resources

1. The detailed report for this project is located at the [Wiki](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine) of this GitHub repository. 

2. We have built a prototype application on Streamlit which allows users to search products over our evaluation data set - Apparel in 2015 with 10-14 reviews. 

