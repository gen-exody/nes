# Customer Review Driven Product Search Engine

*Elaine Chen, Gen Ho, Varshini Rana*

## Motivation 

When customers go online to purchase from e-commerce websites, product reviews are a vital source of information to reference as the target products cannot be physically touched. However, reading through all the reviews and comparing across different products could be time consuming and challenging. 

We propose to build a product search application which allows customers to search by natural language queries. Unlike traditional product search engines which are mainly based on predefined categories and keywords in product descriptions, our proposed solution returns results with consideration of customer reviews. We believe this search engine can greatly improve customer experience over product searching on e-commerce websites. 

## Data Source and scope

We used the [Amazon Customer Review](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) dataset for our application. This dataset contains information of Amazon products and corresponding reviews from 1995 to 2015. There are 37 product categories where we have chosen the Apparel product group. We have also limited our scope to 2015 and further divided the data into two sub-groups. 
- The first group is products with 15-25 reviews. This becomes our development dataset
- The second subgroup is products with 11-14 reviews. This serves as our evaluation set. 

## Solution Architecture

Our solution is an embedding-based retrieval (EBR) system where the embeddings are created by the Large Language Model (LLM), OpenAI GPT-3 specifically.   
EBR is widely accepted nowadays as a better approach than traditional term-based retrieval methods for information retrieval as term-based methods only focus on exact matching, thus lack the consideration for semantically related words <sup>[1]</sup>.  
On the other hand, we have also formulated our own ranking algorithm with the aid of a mechanism which we called the opposite query.  

Here is the logical architecture of our solution. We will go over the implementation details in the next section.  

![Logical Architecture](https://github.com/gen-exody/nes/blob/master/resources/img/architecture.png?raw=true) <figcaption>Fig. 1 Application Logical Architecture.</figcaption>



## References

[1] Davis Liang et al. (Sep 22, 2020). "Embedding-based Zero-shot Retrieval through Query Generation" https://arxiv.org/abs/2009.10270. Accessed Jul 10, 2023