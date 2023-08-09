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

<figure>
  <img src="https://github.com/gen-exody/nes/blob/master/resources/img/architecture.png?raw=true" alt="Logical Architecture"/>
  <figcaption>Figure 1: Application Logical Architecture.</figcaption>
</figure>  
  
1. Use OpenAI service to transform the documents of `product_tile + review_body` into document embeddings
2. The document embeddings are indexed by Faiss 
3. Use OpenAI service to transform the product search query into query embeddings
4. Conduct a similarity search over  the Faiss index with the query embeddings
5. The returned result is sent to our ranking algorithm to produce the finalized, ranked result. 


## Implementation 

In this section we will go over the major techniques and considerations for building the search engine.    

### Create Embeddings

Rather than reinventing the wheel by training our own embeddings, we have decided to use the service offered by OpenAI. Specifically, we have used the `text-embedding-ada-002` model for transforming text into embeddings. This is the best embedding model offered by OpenAI as of now <sup>[2]</sup>. 

Creating embedding using API is straightforward, just calling the `openai.Embedding.create()` API then things can be done. However, there is a problem with calling OpenAI service with a large number of calls, that is the rate limit. The rate limit is a restriction that an API imposes on the number of times a client can access the server within a specified period of time. Exceptions will be thrown during our code execution if we call the API in a single batch with our data (around 90k records). Luckily, there are workarounds. 

The first strategy we employed is using the [Python Tenacity Library](https://github.com/jd/tenacity) to set up a retry mechanism. The second strategy we used is dividing the API calls in batches with 60s delay between each call. 
OpenAI has an [official document](https://platform.openai.com/docs/guides/rate-limits/error-mitigation) talking about this rate limit issue and the error mitigation strategy. 

Here is our implementation.

```python
# helper function 
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Use tenacity retry to tackle the OpenAI "rate limits" problem
# reference: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding_with_backoff(**kwargs):
    return get_embedding(**kwargs)

def transform_column_to_embedding(df, source_column, target_column_name, rate_limit_per_minute=3000, delay=60.0):
    
    num_of_batch = math.ceil(len(df) / rate_limit_per_minute)
    # also use batch strategy to handle the OpenAI "rate limits" problem apart from the retry mechanism above 
    chunks = []
    tqdm.pandas(desc='Processing rows')
    for chunk in np.array_split(df, num_of_batch):
        chunk[target_column_name] = chunk[source_column].progress_apply(lambda x:get_embedding_with_backoff(text=x))
        chunks.append(chunk)
        time.sleep(delay)

    return pd.concat(chunks, ignore_index=True)
```

### Create Index

We use the [Facebook AI Similarity Search (Faiss)](https://github.com/facebookresearch/faiss/wiki/) for vector index creation and similarity search. Faiss offers different kinds of index types. For our use case, we prefer cosine similarity over euclidean distance for similarity (or distance) measurement in building the indexes because of two major reasons <sup>[3]</sup>.   

- cosine similarity is calculated based on the angle between two vectors rather than their magnitudes, thus is better for comparing sentences of variety lengths.   
- cosine similarity is ranged from -1 (completely dissimilar) to 1 (highly similar) which allows intuitive analysis and comparison. 

Faiss does not directly provide an index type of cosine similarity. However, it does provide an index using the inner product - [IndexFlatIP](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_inner_product). We just need to normalize the document vectors before creating the index, also normalize the query vector prior the search, then the inner product becomes cosine similarity.

Here is our implementation.

```python
# initialize the IndexFlatIP with the embedding dimension 
index_dev = faiss.IndexFlatIP(len(df_development['embedding'][0]))
index_dev.is_trained

# create the embeddings array
# this array is required by faiss to be float32 
embeddings_dev = np.array(df_development['embedding'].to_list(), dtype='float32')

# Normalize the embeddings and add to to the index
faiss.normalize_L2(embeddings_dev)
index_dev.add(embeddings_dev)
```

### Design Ranking Algorithm

Although our search engine allows the searching of products with the context from user reviews, in the end what a customer wants is a list of products which matches the search criteria. Thus,  we need to find a way to group the returning records in product level and order them according to the relevance to the search query.

Under current implementation, the engine will fetch the top 100 similar records. We firstly normalize the cosine similarity of each record by dividing it with the maximum similarity of the result set. Then we can calculate the product level similarity score.  

We have come up with 3 strategies for calculating the product level similarity score. 

1. **Average**

   To get the product level similarity score, the most intuitive way is taking the arithmetic mean of  the review level scores. However, the result could be easily affected by extreme values. More importantly, this does not reflect the collective information across reviews. We consider this method as the baseline for comparison.

2. **Discounted Reward**

   As each review can contain specific information, in calculating the product level similarity score, we should collectively consider all reviews. On the other hand, we actually want to reward a product with more reviews that match the search criteria since it increases a customer’s confidence in this product. Therefore, aggregation is a better choice than averaging. However, we also do not want the result to be dominated by the number of reviews at the same time. So, here comes the idea of Discounted Reward.

   Under this method, we firstly sort records by similarity scores in descending order within each product. Then, we have:

   $$Product Similarity Score = \displaystyle\sum_{i=1}^{n}\frac{S_i}{2^2}$$
   where $S$ is the record level similarity score, $n$ is the total number of records within a product, $i$ is the rank of the current record

   The idea here is to discount the similarity score by 2 to the power of the rank of the record, then adding the results up.

3. **Discounted Reward with Adjustment by Opposite Query**  

   Customer reviews provide very rich context for product searches. However, with similarity search, there could be chances that the returning result contains contradictory information against the search query. 
   
   For example, let's say we want to search for **_“Cotton socks for men which are breathable and keep feet cool for summer time.”_** The returning result could be **_“These cotton socks are breathable but still make my feet too warm during summer”_**. Here, while these two sentences are very similar, there is a contradicting concept - cool vs. warm. 
   
   In order to address this and hence further enhance the accuracy of the search result, we introduce the opposite query. The idea is to find the distance between the original search results against a query that has opposite ideas from the original query. The resulting distance can then be used to penalize the search result from the original query if they are close to the opposite query. 

   #### Generate Opposite Query

   In a traditional product search engine, the query criteria is built around the product titles and perhaps descriptions. Thus, most of the wordings are nouns. However, for user reviews, there can be lots of adjectives since they talk about user experience. So, in designing the opposite query, we target to transform the adjectives into their antonyms. 
   
   To generate the opposite query, we again use OpenAI service. This time, we use the `openai.Completion.create()` [API](https://platform.openai.com/docs/guides/gpt/completions-api) with the `text-davinci-003` model.   

   ```python
   def generate_opposite_query(orignal_query='', prompt=''):

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt.format(orignal_query),
            temperature=0,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return response['choices'][0]['text']
   ```
   Our initiation idea was to write the original query by replacing all the adjectives into their corresponding antonyms. For example: 
   - **Original Query**: _Wrinkle free chiffon blouse, sleek style, long sleeve, slim fit, with comfortable inside layer._
   - **Opposite Query**: _Wrinkle-ridden chiffon blouse, bulky style, short sleeve, baggy fit, with uncomfortable inside layer._ 
   
   However, upon reviewing the line chart **(Figure 2, left)** of the distance between the original query and the search results, plus the one of the opposite query, we found the behavior was not as expected. Specifically, the two lines were too close which can not effectively reflect the opposite concept. We suspected this was caused by the nouns in the query.

   <figure>
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/design_opposite_query.png?raw=true" alt="Design Opposite Query"/>
    <figcaption>Figure 2: (Left) Rewrite Whole Query. (Right) Use Antonyms with corresponding meaning</figcaption>
   </figure>  
   
   Then we changed our strategy to just build the opposite query with the antonyms and their meanings. This became:
   - **Original Query**: _Wrinkle free chiffon blouse, sleek style, long sleeve, slim fit, with comfortable inside layer._
   - **Opposite Query**: _Wrinkled means having many creases or folds. Clumsy means lacking grace in movement or posture. Short means having little length. Bulky means large and unwieldy. Uncomfortable means causing discomfort._
   
   The resulting line chart **(Figure 2, right)** looked more reasonable than the first one  and we decided to pick this option.
   
   Here is the prompt for calling the OpenAI service. 
   ```python
    prompt = """
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
   ```

   #### Design the formula for adjustment with opposite query 

   After investigating the search results sorted by distance_opposite in descending order, we found that while top portions are really far away from the opposite query, their contents were not most relevant to the original search query **(Figure 3)**. This means we cannot just adding the reciprocal of the opposite query distance as a penalty term (we want to penalize reviews closer to the opposite query thus taking reciprocal) since this would potentially overboost those `product_title + review_body` that are actually not relevant to the product search.

   <figure>
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/opposite_query_top5.png?raw=true" alt="Top 5 results sorted by distance_opposite in descending order"/>
    <figcaption>Figure 3: Top 5 results sorted by distance_opposite in descending order</figcaption>
   </figure>  
   
   We came up with 2 strategies to handle this issue. 
   - We have introduced a clipping mechanism where we flatten certain portions of the top ranked (descending order) opposite query distances. Through testing with different samples, we have decided to clip the top 10 percentile for our implementation. The idea is illustrated by **Figure 4** below.
   - We have added weight to the penalty term which has been set to 0.5 in our implementation.  

   <figure>
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/clipping.png?raw=true" alt="Clipping"/>
    <figcaption>Figure 4: Distance before and after clipping (highlighted in red) </figcaption>
   </figure>

   The finalized formula for the adjusted distance is shown below. 
   

      $$Adjusted Distance =  clip\ f(D_{original}) + K\times\frac{1}{D_{opposite}}$$

   where $clip\ f()$ is our custom function for clipping, $D_{original}$ is the cosine distance of the original query, $D_{opposite}$ is the cosine distance of the opposite query, and $K$ is the weight of the penality term.

## Unsupervised Data Analysis  

Apart from designing and building the ranking algorithm, we also conducted an unsupervised data analysis over our development dataset. The objective was to identify product types in the dataset in order to support the construction of the product search queries for development and  result evaluation. 

We firstly used [Uniform Manifold Approximation and Projection (UMAP)](https://umap-learn.readthedocs.io/en/latest/index.html) for Dimension Reduction to reduce the product_title dimension from 1,536 to 2. UMAP is a non-linear dimension reduction technique which can provide more optimized separation for 2-dimensions when compared to [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)<sup>[4]</sup>. 

In preparing the below scatter plot **(Figure 5)**, we colored the dots with the star_rating values. This field contains 1-5 star rating of the review. More diverging colors within a product type can hit a higher chance of having contradicting reviews, which might mean more suitable for us to construct the queries for development and evaluation. 

Please refer to the Appendix for the list of queries we have defined.

<figure>
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/unsupervised_analysis.png?raw=true" alt="Product Type Analysis"/>
    <figcaption>Figure 5: Product Types Analysis </figcaption>
</figure>  
    
 

## Result Analysis 

We have defined 9 queries to search over the evaluation dataset (products with 11-14 reviews). The output was then rated by 3 raters with a scale of 1 to 5 where 1 denoted “Not relevant at all” while 5 denoted “Perfectly relevant”. 

We use [Normalized Discounted Cumulative Gain (NDCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) to evaluate the goodness of ranking for our search engine under the 3 ranking methods, i.e. i) average, ii) discounted reward, and iii) discounted reward with adjustment by opposite query. 

> NDCG is a measure of the effectiveness of a ranking system, taking into account the position of relevant items in the ranked list. It is based on the idea that items that are higher in the ranking should be given more credit than items that are lower in the ranking. NDCG ranges from 0 to 1, with higher values indicating better performance <sup>[5]</sup>

For the details on how NDCG is calculated, we highly recommend this article [“Demystifying NDCG” by Aparna Dhinakaran](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0). 

Based on the scores collected from the 3 raters, we calculated the NDCG at 3, 5, and 10 which refers to scores calculated up to the top 3, 5 and 10 results respectively. 

Below table **(Table 1)** shows the resulting mean NDCG@n scores across the 3 ranking methods. 

| Ranking Method                | Mean NDCG@3   | Mean NDCG@5   | Mean NDCG@10  |
| :--------------------------   | -----------:  | ----------:   | -----------:  |
| Average                       | 0.819         | 0.847         | 0.935         |
| Discounted Reward Only        | 0.837         | 0.861         | 0.939         |
| Discounted Reward with Adjustment by Opposite Query   | 0.837 | 0.862 | 0.941 |

Table 1: NDCG@n scores across the 3 ranking methods

The visualization below on the left **(Figure 6, left)**  shows the mean NDCG scores with confidence intervals across the three ranking methods, whereas the one on the right **(Figure 6, right)** shows the top ranking method (with highest mean value) for the 9 queries. You can refer to the Appendix for the detailed scores. 


<figure>
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/eval_analysis_chart.png?raw=true" alt="Evaluation Result Analysis"/>
    <figcaption>Figure 6: (Left) Mean NDCG across the three ranking methods. (Right) Top ranking method for the 9 queries</figcaption>
</figure>  


From the table and visualizations above, we can notice that generally the *Method 2) Discounted Reward Only* and *Method 3) Discounted Reward with Adjustment by Opposite Query* performed better than the *Method 1) Average*. However, *Method 2* is just a tiny bit better than *Method 3* in NDCG@5 and NDCG@10. The 95% confidence intervals are very similar for all three methods within the same NDCG group. On the other hand, *Method 1* had a 37% (10/27) chance of getting the best results while for *Method 2* and *Method 3* it was 41% (11/27) and 22% (6/27) respectively.


Overall, while *Method 3* had the highest mean NDCG scores, in practice *Method 2* had almost a double chance to outperform *Method 3*. It means *Method 3* performed much better in some cases only but not all. 

*Method 2* and *Method 3* performed better than *Method 1* in terms of NDCG scores which is within our expectation since *Method 2 and 3* have considered the collective information from all reviews which can better represent human decision models. However, the Opposite Query of *Method 3* did not perform as good as we originally thought. 

Through investigation, we believe the reason boils down to a problem with our assumption. We assume semantic oppositeness means a complete inverse of semantic similarity in the computing domain, while we have tried to use this oppositeness measure to find the contradicting concepts in customer reviews with the opposite queries. 

We took an example to further illustrate the idea. Here we use this original query and its corresponding opposite query to create the visualizations below. 
- **Original Query**: _Long thin cotton socks for men, need to be breathable, even feeling cool for summer time._
- **Opposite Query**: _Short means having little length. Thick means having a greater than usual measure across. Unbreathable means not allowing air to pass through. Hot means having or giving out a great deal of heat._









## References

[1] Davis Liang et al. (Sep 22, 2020). "Embedding-based Zero-shot Retrieval through Query Generation". https://arxiv.org/abs/2009.10270. Accessed Jul 10, 2023

[2] Ryan Greene et al. (Dec 15, 2022). "New and improved embedding model". https://openai.com/blog/new-and-improved-embedding-model. Accessed Jul 2, 2023

[3] baeldung. (Nov 24, 2022). "Euclidean Distance vs Cosine Similarity". https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity. Accessed Jun 30, 2023

[4] Ron Rotkopf. (Jan 4, 2021). "Less is more: An Intro to Dimensionality Reduction". http://dors.weizmann.ac.il/course/workshop2021/scRNA/Dimensionality_Reduction.pdf. Accessed Jul 3, 2023

[5] Aparna Dhinakaran. (Jan 25, 2023). "Demystifying NDCG - How to best use this important metric for monitoring ranking models". https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0. Accessed Jul 16, 2023 

## Appendix