# Customer Review Driven Product Search Engine

<img src="https://github.com/gen-exody/nes/blob/master/resources/img/umsi.png?raw=true" align="left"/>

*This is a Capstone project for the Master of Applied Data Science program under School of Information at the University of Michigan*  

*[Elaine Chen](mailto:yulchen@umich.edu), [Gen Ho](mailto:genho@umich.edu), [Varshini Rana](mailto:varshini@umich.edu) (8-14-2023)*

## Motivation 

When customers go online to purchase from e-commerce websites, product reviews are a vital source of information to reference as the target products cannot be physically touched. However, reading through all the reviews and comparing across different products could be time consuming and challenging. 

We propose to build a product search application which allows customers to search with natural language queries. Unlike traditional product search engines which are mainly based on predefined categories and keywords in product descriptions, our proposed solution returns results with consideration of customer reviews. We believe this search engine can greatly improve customer experience over product searching on e-commerce websites. 

## Data Source and Scope

We used the **Amazon Customer Review** dataset for our application. This dataset contains information of Amazon products and corresponding reviews from 1995 to 2015. There are 37 product categories where we have chosen the Apparel product group. We have also limited our scope to 2015 and further divided the data into two sub-groups. 
- The first group is products with 15-25 reviews. This becomes our development dataset
- The second subgroup is products with 11-14 reviews. This serves as our evaluation set. 

The source data for this projet can be downloaded at [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Apparel_v1_00.tsv). More detailed information including the license model can be found [here](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). 

## Solution Architecture

Our solution is an embedding-based retrieval (EBR) system where the embeddings are created by the Large Language Model (LLM), OpenAI GPT-3 specifically. EBR is widely accepted nowadays as a better approach than traditional term-based retrieval methods for information retrieval as term-based methods only focus on exact matching, thus lack the consideration for semantically related words <sup>[[1](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine#references)]</sup>.  
On the other hand, we have also formulated our own ranking algorithm with the aid of a mechanism which we called the opposite query.  

Here is the logical architecture of our solution. We will go over the implementation details in the next section.  


<p align="center">
  <img src="https://github.com/gen-exody/nes/blob/master/resources/img/architecture.png?raw=true" title="Logical Architecture">
</p>
<p align="center"><i>Figure 1: Application Logical Architecture.</i></p>




1. Use OpenAI service to transform the documents of `product_tile + review_body` into document embeddings
2. The document embeddings are indexed by Faiss 
3. Use OpenAI service to transform the product search query into query embeddings
4. Conduct a similarity search over  the Faiss index with the query embeddings
5. The returned result is sent to our ranking algorithm to produce the finalized, ranked result. 


## Implementation 

In this section we will go over the major techniques and considerations for building the search engine.    

### Create Embeddings

Rather than reinventing the wheel by training our own embeddings, we have decided to use the service offered by OpenAI. Specifically, we have used the `text-embedding-ada-002` model for transforming text into embeddings. This is the best embedding model offered by OpenAI as of now <sup>[[2](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine#references)]</sup>. 

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

We use the [Facebook AI Similarity Search (Faiss)](https://github.com/facebookresearch/faiss/wiki/) for vector index creation and similarity search. Faiss offers different kinds of index types. For our use case, we prefer cosine similarity over euclidean distance for similarity (or distance) measurement in building the indexes because of two major reasons <sup>[[3](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine#references)]</sup>.   

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

   Under this method, we firstly sort records by similarity scores in descending order within each product. Then, we have: <br>

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

   <P align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/design_opposite_query.png?raw=true" alt="Design Opposite Query"/>
   </P>
   <p align="center"><i>Figure 2: (Left) Rewrite Whole Query. (Right) Use Antonyms with corresponding meaning</i></p>
   



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

   <p align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/opposite_query_top5.png?raw=true" alt="Top 5 results sorted by distance_opposite in descending order"/>
   </p>
   <p align="center"><i>Figure 3: Top 5 results sorted by distance_opposite in descending order</i></p>
   
   We came up with 2 strategies to handle this issue. 
   - We have introduced a clipping mechanism where we flatten certain portions of the top ranked (descending order) opposite query distances. Through testing with different samples, we have decided to clip the top 10 percentile for our implementation. The idea is illustrated by **Figure 4** below.
   - We have added weight to the penalty term which has been set to 0.5 in our implementation.  

   <p align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/clipping.png?raw=true" alt="Clipping"/>
   </p>
   <p align="center"><i>Figure 4: Distance before and after clipping (highlighted in red)</i></p>


   The finalized formula for the adjusted distance is shown below.  
   
   $$Adjusted Distance =  clip\ f(D_{original}) + K\times\frac{1}{D_{opposite}}$$ 

   where $clip\ f()$ is our custom function for clipping, $D_{original}$ is the cosine distance of the original query, $D_{opposite}$ is the cosine distance of the opposite query, and $K$ is the weight of the penalty term.

## Unsupervised Data Analysis  

Apart from designing and building the ranking algorithm, we also conducted an unsupervised data analysis over our development dataset. The objective was to identify product types in the dataset in order to support the construction of the product search queries for development and  result evaluation. 

We firstly used [Uniform Manifold Approximation and Projection (UMAP)](https://umap-learn.readthedocs.io/en/latest/index.html) for Dimension Reduction to reduce the `product_title` dimension from 1,536 to 2. UMAP is a non-linear dimension reduction technique which can provide more optimized separation for 2-dimensions when compared to [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)<sup>[[4](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine#references)]</sup>. 

In preparing the below scatter plot **(Figure 5)**, we colored the dots with the star_rating values. This field contains 1-5 star rating of the review. More diverging colors within a product type can hit a higher chance of having contradicting reviews, which could mean more suitable for us to construct the queries for development and evaluation. 

Please refer to the Appendix for the list of queries we have defined.


<p align="center">
  <img src="https://github.com/gen-exody/nes/blob/master/resources/img/unsupervised_analysis.png?raw=true" title="Product Type Analysis">
</p>
<p align="center"><i>Figure 5: Product Types Analysis</i></p>


## Result Evaluation and Analysis 

We have defined 9 queries to search over the evaluation dataset (products with 11-14 reviews). The output was then rated by 3 raters with a scale of 1 to 5 where 1 denoted “Not relevant at all” while 5 denoted “Perfectly relevant”. 

We use [Normalized Discounted Cumulative Gain (NDCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) and [Mean Reciprocal Rank (MRR)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) to evaluate the goodness of ranking for our search engine under the 3 ranking methods, i.e. *i) Average*, *ii) Discounted Reward*, and *iii) Discounted Reward with Adjustment by Opposite Query*. The *i) average* method acts as the baseline for our evaluation.  

###  Normalized Discounted Cumulative Gain (NDCG)

> NDCG is a measure of the effectiveness of a ranking system, taking into account the position of relevant items in the ranked list. It is based on the idea that items that are higher in the ranking should be given more credit than items that are lower in the ranking. NDCG ranges from 0 to 1, with higher values indicating better performance <sup>[[5](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine#references)]</sup>

For the details on how NDCG is calculated, we highly recommend this article [“Demystifying NDCG” by Aparna Dhinakaran](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0). 

Based on the scores collected from the 3 raters, we calculated the NDCG at 3, 5, and 10 which refers to scores calculated up to the top 3, 5 and 10 results respectively. 

Below table **(Table 1)** shows the resulting mean NDCG@n scores across the 3 ranking methods. 

| Ranking Method                | Mean NDCG@3   | Mean NDCG@5   | Mean NDCG@10  |
| :--------------------------   | -----------:  | ----------:   | -----------:  |
| Average                       | 0.819         | 0.847         | 0.935         |
| Discounted Reward Only        | 0.837         | 0.861         | 0.939         |
| Discounted Reward with Adjustment by Opposite Query   | 0.837 | 0.862 | 0.941 |

<p align="center"><i>Table 1: NDCG@n scores across the 3 ranking methods</i></p>

The visualization below on the left **(Figure 6, left)**  shows the mean NDCG scores with confidence intervals across the three ranking methods, whereas the one on the right **(Figure 6, right)** shows the best ranking method (with highest mean value) for the 9 queries. 


<p align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/eval_analysis_chart.png?raw=true" alt="Evaluation Result Analysis"/>
</p>
<p align="center"><i>Figure 6: (Left) Mean NDCG across the three ranking methods. (Right) Top ranking method for the 9 queries</i></p> 


From the table and visualizations above, we can notice that generally the *Method 2) Discounted Reward Only* and *Method 3) Discounted Reward with Adjustment by Opposite Query* performed better than the *Method 1) Average*. However, *Method 3* is just a tiny bit better than *Method 2* in NDCG@5 and NDCG@10. The 95% confidence intervals are very similar for all three methods within the same NDCG group. On the other hand, *Method 1* had a 37% (10/27) chance of getting the best results while for *Method 2* and *Method 3* they were 41% (11/27) and 22% (6/27) respectively.

Overall, while *Method 3* had the highest mean NDCG scores, in practice *Method 2* had almost a double chance to outperform *Method 3*. It means *Method 3* performed much better in some cases only but not all. 

### Mean Reciprocal Rank (MRR) 

Mean Reciprocal Rank (MRR)  is another measurement of the goodness of ranking by measuring how far down the ranking the first relevant document is. For details, please refer to this article - [Compute Mean Reciprocal Rank (MRR) using Pandas](https://softwaredoug.com/blog/2021/04/21/compute-mrr-using-pandas.html). The MRR result is shown in **Figure 7** below.

<p align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/mrr_box_plot.png?raw=true" alt="MRR across Ranking Methods"/>
</p>
<p align="center"><i>Figure 7: MRR across Ranking Methods</i></p> 

When considering Mean Reciprocal Rank (MRR), *Method 3) Discounted Reward with Adjustment by Opposite Query* performed the best, followed by *Method 1) Average* and *Method 2) Discounted Reward Only* performed the worst. Based on the box plot, user U1 was more inclined to rate a product higher up the resulting product list as "Most relevant" than the other two users (i.e. more lenient), while user U3 was more inclined to rate a product lower down the list as "Most relevant" (i.e more strict).

Overall, *Method 2* seems to have unanimously been deemed to have trouble in putting the most relevant product at the top of the list compared to the other two methods.


### Analysis 

*Method 2* and *Method 3* performed better than *Method 1* in terms of NDCG scores which is within our expectation since *Method 2 and 3* have considered the collective information from all reviews which can better represent human decision models. However, the Opposite Query of *Method 3* did not perform as good as we originally thought. 

Through investigation, we believe the reason boils down to a problem with our assumption. We assume semantic oppositeness means a complete inverse of semantic similarity in the computing domain, while we have tried to use this oppositeness measure to find the contradicting concepts in customer reviews with the opposite queries. 

We took an example to further illustrate the idea. Here we use this original query and its corresponding opposite query to create the visualizations below. 
- **Original Query**: _Long thin cotton socks for men, need to be breathable, even feeling cool for summer time._
- **Opposite Query**: _Short means having little length. Thick means having a greater than usual measure across. Unbreathable means not allowing air to pass through. Hot means having or giving out a great deal of heat._


<p align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/embedding_analysis.png?raw=true" alt="Embedding Analysis"/>
</p>
<p align="center"><i>Figure 8: (Left) Heatmap of the Original query, Opposite query and first 20 results. (Right) Adjectives and their Antonyms in the Embedding Space</i></p>


In **Figure 8**, on the the left it shows the heatmap of the embeddings (reduced to 10 dimensions by UMAP) for the original query, opposite query and the first 20 search results; whereas on the right it shows the scatter plot of the embeddings (reduced to 2 dimension by UMAP) for the adjectives and their corresponding antonyms used in the original query and the opposite query. 

From the heatmap, we can realize that while the original query and the opposite query have some differences, it is not as different as we expect. On the other hand, from the scatter plot we can see that an adjective and its antonym are not very far away from each other in the vector space. Look at _“hot and cool”_, they are actually very close together!

In fact, while world embeddings can capture semantic relationships between words very well, they may not always capture antonymy accurately. The reason why word embeddings may still find antonyms to be similar is because the way language is used in a context. For example, consider the phrases _“hot coffee”_ and _“cold tea”_. _“Hot”_ and _“cold”_ are antonyms, but they co-occur with similar contexts like beverages. As a result, the word embeddings might learn to place _“hot”_ and _“cold”_ proximity to each other, even though they have opposite meanings. 

This is the reason why our opposite query strategy did not perform as good as we originally expected. Having said that, since the adjectives and their antonyms do have certain distances in the embedding space, they still can help in identifying the contradicting concepts in the reviews and hence improving the search results.

There is an excellent paper [“Semantic Oppositeness for Inconsistency and Disagreement Detection in Natural Language” by Naida et al.](https://www.cs.uoregon.edu/Reports/PHD-202012-deSilva.pdf) covers the topic of semantic oppositeness in the computing domain in very details.<sup>[[6](https://github.com/gen-exody/nes/wiki/Customer-Review-Driven-Product-Search-Engine#references)]</sup> 


## Final Thoughts

### Ethical Consideration

Since our proposed search engine uses product description and customer reviews as the search content, results are inevitably biased to products with more reviews. In other words,  products with more reviews will have a higher probability to be returned. This can create a harmful effect for new products where they will have very low chances for being found and reached by customers.

However, as a matter of fact, this is a general problem for all search engines and recommender systems that the control of decision making is shifted from users to algorithms. Thus, bias and discrimination created from the data and algorithms can easily be raised to impact users in favoring or discriminating against certain products.

For our application, one potential solution to reduce this unfairness is to introduce randomness in the search engine where products with similar natures (according to the product descriptions) but no corresponding reviews can still be returned by chance. Another possible solution is from a product design perspective where we can recommend new products along with the search return results. 

### Future Improvement

In this project, we have built a product search engine which uses product descriptions and customer reviews as the content for search.  This provides a richer context for customers to find their targets on e-commerce web sites. We have also introduced our ranking algorithms for calculating the product level similarity scores which beat the baseline method using average. Although we found the opposite query method is not as effective as we originally thought due to the nature of how embeddings are built in the computing domain, still we have found semantic oppositeness has huge potential in improving information retrieval by separation of contradicting findings. 

For future work, we believe there is a need to design a more effective measure on semantic oppositeness such that embeddings can better represent the oppositeness between two words. On the other hand, our current product search engine implementation just directly uses similarity to come up with the result. We believe building a deep learning model with the embeddings of the product review context and the search query, where the embeddings have considered the semantic oppositeness measure mentioned above, could greatly improve the overall result. 

## Resources 

1. All our source codes including Jupyter notebooks and a Streamlit application can be found on Github https://github.com/gen-exody/nes

2. We have built a prototype application on Streamlit which allows users to search products over our evaluation data set - Apparel in 2015 with 10-14 reviews. https://nescapstone.streamlit.app/

## Statement of Work

- Elaine Chen -  evaluate search engine result, design and develop Streamlit application
- Gen Ho - generate project idea, data preprocessing and analysis, design solution architecture, design and development search algorithm, evaluate search engine result, analyze search engine result, report writeup, prepare poster
- Varshini Rana - define search engine result evaluation guideline, evaluate search engine result, calculate evaluation metrics and analyze of search engine result 


## References

[1] Davis Liang et al. (Sep 22, 2020). "Embedding-based Zero-shot Retrieval through Query Generation". https://arxiv.org/abs/2009.10270. Accessed Jul 10, 2023

[2] Ryan Greene et al. (Dec 15, 2022). "New and improved embedding model". https://openai.com/blog/new-and-improved-embedding-model. Accessed Jul 2, 2023

[3] baeldung. (Nov 24, 2022). "Euclidean Distance vs Cosine Similarity". https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity. Accessed Jun 30, 2023

[4] Ron Rotkopf. (Jan 4, 2021). "Less is more: An Intro to Dimensionality Reduction". http://dors.weizmann.ac.il/course/workshop2021/scRNA/Dimensionality_Reduction.pdf. Accessed Jul 3, 2023

[5] Aparna Dhinakaran. (Jan 25, 2023). "Demystifying NDCG - How to best use this important metric for monitoring ranking models". https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0. Accessed Jul 16, 2023 

[6] Naida et al. (Apr 27, 2021). "Semantic Oppositeness for Inconsistency and Disagreement Detection in Natural Language". https://www.cs.uoregon.edu/Reports/PHD-202012-deSilva.pdf. Accessed Jul 28, 2023


## Appendix

### Queries for Evaluation

Below table shows the 9 queries that we used in the search engine result evaluation. 

| Num.  | Original | Opposite |
| ---   | ---      | ---      |
| Q1    | Long thin cotton socks for men, need to be breathable, even feeling cool for summer time. | Short means having little length. Thick means having a greater than usual measure across. Unbreathable means not allowing air to pass through. Hot means having or giving out a great deal of heat. |
| Q2    | Wrinkle free chiffon blouse, sleek style, long sleeve, slim fit, with comfortable inside layer. |  Wrinkled means having many wrinkles. Clumsy means lacking grace in movement or posture. Short means having little length. Bulky means large and unwieldy. Loose means not fitting closely or tightly. |
| Q3    | DC or Marvel superhero costumes for adults. Comes with a mask.  Deluxe and high quality. Comfortable to wear. Washable and durable. | Inferior means of lower quality. Basic means of the simplest kind. Uncomfortable means causing discomfort. Unwashable means not able to be washed. Fragile means easily broken. |
| Q4    | Baby footed bodysuit, made with soft and comfortable material, keeps the baby snug and warm, washed and dried well without shrinking. | Rough means having an uneven or irregular surface. Uncomfortable means causing discomfort. Loose means not firmly or tightly fixed in place. Unwarm means not providing or feeling warmth. Shrinking means becoming smaller in size. |
| Q5    | Women's knee-length A-line dress, looks cutie, made with high quality and relatively thick material, fits slim body. | Unattractive means not pleasing in appearance. Low quality means inferior in nature. Thin material means having little thickness. Loose-fitting means not fitting closely. |
| Q6    | Men's lightweight packable down jacket, wind-Resistant, quilted design, snug fit, warm but breathable. | Heavy means having great weight. Unquilted means not having a quilted pattern. Loose means not tight or close. Cold means having a low temperature. Stuffy means not allowing air to circulate freely. |
| Q7    | Women's high-waisted leggings, tummy control for slimmer look, black color, moisture-wicking fabric, non-see-through. | Low-waisted means having a waistline that is lower than usual. Bulky means large and unwieldy. White means the color of light containing equal amounts of all visible wavelengths. Damp means slightly wet. Transparent means allowing light to pass through so that objects behind can be distinctly seen. |
| Q8    | Men's skinny fit stretch jeans, tight on legs but fits good on hips, soft, comfortable and durable. | Loose means not tight or taut. Unfitting means not suitable or appropriate. Rough means having an uneven or irregular surface. Uncomfortable means causing discomfort. Fragile means easily broken. |
| Q9    | Kid's Rain jacket with hood. Made from waterproof material that keeps dry even in heavy rain. Lightweight but able to keep body warm. |  Sodden means extremely wet. Non-waterproof means not impermeable to water. Heavy means having great weight. Unwarm means not warm. |

### Evaluating Rater Agreeability

The [Intra-Class correlation (ICC)](https://en.wikipedia.org/wiki/Intraclass_correlation) is a quantitative measure to assess the reliability of ratings by multiple subjects. The measure compares the variability of different ratings of the same subject to the total variation across all ratings and all subjects.

The [documentation of the pingouin library](https://pingouin-stats.org/build/html/generated/pingouin.intraclass_corr.html), which contains the Python implementation of ICC, can be referred to for more details.

<p align="center">
    <img src="https://github.com/gen-exody/nes/blob/master/resources/img/ICC.png?raw=true" alt="ICC Measurement"/>
</p>
<p align="center"><i>Figure 9: ICC Measurement</i></p>

The above result table provides the ICC measures for various flavors, the definitions for which are detailed in the [Wikipedia page for ICC](https://en.wikipedia.org/wiki/Intraclass_correlation). In this case, we will be focusing on ICC3k, which is delineated by the following criteria:

Two-way mixed: k fixed raters are defined. Each subject is measured by the k raters.
Average measures: the reliability is applied to a context where measures of k raters will be averaged for each subject.
Consistency: in the context of repeated measurements by the same rater, systematic errors of the rater are cancelled and only the random residual error is kept.

For the interpretation of the ICC measure, there are two scales:

**Cicchetti scale:**

* Less than 0.40: poor.
* Between 0.40 and 0.59: fair.
* Between 0.60 and 0.74: good.
* Above 0.75: excellent.

Our ICC value of 0.834 implies that our three raters' inter-rater agreement is "excellent" on the Cicchetti scale with the 95% confidence intervals of 0.8 and 0.87 enabling it to stay within that category.

**Koo and Li scale:**

* Less than 0.50: poor
* Between 0.50 and 0.74: moderate
* Between 0.75 and 0.89: good
* Above 0.90: excellent

Our ICC value of 0.834 implies that our three raters' inter-rater agreement is "good" on the Koo and Li scale with the 95% confidence intervals of 0.8 and 0.87 enabling it to stay within that category.
