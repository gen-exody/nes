# About the source dataset

The dataset we used is the Amazon US Customer Reviews Dataset, which contains product reviews over two decades since 1995. For this project, we will focus on the Apparel product category.

The source data for this projet can be downloaded at [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Apparel_v1_00.tsv). More detailed information including the license model can be found [here](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). 

Please put the downloaded data file `amazon_reviews_us_Apparel_v1_00.tsv.gz` and put in this folder in order to run the `source_data_analysis_preprocessing.ipynb` notebook. 

### Data Definition 

1. marketplace
    - 2 letter country code of the marketplace where the review was written.
2. customer_id
    - Random identifier that can be used to aggregate reviews written by a single author.
3. review_id
    - The unique ID of the review.
4. product_id
    - The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
5. product_parent
    - Random identifier that can be used to aggregate reviews for the same product.
6. product_title
    - Title of the product.
7. product_category
    - Broad product category that can be used to group reviews
8. star_rating
    - The 1-5 star rating of the review.
9. helpful_votes
    - Number of helpful votes.
10. total_votes
    - Number of total votes the review received.
11. vine
    - Review was written as part of the Vine program.
12. verified_purchase
    - The review is on a verified purchase.
13. review_headline
    - The title of the review.
14. review_body
    - The review text.
15. review_date
    - The date the review was written.
