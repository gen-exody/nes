# About the source dataset

The dataset we used is the Amazon US Customer Reviews Dataset, which contains product reviews over two decades since 1995. For this project, we will focus on the Apparel product category.

The source data for this projet can be downloaded at Kaggle at https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Apparel_v1_00.tsv

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

### License

By accessing the Amazon Customer Reviews Library ("Reviews Library"), you agree that the Reviews Library is an Amazon Service subject to the Amazon.com Conditions of Use (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&nodeId=508088) and you agree to be bound by them, with the following additional conditions:

In addition to the license rights granted under the Conditions of Use, Amazon or its content providers grant you a limited, non-exclusive, non-transferable, non-sublicensable, revocable license to access and use the Reviews Library for purposes of academic research. You may not resell, republish, or make any commercial use of the Reviews Library or its contents, including use of the Reviews Library for commercial research, such as research related to a funding or consultancy contract, internship, or other relationship in which the results are provided for a fee or delivered to a for-profit organization. You may not (a) link or associate content in the Reviews Library with any personal information (including Amazon customer accounts), or (b) attempt to determine the identity of the author of any content in the Reviews Library. If you violate any of the foregoing conditions, your license to access and use the Reviews Library will automatically terminate without prejudice to any of the other rights or remedies Amazon may have. https://s3.amazonaws.com/amazon-reviews-pds/license.txt