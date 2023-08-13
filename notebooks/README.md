# Notebooks for Data Preparation and Analysis

This set of notebooks is recommended to be read in the following order.

1. source_data_analysis_preprocessing.ipynb
2. create_embedding.ipynb
3. create_index.ipynb
4. product_type_analysis.ipynb
5. design_opposite_query.ipynb
6. design_product_level_similarity_score.ipynb
7. generate_search_result.ipynb
8. generate_evaluation_result.ipynb
9. evaluation_analysis.ipynb
10. opposite_query_analysis.ipynb
11. antonym_analysis.ipynb


In order to run the Jupyter notebooks, you need to have your own [OpenAI subscription](https://openai.com/).
Please create a file named `nes.ini` in this folder with following content. 
```
[OpenAI]
api_key = [your_openai_key]
```

Also, please run below commond to install project dependencies
```
pip install -r requirements.txt
```