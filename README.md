# AI-GenAI_projects
# 1. multi-ML_comparison_Flask_frontend

Python project, where I used modules such as Plotly, scikit-learn, Flask, JSON, Pandas, Numpy, etc. to analyze the relationship between data from databases and calculate predictions using models such as linear regression, assembly methods, etc. It display the graphs in a Web app created in Flask

multi-ML_comparison_Flask_frontend.py is a multi-ML choice application, it uses of http://api.worldbank.org public data for model training. Server and Frontend code: index.html is generated to plot the 3d graph createad with Flask that shows real and predicted values. It also compares the models performances

# 2. LLM use cases projects

2.1. app contador de tokens_transf_huggingface.py:

<b> Tokens counter </b>. Python code to  calculate the number of tokens of a text using the Hugging Face library

2.2. app RAG_transformer_huggingface.py:

<b> RAG application </b> . The code uses two different models to implement the RAG system:

For the calculation of embeddings: The model used is 'paraphrase-MiniLM-L6-v2', which is a variant of the all-MiniLM-L6-v2 model. This model maps sentences and paragraphs to a dense 384-dimensional vector space1. 
It is used to encode both the knowledge base and the user's query into vectors, allowing similarity search to retrieve the relevant context.

For the generation of the response: The model used is 'google/flan-t5-base', which is a larger version of the T5 (
Text-to-Text Transfer Transformer) model developed by Google. This model is used to generate the final response based on the retrieved context and the user's question.

2.3. app sentiment analysis_transf_huggingface.py:

 <b> Sentiment analysis </b>. This code performs the sentiment analysis of a given text using the Hugging Face Transformers library. 

2.4. app text summary_transformers_huggingface.py:

 <b> Text summarization </b>. This code performs text summarization of a given text using the Hugging Face Transformers library. 

# 3. ML use case: price prediction simulation of three scenarios of demand/competition

ML price prediction_demand_competition scenarios.py: ML application to predict the price given 3 scenarios of competition (severe, mid,low). index.html is generated to plot the 3d graph created with Flask that shows the predicted price evolution

multi-ML_comparison_Flask_frontend.py is a multi-ML choice Flask application, it uses of http://api.worldbank.org public data for model training. Server and Frontend code: index.html is generated to plot the 3d graph with real and predicted values. It also compares the predicted model performance

# 4. ML for drug efficacy prediction from random clinical trial data

ML_drug_efficacy prediction_clinical_study_SQL_randomDB.py: # ML model for clinical study - prediction of the druf efficacy based on the dose and age. Usage of a linear regression model.
# WARNING: the model is trained with a randon dataset of clinical study created just for the purpose of validating the end-to-end process of ML model creation
