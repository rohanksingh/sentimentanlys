import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd 
import yfinance as yf 
import datetime
from datetime import date, timedelta
import numpy as np
import logging
# import torch
# from transformers import AutoModelForSequenceClassification , AutoTokenizer



nltk.download('vader_lexicon')

def calculate_sentiment_score(text):
    sia= SentimentIntensityAnalyzer()
    sentiment_scores= sia.polarity_scores(text)
    return sentiment_scores['compound']

stock='TSLA'

# Stock data pull date 
today= datetime.date.today()
days=30
start=today -timedelta(days)
start= start.strftime("%Y-%m-%d")
end=today.strftime("%Y-%m-%d")
# print(start)
# print(end)

# Sentiment analysis start and end date 
sentimentstrdt=today -timedelta(days)
sentimentstrdt=sentimentstrdt.strftime("%Y-%m-%d")
sentimentenddt=today -timedelta(1)
sentimentenddt= sentimentenddt.strftime("%Y-%m-%d")
# print(sentimentstrdt)
# print(sentimentenddt)

stock_data= yf.download(stock,  start=start , end=end)
stock_data['Return']= stock_data['Close'].pct_change()


stock_data= pd.DataFrame(stock_data)
stock_data.info()
stock_data['Date']= stock_data.index  
stock_data.info()
stock_data.to_csv('stock_data.csv', index=False)
# stock_data= stock_data.reset_index()
# stock_data.info()

# stock_data.head()

import newsapi
from newsapi import NewsApiClient

newsapi= NewsApiClient(api_key='6a0ea5549aa04467b5400176d3bd096e')
news_data= newsapi.get_everything(q=stock, language='en',
                                  from_param=sentimentstrdt,
                                  to=sentimentenddt, page_size=30)

news_headlines= [article['title'] for article in news_data['articles']]

sentiment_scores= [calculate_sentiment_score(headline) for headline in news_headlines]

# sentiment_scores

news_sentiment_data= pd.DataFrame({'Date': pd.date_range(start=sentimentstrdt, end=sentimentenddt),
                                   'Sentiment_Score': sentiment_scores})

# news_sentiment_data.info()
# news_sentiment_data['Date']= news_sentiment_data.index 
news_sentiment_data.to_csv('news_sentiment_data.csv', index=False)

stock_data1= pd.read_csv('stock_data.csv')
# stock_data1.info()
news_sentiment_data1= pd.read_csv('news_sentiment_data.csv')

merged_data = pd.merge(stock_data1, news_sentiment_data1, on= 'Date', how='inner')

# print(merged_data)

merged_data.to_csv('merged_data.csv', index=False)


# Getting some statistics from the data 
merged_data['Mean']= np.mean(merged_data['Return'])
merged_data['Min'] = np.min(merged_data['Return'])
merged_data['Max'] = np.max(merged_data['Return'])
merged_data['Median'] = np.median(merged_data['Return'])

# Recomendation based on the statiscs and the return , Sentiment scores 

merged_data['recommendation'] = np.where(
    (merged_data['Return'] >merged_data['Mean']) & (merged_data['Sentiment_Score'] >=0), 'Buy', 
    np.where((merged_data['Return']> merged_data['Mean']) & (merged_data['Sentiment_Score'] <0), 'Hold' , 
             np.where((merged_data['Return']< merged_data['Mean']) & (merged_data['Sentiment_Score'] >0), 'Hold', 'Sell')))


# Final merged result
merged_data.to_csv('merged_data_rec.csv', index=False)

# Function for chatbot to show final result

def chatbot_final_result(merged_data):
    
    last_stock= merged_data.iloc[-1]
    # print(last_stock) # can use later 
    # stock_recm= last_stock['recommendation']
    # print(stock_recm)
    recommendation= last_stock['recommendation']
    # print(recommendation)
    
    return f"The recommendation from the {stock}'s recent price data is to {recommendation}."

chatbot_response= chatbot_final_result(merged_data)
# print(chatbot_response)


# To make the chatbot interactive, we can include a basic input loop:

def chatbot():
    print("Welcome to the stock Performance Chatbot!")
    print("Type 'show final result' to see  the last stock's performance.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ").lower()
        if user_input == "show final result":
            print(f"Bot: {chatbot_final_result(merged_data)}")
        elif user_input == "exit":
            break
        else:
            print("Bot: I didn't understand that. Try 'show final result' or 'exit'.")

# testing the module directly 

if __name__=="__main__":
    
    finaldata=merged_data
    chatbot()
    import matplotlib.pyplot as plt

    plt.scatter(merged_data['Sentiment_Score'], merged_data['Return'])
    plt.xlabel('Sentiment Score')
    plt.ylabel('Stock Return')
    plt.title('Relationship between News Sentiment and Stock Returns')
    plt.show()
    
            
# chatbot()
        
            


