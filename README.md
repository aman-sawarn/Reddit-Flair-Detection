# Reddit-Flair-Detection
## Problem Statement
Given a Reddit URL, predict Flairs from it

## Approach to solve the Problem
1. Scrap data from reddit using the reddit Praw API
2. Perform EDA on it and Understand data
3. Clean Data
4. Vectorize Data using Bag of Vectors and TFIDF
5. Build Models using ML models
6. Save the vectorizer weights and model weights

### Confusion Matrix
![alt text](https://github.com/sawarn69/Reddit-Flair-Detection/blob/master/confusion.png)


## How to reproduce the results
1. Run the Download+EDA.ipynb notebook
2. It will download the Data and put it into a .csv file
3. After that run the Models.ipynb notebook
4. It will dump the vectorizer and model into .pkl files
5. If you want to make it retrainable, just download it as .py file


## Deployment
1. The Models.ipynb dumps two pickle files
2. app.py uses these two .pkl files to predict new results
3. To learn how to productionize the model, use this link: https://www.youtube.com/watch?v=1umQhC2iWdY&t=536s

### Link to Heroku App
https://reddit-flair-detector-aman.herokuapp.com/index

