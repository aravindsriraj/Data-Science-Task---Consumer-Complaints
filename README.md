# Data-Science Task - Consumer Complaints

Text Classification on Consumer complaints dataset into following categories
- Credit reporting, repair, or other
- Debt collection
- Consumer Loan
- Mortgage

Dataset link - https://catalog.data.gov/dataset/consumer-complaint-database

| Date received | Product | Sub-product | Issue | Sub-issue | Consumer complaint narrative | Company public response | Company | State | ZIP code | Tags | Consumer consent provided? | Submitted via | Date sent to company | Company response to consumer | Timely response? | Consumer disputed? | Complaint ID |
| ------------- | ------- | ----------- | ----- | --------- | ---------------------------- | ----------------------- | ------- | ----- | -------- | ---- | ------------------------- | ------------- | ---------------------- | ------------------------- | ---------------- | ----------------- | ------------ |
| 2023-08-25 | Credit reporting or other personal consumer re... | Credit reporting | Incorrect information on your report | Information belongs to someone else | NaN | NaN | EQUIFAX, INC. | FL | 33009 | NaN | NaN | Web | 2023-08-25 | Closed with explanation | Yes | NaN | 7523056 |
| 2023-08-25 | Credit reporting or other personal consumer re... | Credit reporting | Improper use of your report | Credit inquiries on your report that you don't... | NaN | NaN | EQUIFAX, INC. | MI | 48234 | NaN | NaN | Web | 2023-08-25 | Closed with explanation | Yes | NaN | 7523057 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Problem with a credit reporting company's inve... | Their investigation did not fix an error on yo... | NaN | NaN | EQUIFAX, INC. | GA | 30034 | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7446803 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Problem with a credit reporting company's inve... | Was not notified of investigation status or re... | NaN | NaN | EQUIFAX, INC. | CO | 80249 | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7446804 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Problem with a credit reporting company's inve... | Their investigation did not fix an error on yo... | NaN | NaN | TRANSUNION INTERMEDIATE HOLDINGS, INC. | NY | XXXXX | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7442566 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Improper use of your report | Credit inquiries on your report that you don't... | NaN | NaN | EQUIFAX, INC. | CA | 95648 | NaN | Other | Web | 2023-08-23 | In progress | Yes | NaN | 7446815 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Problem with a credit reporting company's inve... | Their investigation did not fix an error on yo... | NaN | NaN | EQUIFAX, INC. | NY | XXXXX | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7446610 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Improper use of your report | Reporting company used your report improperly | NaN | NaN | Experian Information Solutions Inc. | OR | 97209 | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7442804 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Incorrect information on your report | Information belongs to someone else | NaN | NaN | Experian Information Solutions Inc. | NY | 11550 | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7442745 |
| 2023-08-23 | Credit reporting, credit repair services, or o... | Credit reporting | Incorrect information on your report | Information belongs to someone else | NaN | NaN | Experian Information Solutions Inc. | GA | 30228 | NaN | NaN | Web | 2023-08-23 | In progress | Yes | NaN | 7442823 |


Steps Followed:

1. Explanatory Data Analysis and Feature Engineering
2. Text Pre-Processing
3. Selection of Multi Classification model
4. Comparison of model performance
5. Model Evaluation
6. Prediction

### Explanatory Data Analysis and Feature Engineering

**Understanding the Data:**

- Started by loading the dataset using pd.read_csv and displayed the top 10 rows using df.head(10).
- Checked the shape of the dataset using df.shape, which showed that it contains 4,051,252 rows and 18 columns.
- Used df.info() to get information about the data types and memory usage of each column.

**Handling Missing Values:**

- Checked for missing values using df.isnull().any() and found several columns with missing data.
- Calculated the number of missing values in each column using df.isnull().sum().
- Created a list of columns with missing values and displayed them using print(missing_cols).
- Visualized the missing values in columns using a bar plot.

![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/3862093c-5bef-42b1-b267-0a30e7ed28f9)


**Data Categorization:**

- Noticed that there were many product categories, and you decided to group similar categories into four main categories: Credit reporting, repair, or other; Debt collection; Consumer Loan; and Mortgage.
- Updated the "Product" column to reflect these four main categories using df['Product'].replace.
  
**Data Visualization:**

- Visualized the distribution of complaints across different product categories using a count plot.
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/55bb3b23-f00e-42f7-8381-20bad0ae7e39)

- Explored the distribution of company responses to consumer complaints using another count plot.
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/c12f9a95-efee-4302-8904-1f2b805f6b6a)

- Analyzed whether consumers disputed complaints and visualized it using a count plot.
- Identified the top 5 disputed and non-disputed companies and created bar plots to visualize their frequencies.
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/b46cce0c-944f-415c-8a70-505f1a457f74)
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/16131c91-4329-4a44-b097-2d2fd01cd6ff)
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/d6c031a8-f608-4f20-938a-f26f7071ffa6)

**Temporal Analysis:**

- Converted the "Date received" column to the pandas datetime format.
- Extracted the year and month from the "Date received" column and added them as separate columns.
- Visualized Number of Disputes (Top 4 Companies) from 2011-2017
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/0af5c6de-5ab0-490f-8711-ab0ff66f1374)

- Grouped the data by year and whether the consumer disputed the complaint.
- Created a DataFrame to store the counts of complaints for each company, year, and dispute status.
- Analyzed the top 4 companies with the most consumer disputes over the years and plotted the trends.
- Plotted the top 10 complaints in the "Sub-product" category using a bar plot.
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/2571a22a-b6a0-422a-b09c-00502745e5b1)

- Visualized the top 10 companies with the most complaints using a bar plot.
![image](https://github.com/aravindsriraj/Data-Science-Task---Consumer-Complaints/assets/60252521/4b37d3bb-36bd-46ea-a654-6c5755cc34a6)


**Preparing Data for Text Classification:**

- Recognized that text classification on consumer complaints required only two columns: "Product" and "Consumer complaint narrative."
- Created a new DataFrame (new_df) with these two columns.
- Checked for missing values in the "Consumer complaint narrative" column and dropped rows with missing complaints using new_df.dropna().

## Text Pre-processing

**Converting to Lowercase:**

Started by converting all text in the "Complaint" column to lowercase using the .apply(lambda x: x.lower()) method. This step is essential for standardizing text and ensuring that words are treated consistently regardless of their capitalization.

**Noise Removal:**

- Noise in text data includes unnecessary characters or elements that do not contribute to the analysis. Implemented several functions to remove noise:
 - remove_URL: Removes URLs from the text using regular expressions.
 - remove_html: Removes HTML tags and entities (e.g., &) from the text.
 - remove_non_ascii: Removes non-ASCII characters from the text.
 - You applied these noise removal functions to the "clean_text" column using .apply().

**Special Character Removal:**

- Special characters can include symbols, emojis, and other graphic characters that may not provide meaningful information for text classification. You used the remove_special_characters function to remove these characters using regular expressions.
- This step helps reduce the complexity of the text data.
  
**Punctuation Removal:**

- Removed punctuation marks from the text using the remove_punct function, which utilizes the string.punctuation library to identify and remove punctuation characters.
- This step is useful for text analysis as it eliminates punctuation that does not convey semantic information.

**Data Reduction:**

After applying all the pre-processing steps, retained only the "Product" and "clean_text" columns in the DataFrame new_df, effectively reducing the data size and complexity.

**Text Data Statistics:**

Calculated various statistics about the cleaned text data, including:
- Average number of words in a sentence.
- Maximum number of words in a sentence.
- Minimum number of words in a sentence.
- Median number of words in a sentence.
These statistics provide insights into the distribution and characteristics of the text data.

**Encoding Target Variable:**

Encoded the target variable "Product" using label encoding from the scikit-learn library. This step converts categorical labels into numerical values, making it suitable for machine learning algorithms.

## Selection of Multi-Class Classification Models and Model Evaluation:

Trained Different ML models and Compared and evaluated the model performance

- Logistic Regression
- Naive Bayes
- SGD Classifier
- LSTM (Neural networks)
- BERT (Transformers)

## Comparison of Model Performance

**Logistic Regression:**

Multi-Class ROC-AUC Score: 0.9733
Accuracy: 0.897

**Naive Bayes:**

Multi-Class ROC-AUC Score: 0.9354
Accuracy: 0.7837

**SGD Classifier:**

Multi-Class ROC-AUC Score: 0.9494
Accuracy: 0.8188

**Linear SVC:**

Multi-Class ROC-AUC Score: 0.9738
Accuracy: 0.9115

**Comparison:**

- Both Logistic Regression and Linear SVC achieved the highest multi-class ROC-AUC scores of approximately 0.9733 and 0.9738, respectively.
- Linear SVC achieved the highest accuracy of 0.9115, followed by Logistic Regression with an accuracy of 0.897.
- Naive Bayes had the lowest ROC-AUC score (0.9354) and accuracy (0.7837) among the models.
- SGD Classifier had an intermediate ROC-AUC score (0.9494) and accuracy (0.8188).

**Conclusion:**

- Based on the provided metrics, the Linear SVC model outperforms the other models in terms of both multi-class ROC-AUC and accuracy. It achieved the highest ROC-AUC score and the highest accuracy,  indicating its effectiveness in distinguishing between classes and making correct predictions.
- Logistic Regression also performed well, with a high ROC-AUC score and accuracy.
- SGD Classifier, while achieving decent performance, falls slightly behind Logistic Regression and Linear SVC.
- Naive Bayes, although simple, did not perform as well as the other models in this specific task.
  
Overall, if you prioritize both ROC-AUC and accuracy, Linear SVC appears to be the better choice for this multi-class text classification task. However, it's essential to consider other factors such as model interpretability, computational resources, and the specific requirements of your application when choosing the final model.

## Prediction

BERT is comparatively better than other models. Fine tuned BERT for text classification


