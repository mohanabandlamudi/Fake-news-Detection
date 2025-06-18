# Fake News Detection using Natural Language Processing

This project implements a machine learning-based solution for detecting fake news articles using Natural Language Processing (NLP) techniques. The system analyzes text content to classify news articles as either real or fake.

## Features

- Text preprocessing and cleaning
- Feature extraction using NLP techniques
- Machine learning model for classification
- Evaluation metrics for model performance
- Interactive Jupyter notebook for analysis and visualization

## Working

- Loads and merges real and fake news datasets
- Cleans and preprocesses text using NLP techniques
- Extracts features using:
  - CountVectorizer
  - TF-IDF
  - Word2Vec
- Trains models using Logistic Regression
- Evaluates models using classification reports and confusion matrices
- Visualizes word clouds and label distribution

---

## Libraries Used

| Library | Purpose |
|--------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib`, `seaborn`, `wordcloud` | Visualization |
| `nltk` | Tokenization, stopword removal, lemmatization |
| `sklearn` | Machine learning and feature extraction |
| `gensim` | Word2Vec embeddings |

---

##  Data Description

- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains true news articles

After loading:
- A `label` column is added:
  - `1` = True news
  - `0` = Fake news

---

##  Text Preprocessing

###  Steps performed:

1. **Lowercasing**: `Text → text`
2. **Removing punctuation**: Removes `. , ! ? etc.`
3. **Removing stopwords**: Removes common words like “the”, “is”
4. **Removing HTML tags and URLs**
5. **Removing words with digits**
6. **Lemmatization**: Converts words to their dictionary form (`running → run`)

## Requirements

To run this project, you'll need the following Python packages:

```bash
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
jupyter
```

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Project Structure

- `text_processing_ml.ipynb`: Main Jupyter notebook containing the implementation
  - Data preprocessing
  - Feature engineering
  - Model training and evaluation
  - Results visualization

## Usage

1. Clone this repository:
```bash
git clone [repository-url]
cd FakeNewsDetection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook text_processing_ml.ipynb
```

4. Run the cells in the notebook to:
   - Load and preprocess the data
   - Train the model
   - Evaluate the results
   - Visualize the findings

## Model Evaluation

Once the models are trained, we evaluate their performance using the **test set**. The following metrics are used to measure how well the model performs:

### Evaluation Metrics

| Metric        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Accuracy**  | Proportion of correctly predicted labels (true positives + true negatives) |
| **Precision** | How many predicted positives are actually correct                           |
| **Recall**    | How many actual positives were correctly predicted                          |
| **F1-Score**  | Harmonic mean of precision and recall (balance between them)                |


Created by @mohanabandlamudi
