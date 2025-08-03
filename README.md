# Sentiment Analysis Classifier
A Python project for sentiment analysis using multiple models: SVM, LSTM, and GPT-2. You can train and evaluate any combination of models on a movie review dataset like IMDB.

## 🛠 Requirements

### Install requirements locally:

```bash
pip install -r requirements.txt
```
# Or
## 🐳 Docker Support

### Build the Docker Image

```bash
docker build -t sentiment-analysis .
```

### Run the Container (with GPU support)

```bash
docker run \
--gpus all \
-it \
--name sentiment-analysis-container \
-v "$(pwd)":/app \
-w /app sentiment-analysis \
/bin/bash
```
---

## Usage
Modify the `run.sh` script to:
- Choose between models 'svm', 'lstm', 'gpt2' or 'all'
- Specify the number of shots for gpt2
- Add your input review for gpt2
### Then run the script
```bash
sh run.sh
```


## Script Output

Each model prints:

* Classification report
* False Positive Rate (FPR)
* False Negative Rate (FNR)

GPT2 prints the review result:
* Positive/ Negative

## Contact
For any inquiries or support, please contact yahyadaqour@gmail.com.
