# CGT Classification Project

This project involves training and evaluating various machine learning models for classification tasks using the CGT
dataset. The models include a BERT-based model, a Feedforward Neural Network (FFNN), an LSTM-based classifier, and a
pool of traditional classifiers.

Companion workshop paper presented at OVERLAY
2024: ["A Comparison of Machine Learning Techniques for Ethereum Smart Contract Vulnerability Detection"](https://ceur-ws.org/Vol-3904/paper15.pdf).

## Installation

To get started with this project, install the required libraries using `pip`:

```bash
pip install numpy pandas torch scikit-learn tqdm transformers xgboost
```

## Dataset

The dataset used in this project is the CGT dataset, which can be found [here](https://github.com/gsalzer/cgt).

### Download and Setup

1. Clone the dataset repository:

```bash
git clone https://github.com/gsalzer/cgt.git
```

2. Place the cloned repository in the appropriate directory structure:

```plaintext
project_directory/
├── dataset/                  # Cloned CGT dataset repository
└── your_project_files/   # Your project files
```

## Configuration

### Device Setup

The project uses a GPU if available:

```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### Random Seed

Random seeds are set for reproducibility:

```python
RANDOM_SEED = 0
```

### Dataset Path

Specify the path to the dataset:

```python
PATH_TO_DATASET = os.path.join("..", "dataset", "cgt")
```

### Training Configurations

- **Model Type:** BERT (`microsoft/codebert-base`)
- **Max Features:** 500
- **Batch Size:** 1
- **Number of Folds:** 10
- **Number of Epochs:** 25
- **Number of Labels:** 20
- **Learning Rate:** 0.001
- **Test Size:** 0.1

### File Configurations

Handle different file types: `source`, `runtime`, and `bytecode`.

### Log Directory

Logs are stored in a directory created if it doesn't already exist:

```python
LOG_DIR = os.path.join("log", FILE_TYPE)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
```

## Preprocessing

### Preprocessing Functions

Functions are provided to preprocess hex data and Solidity code:

- **Hex Data Preprocessing:** Converts hex data to a readable byte string.
- **Solidity Code Preprocessing:** Removes comments and blank lines.

### Data Initialization

Initialize inputs, labels, and groundtruth from the dataset:

```python
inputs, labels, gt = init_inputs_and_gt(dataset)
```

### Setting Labels

Set up labels based on groundtruth:

```python
labels = set_labels(dataset, labels, gt)
```

### Vectorization

TF-IDF vectorizer is used to convert text data into numerical features:

```python
VECTORIZER = TfidfVectorizer(max_features=MAX_FEATURES)
```

## Models

### BERTModelTrainer

Handles training and evaluation of a BERT-based model. Uses the `transformers` library to load a BERT model for sequence
classification.

### FFNNClassifier

A simple feedforward neural network with three fully connected layers for classification tasks.

### LSTMClassifier

An LSTM-based model for text classification, initialized with pretrained GloVe embeddings.

### Load GloVe Embeddings

Download the GloVe embeddings from [Kaggle](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt) and extract
the file to the appropriate directory:

```plaintext
project_directory/
├── asset/
│   └── glove.6B.100d.txt  # GloVe embeddings file
└── your_project_files/    # Your project files
```

Load the GloVe embeddings:

```python
glove_embeddings = load_glove_embeddings(os.path.join("..", "asset", "glove.6B.100d.txt"))
```

## Training and Evaluation

### Trainer Class

Handles the training and evaluation of a neural network model.

### CrossValidator Class

Performs k-fold cross-validation of a model, training and evaluating it across multiple folds.

### ClassifiersPoolEvaluator Class

Evaluates a pool of classifiers using TF-IDF features and k-fold cross-validation.

### Initializing and Training Models

1. **BERT Model:**
    ```python
    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL_TYPE, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model.config.problem_type = "multi_label_classification"
    model.to(DEVICE)

    tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL_TYPE, ignore_mismatched_sizes=True)

    x, y = tokenizer(INPUTS, add_special_tokens=True, max_length=512, return_token_type_ids=False, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt'), LABELS

    x_train, x_test, y_train, y_test = train_test_split(x['input_ids'], y, test_size=TEST_SIZE)
    train_masks, test_masks = train_test_split(x['attention_mask'], test_size=TEST_SIZE)

    train_data = TensorDataset(x_train, train_masks, torch.tensor(y_train).float())
    test_data = TensorDataset(x_test, test_masks, torch.tensor(y_test).float())

    CrossValidator(BERTModelTrainer(model), train_data, test_data).k_fold_cv(log_id="bert")
    ```

2. **FFNN Model:**
    ```python
    model = FFNNClassifier()

    x = torch.FloatTensor(VECTORIZER.fit_transform(INPUTS).toarray())
    y = torch.FloatTensor(LABELS)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    CrossValidator(Trainer(model), train_data, test_data).k_fold_cv(log_id="ffnn")
    ```

3. **LSTM Model:**
    ```python
    embeddings = load_glove_embeddings("path_to_glove_file")
    vocab_size = len(embeddings)
    embedding_dim = len(next(iter(embeddings.values())))
    hidden_dim = 128

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, np.array(list(embeddings.values())))

    tokenizer = SomeTokenizer(vocab=embeddings.keys())  # Assume you have a tokenizer that converts text to sequences of indices

    sequences = tokenizer.texts_to_sequences(INPUTS)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # Assume you pad sequences to a maximum length
    x = torch.tensor(padded_sequences)
    y = torch.FloatTensor(LABELS)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    CrossValidator(Trainer(model), train_data, test_data).k_fold_cv(log_id="lstm")
    ```

4. **Classifiers Pool Evaluation:**
    ```python
    evaluator = ClassifiersPoolEvaluator()
    evaluator.pool_evaluation()
    ```

### Evaluation and Results

Metrics such as precision, recall, and F1 score are calculated and saved to a CSV file.