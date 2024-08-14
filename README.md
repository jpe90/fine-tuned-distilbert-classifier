This project implements a text classification system using DistilBERT to categorize customer messages into three categories: feedback, inquiries, and refunds.

## Project Structure

- `train.py`: Script for training the DistilBERT model on the provided dataset.
- `classify.py`: Script for using the trained model to classify new text inputs.
- `requirements.txt`: List of Python dependencies required for the project.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/jpe90/fine-tuned-distilbert-classifier.git
   cd fine-tuned-distilbert-classifier
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

Training data is located in the `data` directory:
- `feedback.txt`: Contains customer feedback messages
- `inquiries.txt`: Contains customer inquiry messages
- `refunds.txt`: Contains customer refund request messages

Each file contains one message per line.

## Training the Model

To train the model, run:

```
python train.py
```

## Using the Classifier

To use the trained model for classification, run:

```
python classify.py
```
