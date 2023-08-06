import csv
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def load_data_from_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def word_features(sentence, index):
    word = sentence[index]['word']
    features = {
        'word': word,
        'is_titlecase': word.istitle(),
        'is_numeric': word.isdigit(),
    }
    return features

def data_to_features(data):
    X = []
    y = []
    sentence = []
    labels = []
    for row in data:
        word = row['word']
        tag = row['tag']
        if word == '.':
            if sentence:
                X.append([word_features(sentence, i) for i in range(len(sentence))])
                y.append(labels)
                sentence = []
                labels = []
        else:
            sentence.append({'word': word})
            labels.append(tag)
    return X, y


def train_crf_model(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=50,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf


def extract_disease_spans(labels):
    disease_spans = []
    start = None
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if start is not None:
                disease_spans.append((start, i - 1))
            start = i
        elif label == 'O':
            if start is not None:
                disease_spans.append((start, i - 1))
                start = None
    if start is not None:
        disease_spans.append((start, len(labels) - 1))

    return disease_spans


def evaluate_predictions(y_true, y_pred):
    num_exact_matches = 0
    num_partial_matches = 0
    total_sentences = len(y_true)

    for true_labels, pred_labels in zip(y_true, y_pred):
        true_disease_spans = extract_disease_spans(true_labels)
        pred_disease_spans = extract_disease_spans(pred_labels)

        if true_disease_spans == pred_disease_spans:
            num_exact_matches += 1
        else:
            for span in pred_disease_spans:
                if span in true_disease_spans:
                    num_partial_matches += 1
                    break

    exact_match_score = num_exact_matches / total_sentences
    partial_match_score = (num_exact_matches + num_partial_matches) / total_sentences

    precision = num_exact_matches / (num_exact_matches + num_partial_matches)
    recall = num_exact_matches / total_sentences
    f1_score = (2 * precision * recall) / (precision + recall)

    return exact_match_score, partial_match_score, f1_score


def main():
    print("Loading data...")
    train_data = load_data_from_csv('data/train.csv')
    val_data = load_data_from_csv('data/val.csv')
    test_data = load_data_from_csv('data/test.csv')

    print("Converting data to features...")
    X_train, y_train = data_to_features(train_data)
    X_test, y_test = data_to_features(test_data)

    print("Training the CRF model...")
    crf_model = train_crf_model(X_train, y_train)

    model_filename = 'crf_model.pkl'
    joblib.dump(crf_model, model_filename)
    print(f"Trained model saved to {model_filename}")

    print("Making predictions...")
    y_pred = crf_model.predict(X_test)

    print("Evaluating predictions...")
    exact_match_score, partial_match_score, f1_score = evaluate_predictions(y_test, y_pred)

    print(f"Exact Match Score: {exact_match_score}")
    print(f"Partial Match Score: {partial_match_score}")
    print(f"F1 Score: {f1_score}")


if __name__ == '__main__':
    main()

