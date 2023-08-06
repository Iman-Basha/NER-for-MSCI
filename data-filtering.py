import pandas as pd
import os

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def remove_most_frequent_words_with_tag_O(data, most_frequent_words_O):
    # Convert all words to lowercase
    data['word'] = data['word'].str.lower()

    # Remove the most frequent words with the tag "O" from the dataset
    filtered_data = data[~((data['word'].isin(most_frequent_words_O)) & (data['tag'] == 'O'))]

    return filtered_data

def main():
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')
    test_data = pd.read_csv('data/test.csv')

    # Load data.csv to identify the most frequent 400 words with tag "O"
    data_data = pd.read_csv('data/data.csv')
    most_frequent_words_O = data_data.loc[data_data['tag'] == 'O', 'Word'].str.lower().value_counts().nlargest(400).index
    create_directory_if_not_exists('data/400_frequent_removed')
    # Remove the most frequent 400 words with the tag "O" from train, val, and test datasets
    filtered_train_data = remove_most_frequent_words_with_tag_O(train_data, most_frequent_words_O)
    filtered_val_data = remove_most_frequent_words_with_tag_O(val_data, most_frequent_words_O)
    filtered_test_data = remove_most_frequent_words_with_tag_O(test_data, most_frequent_words_O)
    # Save the filtered datasets to CSV files
    filtered_train_data.to_csv('data/400_frequent_removed/filtered_train.csv', index=False)
    filtered_val_data.to_csv('data/400_frequent_removed/filtered_val.csv', index=False)
    filtered_test_data.to_csv('data/400_frequent_removed/filtered_test.csv', index=False)

if __name__ == "__main__":
    main()
