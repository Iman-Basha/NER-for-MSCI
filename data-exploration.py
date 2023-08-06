import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_tag_distribution(data, dataset_name, output_file):
    # Calculate tag frequency
    tag_counts = data['tag'].value_counts()

    # Calculate percentage of each tag
    tag_percentages = (tag_counts / len(data)) * 100

    # Create a summary DataFrame
    tag_summary = pd.DataFrame({
        'Tag': tag_counts.index,
        'Frequency': tag_counts.values,
        'Percentage': tag_percentages.values
    })

    # Sort the DataFrame by tag frequency
    tag_summary = tag_summary.sort_values(by='Frequency', ascending=False)

    # Save tag distribution summary to a CSV file
    tag_summary.to_csv(output_file, index=False)

    # Plot tag distribution
    plt.figure(figsize=(8, 6))
    plt.bar(tag_summary['Tag'], tag_summary['Frequency'])
    plt.xlabel('Tag')
    plt.ylabel('Frequency')
    plt.title(f'Tag Distribution - {dataset_name} Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'data/400_frequent_removed/exploration/tag_distribution_{dataset_name}.png')

def analyze_sentence_length_distribution(data, dataset_name, output_file):
    sentence_lengths = data.groupby('sent_id')['word'].count()
    max_sentence_length = sentence_lengths.max()
    print(f"Maximum Sentence Length in {dataset_name} Dataset: {max_sentence_length}")
    sentence_lengths.to_csv(output_file, index=True, header=['Sentence Length'])
    # Plot sentence length distribution
    plt.figure(figsize=(8, 6))
    plt.hist(sentence_lengths, bins=30, edgecolor='black')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title(f'Sentence Length Distribution - {dataset_name} Dataset')
    plt.tight_layout()
    plt.savefig(f'data/400_frequent_removed/exploration/sentence_length_distribution_{dataset_name}.png')

def analyze_tag_distribution_per_sentence(data, dataset_name, output_file):
    # Calculate tag distribution per sentence
    tag_distribution_per_sentence = data.groupby('sent_id')['tag'].apply(lambda x: dict(Counter(x)))

    # Save tag distribution per sentence to a CSV file
    tag_distribution_per_sentence.to_csv(output_file, index=True, header=True)

def analyze_word_frequency(data, dataset_name, output_file):
    # Calculate word frequency
    word_counts = data['word'].value_counts()

    # Save word frequency to a CSV file
    word_counts.to_csv(output_file, index=True, header=True)


def main():
    train_data = pd.read_csv('data/400_frequent_removed/exploration/train_filtered.csv')
    val_data = pd.read_csv('data/400_frequent_removed/exploration/val_filtered.csv')
    test_data = pd.read_csv('data/400_frequent_removed/exploration/test_filtered.csv')

    create_directory_if_not_exists('data/400_frequent_removed/exploration/exploration')

    # Analyze tag distribution
    analyze_tag_distribution(train_data, 'Train', 'data/400_frequent_removed/exploration/exploration/tag_distribution_train.csv')
    analyze_tag_distribution(val_data, 'Validation', 'data/400_frequent_removed/exploration/exploration/tag_distribution_val.csv')
    analyze_tag_distribution(test_data, 'Test', 'data/400_frequent_removed/exploration/exploration/tag_distribution_test.csv')

    # Analyze sentence length distribution
    analyze_sentence_length_distribution(train_data, 'Train', 'data/400_frequent_removed/exploration/exploration/sentence_length_distribution_train.csv')
    analyze_sentence_length_distribution(val_data, 'Validation', 'data/400_frequent_removed/exploration/exploration/sentence_length_distribution_val.csv')
    analyze_sentence_length_distribution(test_data, 'Test', 'data/400_frequent_removed/exploration/exploration/sentence_length_distribution_test.csv')

    # Analyze tag distribution per sentence
    analyze_tag_distribution_per_sentence(train_data, 'Train', 'data/400_frequent_removed/exploration/exploration/tag_distribution_per_sentence_train.csv')
    analyze_tag_distribution_per_sentence(val_data, 'Validation', 'data/400_frequent_removed/exploration/exploration/tag_distribution_per_sentence_val.csv')
    analyze_tag_distribution_per_sentence(test_data, 'Test', 'data/400_frequent_removed/exploration/exploration/tag_distribution_per_sentence_test.csv')

    # Analyze word frequency
    analyze_word_frequency(train_data, 'Train', 'data/400_frequent_removed/exploration/exploration/word_frequency_train.csv')
    analyze_word_frequency(val_data, 'Validation', 'data/400_frequent_removed/exploration/exploration/word_frequency_val.csv')
    analyze_word_frequency(test_data, 'Test', 'data/400_frequent_removed/exploration/exploration/word_frequency_test.csv')

if __name__ == "__main__":
    main()

