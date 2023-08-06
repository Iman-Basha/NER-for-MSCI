# NER-for-MSCI
This repository is a documentation of the application of two sequence labeling models, Bidirectional Long Short-Term Memory (BiLSTM) and Conditional Random Fields (CRF), for the task of extracting disease entities from 30,000 clinical documents(Nayak, 2019; Innoplexus, 2019). The dataset, comprising annotated text with tokenized words and labels following the Inside-Outside-Beginning (IOB) tagging format, is analyzed, and the imbalanced class distribution is addressed in model evaluation. The BiLSTM model is constructed with two bidirectional LSTM layers, embedding, dropout, and time-distributed layers, while the CRF model is optimized using the 'lbfgs' algorithm. 
The experiments involve comprehensive hyperparameter tuning, and both models are evaluated using the F1-score, precision, and recall. The results reveal a distinct advantage for the CRF model, achieving an F1 score of 88.75%, significantly outperforming the BiLSTM at 54.66%. The BiLSTM, however, demonstrates consistent performance across datasets, albeit with challenges related to data quality, model complexity, and training.

Note that the model development and implementation were done in google colab and transferred here for accessibility.


References
Innoplexus. (2019). Innoplexus Online Hiring Hackathon: Saving lives with AI. (Analytics Vidhya) Retrieved 2023 from https://datahack.analyticsvidhya.com/contest/innoplexus-online-hiring-hackathon-saving-lives-wi/
Nayak. (2019). Hackathon Disease Extraction: Saving lives with AI. (Kaggle) Retrieved 2023 from https://www.kaggle.com/datasets/rsnayak/hackathon-disease-extraction-saving-lives-with-ai?resource=download
