from utility import make_datasets, map_classes, summarize, predict_classes, print_test_summary, print_training_summary
from numpy import asarray


def naive_bayes(training_file, test_file):
    # Training Step
    dataset, labelset = make_datasets(training_file)  # Do input step
    data_summary = map_classes(dataset, labelset)  # Organize input into reasonable format
    class_summary = summarize(data_summary)  # Do calculations to summarize input into usable forms
    print_training_summary(class_summary)

    # Test Step
    test_dataset, test_labelset = make_datasets(test_file)  # Do input step for test file
    test_dataset = asarray(test_dataset).astype(float)  # cast to float, this step is handled implicity in training
    test_labelset = asarray(test_labelset).astype(float)  # but not in testing

    prediction_set = predict_classes(class_summary, test_dataset)  # Make predictions about test dataset, save dups
    print_test_summary(prediction_set, test_labelset)

    return data_summary

