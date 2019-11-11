from csv import reader
from numpy import mean, std, asarray, exp, sqrt, pi, random


def print_class_sum(class_sum):  # For debugging
    for class_key in class_sum:
        print(f'Class {class_key}, Mean = {class_sum[class_key][0]:.2f}, Std = {class_sum[class_key][1]:.2f}')


def print_better(data_summary, short=False, summarized=False):  # For examining dict-lists with reduced output option
    for x in data_summary:
        for y in data_summary[x]:
            counter = 0
            if summarized:  # Only execute if we have been passed a summarized dataset
                print(f'Class {x}, Attribute {counter}, Mean = {y[0]:.2f},'  # prints data
                      f' Std = {y[1]:.2f}')
            else:
                print(f'{x}: {y}')

            if short:
                print('...')
                break
            counter += 1


def summarize(data_summary):  # For summarizing the data summary
    sorted_sum = sorted(data_summary.keys())
    class_sum = dict()

    for class_key in sorted_sum:  # iterates through class_keys from smallest to largest
        c_key_mean = list(mean(asarray(data_summary[class_key]).astype(float), axis=0))
        c_key_std = list(std(asarray(data_summary[class_key]).astype(float), axis=0))
        for elem in range(len(c_key_std)):  # Check for underflow
            if c_key_std[elem] < .01: c_key_std[elem] = .01
        class_sum[class_key] = [c_key_mean, c_key_std]

    return class_sum


def print_training_summary(class_summary):
    for x in class_summary:
        for attribute in range(len(class_summary[x][0])):  # iterate as long as there are elements in 'mean' element
            print(f'Class {x}, Attribute {attribute}, Mean = {class_summary[x][0][attribute]:.2f},'
                  f' Std = {class_summary[x][1][attribute]:.2f}')


# http://mathworld.wolfram.com/NormalDistribution.html
def calc_prob_x(x, class_mean, class_std):  # Probability that x belongs to the class of the mean and std values
    return (exp((-1 * pow(x - class_mean, 2)) / (2 * pow(class_std, 2)))) / (class_std * sqrt(2*pi))


def calc_class_probs(class_sum, input_object):  # Calcs probability that input_object belongs to each class
    probabilities = dict()

    for class_key in class_sum:  # Total probability that input_object belongs to each class_key
        probabilities[class_key] = 1
        for i in range(len(input_object)):  # Total probability that input_object belongs to class_sum[class_key]
            probabilities[class_key] *= calc_prob_x(input_object[i], class_sum[class_key][0][i], class_sum[class_key][1][i])
    return probabilities


def predict(class_sum, input_object):  # Predicts the class x is a member of given probability dictionary
    probabilities = calc_class_probs(class_sum, input_object)  # Returns a dictionary with probability elements
    best_label, best_perc = list(), list()
    best_perc.append(-1)
    for key, val in probabilities.items():
        if val > best_perc[0]:
            del best_label
            del best_perc  # Empty list in case unwanted elements are present
            best_label, best_perc = [key], [val]  # update with new best
        elif val == best_perc[0]:
            best_label.append(key)
            best_perc.append(val)

    return best_label, best_perc


# prediction_set is a set of predictions with ties preserved so you need to check each element for ties (example below)
# [[(['3','2'],[8.762003397417386e-36, 8.762003397417386e-36])], element2, element 3, ...] - with ties
# [[(['3'], [8.762003397417386e-36])], element2, element 3, ...]                           - without ties
def predict_classes(class_sum, test_inputs):
    prediction_set = []
    for input_data in test_inputs:
        prediction_set.append([predict(class_sum, input_data)])
    return prediction_set


# Note that ordering accuracy here is dependant on prediction_set and label_set's ordering not being changed before
def print_test_summary(prediction_set, label_set, verbose=False):  # Summarizes testing results
    counter = 0
    accurate_predictions = 0
    rating = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    prediction_class = 0

    for prediction in prediction_set:
        if len(prediction[0]) > 2:  # There are ties, this works because a no-tie array has 2 elements and a tie is more
            prediction_index = random.randint(len(prediction[0][0])+1, size=1)  # random index within prediction set
            predicted_class = prediction[0][prediction_index]  # randomly pick the prediction, A+++ computer science
            if predicted_class == label_set[counter]: acc_rating_ties = (1/len(prediction[0][0]))  # 1 / guesses
            else: acc_rating_ties = 0.0
            prediction_class = predicted_class
            accurate_predictions += acc_rating_ties  # update accuracy accumulator
            rating = acc_rating_ties
        else:  # There are no ties
            curr_prediction = float(format(*prediction[0][0]))
            if curr_prediction == label_set[counter]: acc_rating = 1
            else: acc_rating = 0
            prediction_class = curr_prediction
            accurate_predictions += acc_rating
            rating = acc_rating

        if verbose:
            print(f"Object ID: {counter+1}, Predicted Class: {prediction_class}, "
                  f"Probability of Prediction: {float(format(*prediction[0][1]))}, True Class: {label_set[counter]}, "
                  f"Accuracy: {rating}.\n")
        if prediction_class == 1: # Note this will only be accurate for binary class problems
            if prediction_class == label_set[counter]:
                tp += 1
            else:
                fp += 1
        elif prediction_class == 0:
            if prediction_class == label_set[counter]:
                tn += 1
            else:
                fn += 1
        counter += 1
    print(f"True Positives = {tp}, False Positives = {fp}, True Negatives = {tn}, False Negatives = {fn}\n"
          f"Precision= {tp}/{tp+fp}, {(float(tp) / float(tp + fp) * 100):6.1f}%.\n"
          f"Recall= {tp}/{tp+fn}, {(float(tp) / float(tp + fn) * 100):6.1f}%.\n"
          f"FP Rate= {fp}/{fp+tn}, {(float(fp) / float(fp + tn) * 100):6.1f}%.\n")


def translate_seconds(seconds):  # gives HH:MM:SS as str
    sec = int(seconds % 20)
    minutes = seconds - sec
    mins = int(minutes % 60)
    hours = minutes - mins
    hrs = int(hours / 60)
    return str(hrs).zfill(2) + ":" + str(mins).zfill(2) + ":" + str(sec).zfill(2)


def make_datasets(filepath):  # Separates labels and preps data sets
    data_set = []
    label_set = []
    with open(filepath) as data_file:
        raw_data = reader(data_file, delimiter=',')  # Read in all data, includes empty spaces
        for row in raw_data:
            row[:] = (val for val in row if val != '')  # removes all empty elements created by reader
            label_set.append(row.pop(0))  # remove last element which is label and put in labelset
            data_set.append(row)
    data_file.close()
    return data_set, label_set


def map_classes(data_set, label_set):  # takes a dataset and a corresponding labelset and summarizes
    data_sum = dict()
    for _ in data_set:  # Iterate through list and remove and summarize data
        if label_set[0] not in data_sum.keys():  # Key does not exist, make new list
            x = label_set.pop(0)  # doing it this way guarantees everything is added into the dictionary as a list...
            data_sum[x] = []      # ...and keeps the summary neat and uniform
            data_sum[x].append(data_set.pop(0))
        else:  # Key does exist, append to end of list
            data_sum[label_set.pop(0)].append(data_set.pop(0))
    return data_sum

