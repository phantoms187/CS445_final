#David Davis CS545 Fall 2020 Program 2

# importing csv module 
import csv 
import random
from math import sqrt
from math import exp
from math import pi
from math import pow
from math import log

# Determine the average for a list of numbers
def average(nums):
	return sum(nums)/len(nums)
 
# Calculate the standard deviation
def standard_deviation(nums):
    avg = average(nums)
    variance = sum([pow(x-avg,2) for x in nums]) / float(len(nums)-1)
    root = sqrt(variance)
    if root == 0:
        return 0.0000001
    return root
#Calculate average, standard deviation
def data_info(data):
	info = [(average(column), standard_deviation(column)) for column in zip(*data)]
	return info

# Calculate the Gaussian probability distribution function for a given value x
def probability(x, average, standard_deviation):
    e = exp(-(pow(x-average, 2) / (2 * pow(standard_deviation, 2) )))
    if (1 / (sqrt(pi * 2) * standard_deviation) * e ) <= 0:
        return 0.0000001
    return (1 / (sqrt(pi * 2) * standard_deviation)) * e

email_data = [] 
non_spam_emails = []
spam_emails = []
file_email_data = "spambase.data"

#Open file and load email data
with open(file_email_data, 'r') as email_reader: 
    csvreader = csv.reader(email_reader)  
    for row in csvreader: 
        row = [float(item) for item in row]
        email_data.append(row)
#Separate emails into spam and not spam for training
for row in email_data:
    if row[-1] == 0:
        non_spam_emails.append(row)
    else:
        spam_emails.append(row)

#Shuffle emails to get new results each time
random.shuffle(non_spam_emails)
random.shuffle(spam_emails)

#Determine the testing and training sizes and split half of each into those groups 
half_of_spam_emails = int(len(spam_emails)/2)
half_of_non_spam_emails = int(len(non_spam_emails)/2)
testing_spam_emails = spam_emails[0:half_of_spam_emails]
training_spam_emails = spam_emails[half_of_spam_emails:]
testing_non_spam_emails = non_spam_emails[0:half_of_non_spam_emails]
training_non_spam_emails = non_spam_emails[half_of_non_spam_emails:]
number_of_training_emails = len(training_spam_emails) + len(training_non_spam_emails)
number_of_testing_emails = len(testing_spam_emails) + len(testing_non_spam_emails)

#Calculate probability of spam and non-spam as a whole for the training data, use log to compensate for small values
spam_class_probability = log(len(training_spam_emails)/ float(number_of_training_emails))
non_spam_class_probability = log(len(training_non_spam_emails)/ float(number_of_training_emails))
#Group all testing emails for testing
testing_emails = testing_non_spam_emails + testing_spam_emails
#Calculate the average and standard deviation for each of the classifiers based on spam or non-spam
spam_training_info = data_info(training_spam_emails)
non_spam_training_info = data_info(training_non_spam_emails)

#Tracking for the confusion matrix calculations
number_true_spam = 0
number_true_non_spam = 0
number_false_spam = 0
number_false_non_spam = 0

#For each testing example add of the probabilities and determine which has a greater probability.
#Check each decision against the actual outcome and sort accordingly  
for row in testing_emails:
    for i in range(len(testing_emails[0])):
        average, standard_deviation = spam_training_info[i]
        spam_class_probability += log(probability(row[i], average, standard_deviation))
        average, standard_deviation = non_spam_training_info[i]
        non_spam_class_probability += log(probability(row[i], average, standard_deviation))
    if spam_class_probability > non_spam_class_probability:
        if row[-1] == 1:
            number_true_spam +=1
        else:
            number_false_spam +=1
    else:
        if row[-1] == 0:
            number_true_non_spam +=1
        else:
            number_false_non_spam +=1
    spam_class_probability = log(len(training_spam_emails)/ float(number_of_training_emails))
    non_spam_class_probability = log(len(training_non_spam_emails)/ float(number_of_training_emails))

#Different calculations for the confusion matrix
precision = number_true_spam / (number_true_spam + number_false_spam)
negative_predictive_value = number_true_non_spam / (number_true_non_spam + number_false_non_spam)
sensitivity = number_true_spam / (number_true_spam + number_false_non_spam)
specificity = number_true_non_spam / (number_true_non_spam + number_false_spam)
accuracy = (number_true_spam + number_true_non_spam) / (number_true_spam + number_true_non_spam + number_false_spam + number_false_non_spam)

print("number_true_spam")
print(number_true_spam)
print("number_true_non_spam")
print(number_true_non_spam)
print("number_false_spam")
print(number_false_spam)
print("number_false_non_spam")
print(number_false_non_spam)

print("precision")
print(round(precision*100, 2))
print("negative_predictive_value")
print(round(negative_predictive_value*100, 2))
print("sensitivity")
print(round(sensitivity*100, 2))
print("specificity")
print(round(specificity*100, 2))
print("accuracy")
print(round(accuracy*100, 2))
