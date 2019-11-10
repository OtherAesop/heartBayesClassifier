# heartBayesClassifier, a naive bayesian classifier
# Copyright (C) 2019  Ian Gore
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import utility as helper
import naive_bayes as bae
from time import time


def test_network(training_file, testing_file):  # function for neatness
    print(f"Beginning training and testing of {training_file} and {testing_file}...")
    start = time()
    bae.naive_bayes(training_file, testing_file)
    end = time()
    test_time = end - start
    print(f'...ending training and testing of {training_file} and {testing_file}, process completed'
          f' in {helper.translate_seconds(test_time)} (HH:MM:SS).\n')
    return test_time


total_time = test_network("datasets/spect-orig.train.csv", "datasets/spect-orig.test.csv")
# total_time += test_network("datasets/satellite_training.txt", "UCI_datasets/satellite_test.txt")
# total_time += test_network("datasets/yeast_training.txt", "UCI_datasets/yeast_test.txt")

print(f'All tests completed in {helper.translate_seconds(total_time)}.\n')
