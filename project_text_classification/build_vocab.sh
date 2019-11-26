#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat Datasets/twitter-datasets/train_pos_full.txt Datasets/twitter-datasets/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
