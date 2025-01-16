# Week 4 Challenge - Full Segmentation with a Dictionary

## Project Overview
This project demonstrates how to perform full segmentation of a given sentence using a provided dictionary. The goal is to generate all possible ways to segment the sentence based on the words available in the dictionary.

## Key Components
- **Dictionary (`Dict`)**: A dictionary where each key is a word and the value is its frequency (though the frequency is not used in the segmentation process).
- **Sentence (`sentence`)**: The input sentence to be segmented.
- **all_cut Function**: A function that performs full segmentation using a depth-first search (DFS) approach.
- **dfs Function**: A helper function that recursively explores all possible segmentations of the remaining part of the sentence.
- **main Function**: The main function that reads the dictionary and sentence, calls the `all_cut` function, and prints the results.
