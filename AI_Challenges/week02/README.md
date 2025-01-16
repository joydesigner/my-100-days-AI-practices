###  Week 3 Challenge:
* Train a multi-class classification model to predict the category of a 5-dimensional input vector based on predefined rules. The output of the model is a probability distribution across multiple categories.
* Use cross entropy to implement a multi-classification task. The maximum number in the 5-dimensional random vector belongs to the category in which dimension.

### Task rules:
1. Input data:
Each input is a 5-dimensional vector x = [x_1, x_2, x_3, x_4, x_5], where x_i is a random value between 0 and 1.
2. Classification rules:
* The category (label) is determined by the index of the maximum value in the vector.
* For example:
x = [0.2, 0.5, 0.1, 0.8, 0.4]
Since the maximum value 0.8 is at index 3, the label of this sample is 3 (category 3).
3. Objective:
* Train the model to correctly predict the class index of the input vector.
* Compare the predicted probabilities with the true class labels using cross entropy loss.
4. Output:
* The model outputs a probability distribution over 5 classes (e.g., [0.1, 0.3, 0.2, 0.4, 0.0]).
* The predicted class is the index with the highest probability.