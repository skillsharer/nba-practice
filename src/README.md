# Train

My training iteration contains the following steps:

1. Split the dataset into train and test set. This is usually 20-30% for the test set and 80-70% for the training set.
2. Drop leakage features which we don't know at prediction time.
In my case this would be the duration time column. At prediction time, we do not know the contact duration during the current campaing. If prediction happens before the current campaign where we also do not know the contact type, day and month, those features would be leaky too. In my case, at prediction time, we know them.
3. Separate features based on type (categorical and numerical).