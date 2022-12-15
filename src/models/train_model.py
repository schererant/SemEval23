# This file is used as inspiration
# Also assignment 2 could be very similar
# https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Make sure the transformers is recent enough (not sure which version is required), 
# otherwise we get problems later on this file
print(transformers.__version__) 

# Setting globals
# In it's final state we should be able to substitute the string below with 
# almost *any* huggingface model name and the file would still work
# Take care to use models pretrained on text classification however
model_checkpoint = "distilbert-base-uncased"
batch_size = 16


data = ...
# as for data the standard format is as a believe [text & label]
# which are packed in dict objects for train, val and test sets
# which in turn are all packed in a DataSetDict

# Also we have 3 features [conclusion, stance, premise], we have to think about how to feed it to the model
# Of course we can concat all these strings into one and then it should be compatible
# But maybe there is a way to explicitly feed 3 different strings at once? 
# Maybe that would improve performance? idk
# If we decide to truncate our data, we have to think about which strings will be truncated, 
# I imagine we shouldn't truncate the stance for example as I imagine it to be pretty crucial for sentiment analysis

num_labels = 20
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

...

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

...