# Install

Necessary modules might be installed with pip 

```
pip install -r requirements.txt
```

# Setup

There is a pretrained model available. It can be downloaded as:
```
./download_models.sh
```

# Testing

To test the trained model it is necessary to provide a csv with the same features x001 to x304.
The output will be a new csv in a file named **predicted.csv** with a new column y at the end.
```
python test.py -i $dataset 
```

# Training

In order to train the model on a new dataset
```
python train.py -i $dataset
```
Some configuration as the number of principal components can be changed.


