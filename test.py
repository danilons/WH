# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model

if __name__ == "__main__":

	print("Running predictions")

	df = pd.read_csv('test.csv', index_col=None)
	
	mean = pd.read_csv('mean.csv', index_col=None)
	sc = joblib.load('sc.pkl')
	pca = joblib.load('pca.pkl')
	model = load_model('wh.h5')

	original_df = df.copy()

	for column in df.columns[:304]:
		df[column].fillna(mean[column].values[0], inplace=True)

	X = df.values[:, :304]
	Xt = sc.transform(pca.transform(X))

	predictions = model.predict(Xt)
	data = pd.DataFrame(data=np.hstack((original_df.values[:, :304], predictions)), columns=df.columns)
	data.to_csv('predicted.csv', index=None)
