# coding: utf-8
import argparse
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-i','--input', help='Input file', required=True)
	parser.add_argument('-m','--model_file', help='Model file output', default='wh.h5')
	parser.add_argument('-p','--pca', help='PCA file', default='pca.pkl')
	parser.add_argument('-s','--scaler', help='Scaler file', default='scl.pkl')
	parser.add_argument('-me','--mean_file', help='Mean file output', default='mean.csv')
	parser.add_argument('-o','--output_file', help='Output file output', default='predicted.csv')
	
	args = vars(parser.parse_args())
	
	print("Running predictions")

	df = pd.read_csv(args['input'], index_col=None)
	
	mean = pd.read_csv(args['mean_file'], index_col=None)
	sc = joblib.load(args['scaler'])
	pca = joblib.load(args['pca'])
	model = load_model(args['model_file'])

	original_df = df.copy()

	for column in df.columns[:304]:
		df[column].fillna(mean[column].values[0], inplace=True)

	X = df.values[:, :304]
	Xt = sc.transform(pca.transform(X))

	predictions = model.predict(Xt)
	data = pd.DataFrame(data=np.hstack((original_df.values[:, :304], predictions)), columns=df.columns)
	data.to_csv(args['output_file'], index=None)
