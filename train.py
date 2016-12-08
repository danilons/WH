# coding: utf-8
import argparse
import pandas as pd
import numpy as np

from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.visualize_util import plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

seed = 42
np.random.seed(seed)
	
def build_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=n_components))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def preprocess(df):
	sc = StandardScaler()
	pca = PCA(n_components=n_components)

	df_train, df_test = train_test_split(df.copy(), random_state=42, test_size=0.2)
	
	# compute mean
	mean = df_train.mean(axis=0)
	mean.drop('y', axis=0, inplace=True)

	for column in df_train.columns[:304]:
		df_train[column].fillna(mean[column], inplace=True)

	dataset = df_train.values
	X_train = dataset[:, :304]
	y_train = dataset[:, 304].tolist()

	pca.fit(X_train)
	sc.fit(pca.transform(X_train))

	for column in df_test.columns[:304]:
		df_test[column].fillna(mean[column], inplace=True)

	dataset_test = df_test.values
	X_test = dataset_test[:, :304]
	y_test = dataset_test[:, 304].tolist()

	return sc.transform(pca.transform(X_train)), y_train, \
	sc.transform(pca.transform(X_test)), y_test, sc, pca, mean


def train_and_eval(df):
	X_train, y_train, X_test, y_test, sc, pca, mean = preprocess(df)

	print("Start training.")

	model = build_model()
	try:
		summary = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
	except (KeyboardInterrupt, SystemExit):
		print("Training stopped.")

	ypred = np.squeeze(model.predict(X_test))

	rmse = mean_squared_error(y_test, ypred)**0.5
	print(u"\nRoot mean squared error {:.4f}".format(rmse))
	print(u"R2 score {:.4f}".format(r2_score(y_test, ypred)))

	# accuracy
	error = abs(y_test - ypred)
	error_threshold = 3  

	print("Accuracy {:.2f}%.".format(float(len(np.where(error <= error_threshold)[0])) / float(len(y_test)) * 100))
	print("True positives number: {}".format(len(np.where(error <= 3)[0])))
	return sc, pca, mean, model

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-i','--input', help='Input file', required=True)
	parser.add_argument('-e','--epochs', help='Number of epochs', type=int, default=200)
	parser.add_argument('-c','--components', help='Number of components', type=int, default=200)
	parser.add_argument('-b','--batch_size', help='Batch size', type=int, default=128)
	parser.add_argument('-m','--model_file_output', help='Model file output', default='wh.h5')
	parser.add_argument('-p','--pca', help='PCA file', default='pca.pkl')
	parser.add_argument('-s','--scaler', help='Scaler file', default='scl.pkl')
	parser.add_argument('-me','--mean_file', help='Mean file output', default='mean.csv')
	parser.add_argument('-g','--net_graph', help='Net file output', default='wh.png')

	args = vars(parser.parse_args())
	
	n_components = args['components']
	nb_epoch = args['epochs']
	batch_size = args['batch_size']


	print("Components {}".format(n_components))
	print("Epochs {}".format(nb_epoch))
	print("------------------------------------------------------")
	
	df = pd.read_csv(args['input'], index_col=None)
	sc, pca, mean, model = train_and_eval(df)

	plot(model, to_file=args['net_graph'])
	pd.DataFrame(mean).T.to_csv(args['mean_file'], index=None)
	joblib.dump(sc, args['scaler']) 
	joblib.dump(pca, args['pca'])
	model.save(args['model_file_output'])
