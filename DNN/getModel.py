
def get_model(modelname):
	return get_model(modelname,14)

def get_model(modelname,dim):
	import keras
	from keras.models import Sequential
	from keras.constraints import maxnorm
	from keras.regularizers import l1l2, l1, l2, activity_l1, activity_l2, activity_l1l2
	from keras.layers.core import MaxoutDense, Dense, Activation, Dropout, Highway
	from keras.layers.normalization import BatchNormalization
	from keras.layers.advanced_activations import PReLU, ELU, SReLU
	
	if modelname == "simplest":
		model = keras.models.Sequential()
		model.add(Dense(8, input_dim=dim, init='uniform', activation='relu'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "tuto":
		model = keras.models.Sequential()
		model.add(Dense(12, input_dim=dim, init='uniform', activation='relu'))
		model.add(Dense(8, init='uniform', activation='relu'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "tuto+":
		model = keras.models.Sequential()
		model.add(Dense(21, input_dim=dim, init='uniform', activation='relu'))
		model.add(Dense(14, init='uniform', activation='relu'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))	
	elif modelname == "tuto++":
		model = keras.models.Sequential()
		model.add(Dense(21, input_dim=dim, init='uniform', activation='relu'))
		model.add(Dense(14, init='uniform', activation='relu'))
		model.add(Dense(7, init='uniform', activation='relu'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "tuto++norm":
		model = keras.models.Sequential()
		model.add(Dense(21, input_dim=dim, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		#~ model.add(Dropout(0.2))
		model.add(Dense(14, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(7, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "tuto++normdropout":
		model = keras.models.Sequential()
		model.add(Dense(21, input_dim=dim, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(14, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(7, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "3hl" :
		model = keras.models.Sequential()
		model.add(Dense(64, input_dim=dim, init='uniform', activation='relu'))
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "3hl_norm" :
		model = keras.models.Sequential()
		model.add(Dense(64, input_dim=dim, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "3hl_norm_dropout_visible" :
		model = keras.models.Sequential()
		model.add(Dropout(0.2, input_shape=(dim,)))
		model.add(Dense(64, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "3hl_norm_dropout_hidden" :
		model = keras.models.Sequential()
		model.add(Dense(64, input_dim=dim, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == '4hl' :
		model = keras.models.Sequential()
		model.add(Dense(128, input_dim=dim, init='uniform', activation='relu'))
		model.add(Dense(64, init='uniform', activation='relu'))
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(Dense(1, init='uniform', activation='sigmoid'))		
	elif modelname == '4hl_norm' :
		model = keras.models.Sequential()
		model.add(Dense(128, input_dim=dim, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(64, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == '4hl_norm_dropout_hidden' :
		model = keras.models.Sequential()
		model.add(Dense(128, input_dim=dim, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(64, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == '4hl_norm_dropout_visible' :
		model = keras.models.Sequential()
		model.add(Dropout(0.2, input_shape=(dim,)))
		model.add(Dense(128, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(64, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(32, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(16, init='uniform', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(1, init='uniform', activation='sigmoid'))
	elif modelname == "4hl_inspiredMarie":
		model = keras.models.Sequential()					# From Marie's work:  first layer has nb_classes*2**4 neurons for 5 hidden layers
		model.add(Dense(2*2**3, input_dim=dim, init='uniform', activation='sigmoid', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(2*2**2, init='uniform',activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(2*2, init='uniform', activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(Dense(1, init='uniform', activation='sigmoid'))
		
		
	return model
