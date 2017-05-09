

def get_model(args,dim=14):
	
	'''
	args is a dictionary that contains the following entries:
		- architecture : a list that contains the number of neurons per layer (nr of layers = len(list))
		- dropout : a float, the fraction of dropout. 0.0 = no dropout
		- batchnorm : boolean, whether or not to use BatchNorm
		- activation : activation function to use on hidden layers
		- acti_out : activation function on output
		- init : initialisation of layers
		- loss : a string, name of a loss function
		- optimizer : either string or Keras optimizer object
		- metrics: a list of string, metrics to evaluate
		
		No L1/L2 regularization (yet)
	'''

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout
	from keras.layers.normalization import BatchNormalization
	from keras import regularizers
	from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
	
	
	model = Sequential()
	model.add(Dense(args['architecture'][0],
					units=dim,
					kernel_initializer=args['init'],
					activation=args['activation']))
	for i in args['architecture'][1:-1] :
		model.add(Dropout(args['dropout']))
		if args['batchnorm']:
			model.add(BatchNormalization())
		model.add(Dense(i,kernel_initializer=args['init'],activation=args['activation']))
	model.add(Dropout(args['dropout']))
	if args['batchnorm']:
		model.add(BatchNormalization())
	model.add(Dense(args['architecture'][-1],kernel_initializer=args['init'],activation=args['acti_out']))
	
	model.compile(loss=args['loss'], optimizer=args['optimizer'], metrics=args['metric'])
		
	return model
