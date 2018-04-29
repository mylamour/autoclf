class CNN():
    def __init__(self,kernel_initializer='he_normal',optimizer='adam',activation='relu',loss='binary_crossentropy',dropout=0.5):
        
        self.name = '_CNNTrain'

        self.kernel_initializer = kernel_initializer
        self.optimizer = optimizer
        self.activation = activation
        self.dropout = dropout
        self.loss = loss
        self.model = None

    
    def model(self):
        inputs = Input(shape=(sequence_length,), dtype='int32')
        embedding = Embedding(input_dim=vocabulary_size,
                            output_dim=embedding_dim, input_length=sequence_length)(inputs)
        reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(
            filter_sizes[0], embedding_dim), padding='valid', kernel_initializer=self.kernel_initializer, activation=self.activation)(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(
            filter_sizes[1], embedding_dim), padding='valid', kernel_initializer=self.kernel_initializer, activation=self.activation)(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(
            filter_sizes[2], embedding_dim), padding='valid', kernel_initializer=self.kernel_initializer, activation=self.activation)(reshape)

        maxpool_0 = MaxPool2D(pool_size=(
            sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(
            sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(
            sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)(
            [maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(dropout)(flatten)
        output = Dense(units=2, activation='softmax')(dropout)

        # this creates a model that includes
        model = Model(inputs=inputs, outputs=output)

        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy'])

        return model

    def creat_model(self):
        if not os.path.isdir(self.name):
            os.mkdir(self.name)
        checkpoint = ModelCheckpoint('{}/weights.{epoch:03d}-{val_acc:.4f}.hdf5'.format(self.name), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

        self.create_model = self.model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

    def search_model(self):
        pass
    
    def fit(self,x_train,y_train,x_test,y_test):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_test, y_test))
        self.model.save("{}/model.h5".format(self.name))
        