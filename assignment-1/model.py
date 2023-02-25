class Model:
    def __init__(self,predictors, label, input_len, total_words,X_val,Y_val,path):
        self.optim ='adam'
        self.loss ='categorical_crossentropy'
        mdl = Sequential()
        mdl.add(Embedding(total_words, 12, input_length=input_len))
        mdl.add(LSTM(132))
        mdl.add(Dropout(0.35))
        mdl.add(Dense(total_words, activation='softmax'))
        mdl.compile(loss=self.loss, optimizer=self.optim,metrics=['accuracy'])
        stop = EarlyStopping(monitor = 'val_loss', patience = 3)
        save = ModelCheckpoint(path, monitor = 'val_loss', save_best_only = True)
        call_back_pointers = [stop, save]
        print(mdl.summary())
        final=mdl.fit(predictors, label, validation_data=(X_val,Y_val),epochs=80, batch_size=128,callbacks = call_back_pointers)
        return final
