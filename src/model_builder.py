#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy
import pandas

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

class ModelBuilder:
    """Class ModelBuilder. Creates, validate, evaluate and saves the model.
    
    Methods:
        classifier_model(): defines the neural network model to be used in the scikit-learn wrapper.
        get_classifier(): gets the scikit-learn wrapped keras classifier model.
        finalize_and_save(): fits the final model and save to disk.
        save_model(): saves the model to disk.
        load_model(): loads and returns model from disk.
        check_prediction(): checks prediction accuracy.
        save_predictions(): save predictions to disk.        
    
    """
    
    def classifier_model(self):
        """Bu kod ile Neural agimin katmanlarini ve ozniteliklerini belirleyerek modelimi olusturuyorum.
        Genelde cogu ANN'de kullanilan Kerasin Sequential() methodu ile modelimi kuruyorum. Modelime 48 dense ekliyorum ve girdi boyutunu 21 olarak ayarliyorum. Bunun sebebi verilerimizin 21 kategoriden olusmasi. Her kategorik veriyi bir girdi neuronuna bagliyorum. Aktivasyon fonksiyonu olarak ise Relu aktivasyon fonksiyonunu seciyorum.
        Her katmandan sonra 0.25 lik bir Dropout ekliyorum. Dropout eklememin sebebi modelimin veri setini ezberlemesinden kacindirmak yani her forward propagation asamasinda %0.25 lik veriyi unutuyorum bu sayede validation gercek dunya verilerine daha guzel fit edebilicek.
        iki 48 lik ara hidden layer daha ekledikten  sonra cikti katmanimi 3 layer olarak belirliyorum ve Aktivasyon fonksiyonunu softmax seciyorum. Softmax olasiliksal dagilim acisindan output katmaninda tam ihtiyacimiz olan araliga sahip ve olceklememize musait olan aktivasyon fonksiyonu oldugu icin bunu sectim.
        En son durumda modelimizi return ediyoruz

        """
        #Sequential model
        model = Sequential()
        
        #Input layer and first hidden layer.
        model.add(Dense(48, kernel_initializer = 'uniform', input_dim=21, activation='relu'))
        
        #25% of neurons are droppedout to avoid over learning/fitting.
        model.add(Dropout(0.25))
        
        #2nd hidden layer
        model.add(Dense(48, kernel_initializer = 'uniform', activation='relu'))
        
        #25% of neurons are droppedout to avoid over learning/fitting.
        model.add(Dropout(0.25))
        
        #3rd hidden layer
        model.add(Dense(48, kernel_initializer = 'uniform', activation='relu'))
        
        #25% of neurons are droppedout to avoid over learning/fitting.
        model.add(Dropout(0.25))
        
        #Output layer
        model.add(Dense(3, kernel_initializer = 'uniform', activation='softmax'))
        
    	#Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    
    def get_classifier(self):
        """Bu fonksiyonumuzda olusturdugumuz modelin batch yani
         yigin veri boyutu parametresini ve egitim epoch sayisini ayarliyoruz. 
         Default olarak 10 batch ve 100 epochun boyle sayisal bir veri icin 
         yeterli olucagini dusunuyorum.
        """
        #Creates model
        classifier = KerasClassifier(build_fn = self.classifier_model, batch_size = 10, epochs = 100)
        
        return classifier
    
    def finalize_and_save(self, model, train_X, train_y, filename='../model/final_model'):
        """Model egitimini gerceklestiren ve modelin sonucunu kayit eden fonksiyondur.
        """
        #Fits the model to train data
        model.fit(train_X, train_y)
        
        #Saves the model to disk
        self.save_model(model, filename)
        
    def save_model(self, model, filename='../model/saved_model'):
        """
        Finalize_and_save fonksiyonu tarafindan cagirilir. 
        Model verilerini pickle kutuphanesi ile dosya olarak kayit eder.
        """
        #Save the model to disk
        pickle.dump(model, open(filename, 'wb' ))
        print("\nModel is saved..\n")
    
    def load_model(self, model_filename):
        """Pickle formatinda kayit edilen verileri 
        tekrar cagirip pickle.load modulu ile modelleri dahil etmemizi saglar. 
        """
        #Load the model from disk
        loaded_model = pickle.load(open(model_filename, 'rb' ))
        
        return loaded_model
    
    def validate(self, model, train_X, train_y):
        """
        Model,train_x,train_y girdilerini alir. Cross-validation methodu ile accuracy skorunu hesaplar ve print eder.
        """
        results = cross_val_score(estimator = model, X = train_X, y = train_y, cv = 10, n_jobs = 3)
        
        print("\nCross Validation - Accuracy : %.2f%% (%.2f%%)\n" % (results.mean()*100.0, results.std()*100.0))
        
    def evaluate(self, model, train_X, train_y, test_X, test_y):
        """Takes arguments: 'model', 'train_X', 'train_y', 'test_X', 'test_y'.
        model: model to be evaluated.
        train_X: input part of train data.
        train_y: output part of train data.
        test_X: input part of test data - validation.
        test_y: output part of test data - validation.
        
        Perfoms evaluation on the specified model and prints accuracy.
        """
        #Fits the model to traindata
        model.fit(train_X, train_y, batch_size = 10, epochs = 100) 
        
        #prediction
        y_test_pred = model.predict(test_X)
        
        #Confustion matrix from predictions
        cm = confusion_matrix(test_y, y_test_pred)
        
        print("\nModel Evaluation - Accuracy is %.3f%% \n" % ((cm[0][0]+cm[1][1]+cm[2][2])*100/test_y.size))
        
    def check_prediction(self, model, test_X, test_y):
        """
        Model, test_x ve test_y input degerlerini alir. 
        Modelin prediction skorunu hesaplar.
         Test verisinin accuracy degerini hesaplar. 
         Bu dis dunya verilerine karsi saglayacagi basari olarak gorulebilir.
        """
        #Prediction
        y_test_pred = model.predict(test_X)
        
        #Confustion matrix from predictions
        cm = confusion_matrix(test_y, y_test_pred)
        
        #Map predictions to class name
        y_test_pred = self.map_pred_class(y_test_pred)
        
        print("\n............Predictions............\n")
        print(y_test_pred.reshape(-1,1))
        print("\nTest Data - Accuracy is %.3f%% \n" % ((cm[0][0]+cm[1][1]+cm[2][2])*100/test_y.size))
    
    def save_predictions(self, model, test_X):
        """
        Tahmin sonuclarini alip csv formatinda prediction klasoru altina tablo olarak kayit eder.
        """
        #Prediction
        predictions = model.predict(test_X)

        #Map predictions to class name
        predictions = self.map_pred_class(predictions)
        
        #Saves to disk
        pandas.DataFrame(predictions).to_csv('../prediction/predictions.csv', index=False)
        print("\nPredictions are saved..\n")
    
    def map_pred_class(self, preditions):
        """Predictions girdisini alir ve tahmin degerlerine gore Tiroit durumu normal mi , subnormal mi yoksa Hipertroid mi bunu belirler ve bastirir.
        """
        
        pred_map = ['Normal'  if(x==3) else 'Subnormal' if (x==2) else 'HyperThyroid'  for x in preditions]
        
        return numpy.array(pred_map)