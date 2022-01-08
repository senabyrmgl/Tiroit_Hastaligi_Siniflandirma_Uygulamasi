#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset_loader import DatasetLoader
from data_preprocessor import DataPreprocessor
from model_builder import ModelBuilder

      
#verisetini yüklüyoruz
data_loader = DatasetLoader('../resources/ann-train.data')
train = data_loader.load()

data_loader = DatasetLoader('../resources/ann-test.data')
test = data_loader.load()

#ön işleme kütüphanemizi dahil ediyoruz
dp = DataPreprocessor()
train, test = dp.preprocess(train, test) 

#verileri test ve train olarak ayırıyoruz
train_X, train_y = dp.split_predictors(train)
test_X, test_y = dp.split_predictors(test)


#validation kümemizi de eğitim ve test için ayırıyoruz
X_train, X_val, y_train, y_val = dp.validation_split(train_X, train_y)

#verileri ölçekliyoruz
X_train, X_val = dp.scale_data(X_train, X_val)

# model kütüphanemizden modeli çekip classifier modelini çekiyoruz
mb = ModelBuilder()
classifier = mb.get_classifier()        
       
#daha kucuk bir kumede cross validation yapıyoruz
#mb.validate(classifier, X_train, y_train)
##Cross Validation - Accuracy : 98.11% (1.13%)

#validation kumesi ile tahmin ettiriyoruz
#mb.evaluate(classifier, X_train, y_train, X_val, y_val)
##Accuracy is 99.073% 

#test verisini ölçeklendiriyoruz
train_X, test_X = dp.scale_data(train_X, test_X)

#cross-validation tüm train verisinde çapraz korelasyon yapıyoruz
#mb.validate(classifier, train_X, train_y)
##Cross Validation - Accuracy : 98.57% (0.41%)

#tüm train verisini eğitiyoruz
classifier.fit(train_X, train_y, batch_size = 10, epochs = 100)

#test verisini tahmin ediyoruz
mb.check_prediction(classifier, test_X, test_y)
#Test verisi - Accuracy is 98.279% 

#Modeli diske kaydediyoruz
mb.save_model(classifier, '../model/final_model1')

#Tahminleri kayıt ediyoruz
mb.save_predictions(classifier, test_X)     