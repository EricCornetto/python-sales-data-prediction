# ===========PROGRAM MACHINE LEARNING Ver.1.0==================================
#  *   Program Pengaruh Biaya Promosi Terhadap Nilai Penjualan.
#  *   Linear Regression Alogrithm
# =============================================================================
#  *   Created By Eric (Ida Bagus Dwi Putra Purnawa).
#  *   Github (https://github.com/EricCornetto).
# =============================================================================
#  *   Start Project Version 1.0 = 16 February 2019.
#  *   End Project Version 1.0 = 16 February 2019.
#  *   Update Project Version = Coming Soon.
#  *   GNU General Public License v3.0.
# =============================================================================
#             Python Machine Learning
# =============================================================================




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt

# Linear Algorithm
# Y = a + bx
class LinearRegression():
    #fitting data
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.n_train = len(self.x_train)
    #find sigma X
    def sigmaX(self):
        result = np.sum(self.x_train)
        return result

    #find sigma Y
    def sigmaY(self):
        result = np.sum(self.y_train)
        return result

    #find sigma XY
    def sigmaXY(self):
        multiply = np.multiply(self.x_train,self.y_train)
        result = np.sum(multiply)
        return result

    #find sigma X^2
    def sigmaSQRX(self):
        sqr = np.square(self.x_train)
        result = np.sum(sqr)
        return result

    #find sigma Y^2
    def sigmaSQRY(self):
        sqr = np.square(self.y_train)
        result = np.sum(sqr)
        return result

    #find A
    def findA(self):
        sigmaX = self.sigmaX()
        sigmaY = self.sigmaY()
        sigmaXY = self.sigmaXY()
        sigmaSQRX = self.sigmaSQRX()
        sigmaSQRY = self.sigmaSQRY()
        sigmaXSQR = np.square(sigmaX)
        result = ((sigmaY)*(sigmaSQRX) - (sigmaX)*(sigmaXY)) / ((self.n_train)*(sigmaSQRX) - (sigmaXSQR))
        return result

    #find B
    def findB(self):
        sigmaX = self.sigmaX()
        sigmaY = self.sigmaY()
        sigmaXY = self.sigmaXY()
        sigmaSQRX = self.sigmaSQRX()
        sigmaSQRY = self.sigmaSQRY()
        sigmaXSQR = np.square(sigmaX)
        result = ((self.n_train)*(sigmaXY) - (sigmaX)*(sigmaY)) / ((self.n_train)*(sigmaSQRX) - (sigmaXSQR))
        return result

    #Prediction
    def predict(self,x_data):
        valueA = self.findA()
        valueB = self.findB()
        y_pred = valueA + (valueB*x_data)
        return y_pred

def main():
    #Setting Dataset
    raw_dataset = pd.read_csv("Sales_Data.csv")
    mydata = pd.DataFrame(raw_dataset)

    raw_x = raw_dataset.iloc[:,0]
    raw_y = raw_dataset.iloc[:,1]

    #convert dataset to array
    x = np.asarray(raw_x)
    y = np.asarray(raw_y)

    #split dataset
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    #fitting dataset
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)

    #plot Sales_Data
    plt.scatter(mydata.BiayaPromo,mydata.NilaiPenjualan)
    plt.xlabel("Biaya Promosi")
    plt.ylabel("Nilai Penjualan")
    plt.title("Grafik Biaya Promosi dan Nilai Penjualan")
    plt.show()

    #plot Sales_Data (Training Set)
    plt.figure(figsize=(10,8))
    plt.scatter(x_train,y_train,color='blue')
    plt.plot(x_train,regressor.predict(x_train),color='red')
    plt.xlabel("Biaya Promosi")
    plt.ylabel("Nilai Penjualan")
    plt.title("Biaya Promosi Terhadap Penjualan (Training Set)")
    plt.show()

    #plot Sales_Data (Testing Set)
    plt.figure(figsize=(10,8))
    plt.scatter(x_test,y_test,color='blue')
    plt.plot(x_test,regressor.predict(x_test),color='red')
    plt.xlabel("Biaya Promosi")
    plt.ylabel("Nilai Penjualan")
    plt.title("Biaya Promosi Terhadap Penjualan (Testing Set)")
    plt.show()

    #Prediction
    y_pred_train = regressor.predict(x_train)
    y_pred_test = regressor.predict(x_test)

    #Score
    score_train = explained_variance_score(y_train,y_pred_train)
    score_test = explained_variance_score(y_test,y_pred_test)

    #Print Score
    print("Training Set Score = {}".format(score_train))
    print("Testing Set Score = {}".format(score_test))

if __name__ == "__main__":
    main()
