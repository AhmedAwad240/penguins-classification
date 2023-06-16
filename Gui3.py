from tkinter import *
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import MLP


base = Tk()

# Using the Geometry method to the form certain dimensions
base.geometry("500x500")

# Using title method to give the title to the window
base.title('Task 1')

# Adelie = 0 Chinstrap = 1 Gentoo = 2 this is the encoding order


df = pd.read_csv('penguins.csv', index_col=False, encoding="utf-8")
df.reset_index(drop=True, inplace=True)

# preprocessing
df['gender'] = df['gender'].fillna('male')
le = LabelEncoder()
df["gender"] = df.apply(le.fit_transform)["gender"]
df["species"] = df.apply(le.fit_transform)["species"]
train, test = MLP.dataframesplit(df)

# Data Splitting
y_train = train["species"]
y_test = test["species"]
X_train = train.drop(["species"], axis=1).values
X_test = test.drop(["species"], axis=1).values

# normalize the input X
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
# Model Training and testing


def GetData():
    l = layer.get()
    learning_str = learningRate.get()
    alpha = learningRate.getdouble(learning_str)
    itr = int(num_of_ebochs.get())
    Bias = int(bias.get())
    threshold = fun.get()
    l =list(map(int, l.split()))
    print(l,len(np.unique(y_train)),X_train.shape[1])
    model = MLP.NN(learning_rate=alpha, max_iter=itr, bias=Bias, threshold=threshold)
    model.layers(inputSize=X_train.shape[1], layerSizes=l, numOfOutput=len(np.unique(y_train)))
    model.train(X_train, y_train)
    score , matrix = model.test(X_test, y_test)
    print("test score", score )
    print("confusion matrix\n", matrix)


# Using 'Label4' widget to create learning rate label and using place() method to set its position.
lbl_5 = Label(base, text="neurons in each layer", width=20, font=("bold", 11))
lbl_5.place(x=60, y=270)
layer = Entry(base)
layer.place(x=240, y=270)


# Using 'Label4' widget to create learning rate label and using place() method to set its position.
lbl_4 = Label(base, text="Enter learning rate", width=20, font=("bold", 11))
lbl_4.place(x=60, y=100)

# Using Enrty widget to make a text entry box for accepting the input string in text from user.
learningRate = Entry(base)
learningRate.place(x=240, y=100)

# -------------------------------------------------------------------------------------------------------------#

# Using 'Label5' widget to create Num of ebochs label and using place() method to set its position.
lbl_5 = Label(base, text="Number of ebochs", width=20, font=("bold", 11))
lbl_5.place(x=60, y=150)

# Using Enrty widget to make a text entry box for accepting the input string in text from user.
num_of_ebochs = Entry(base)
num_of_ebochs.place(x=240, y=150)

# ------------------------------------------------------------------------------------------------------------#

# Using 'Label6' widget to create Bias label and using place() method, set its position.
lbl_6 = Label(base, text="Bias", width=20, font=('bold', 10))
lbl_6.place(x=120, y=200)

# the new variable 'vars1' is created to store Integer Value, which by default is 0.
bias = IntVar()
# Using the Checkbutton widget to create a button and using place() method to set its position.
Checkbutton(base, text="", variable=bias).place(x=235, y=200)


# -------------------------------------------------------------------------------------------------------------#


# Using 'Label4' widget to create learning rate label and using place() method to set its position.
lbl_8 = Label(base, text="Sigmoid or Hyperbolic ", width=20, font=("bold", 11))
lbl_8.place(x=60, y=300)

# Using Enrty widget to make a text entry box for accepting the input string in text from user.
fun = Entry(base)
fun.place(x=240, y=300)

#



# ----------------------------------------------------------------------------------------------------------#

# Using the Button widget, we get to create a button for submitting all the data that has been entered in the entry boxes of the form by the user.
# button=Button(base, text='Submit', width=20, bg="grey", fg='white',command).place(x=160, y=420)
b = Button(base, text='Submit', width=20, bg='brown', fg='white', command=GetData)
b.place(x=180, y=350)
# b.pack()

# b2=Button(base, text='Submit',width=20,bg='brown',fg='white',command=plot_it)
# b.place(x=180,y=380)
# b.pack()

base.mainloop()
