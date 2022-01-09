import tkinter as tk
import tkinter.ttk as ttk
import time as t
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
import tkcap
import os
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

widget = []

def exxit():
    window.destroy()
    quit()

def screen():
    cap = tkcap.CAP(window)  # master is an instance of tkinter.Tk
    filename = "predict_screen_" + str(dt.datetime.now())
    try:
        desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        cap.capture(str(desktop) + "/"+filename + ".png")
        messagebox.showinfo("showinfo", "saved on Desktop!")
    except:
        cap.capture(filename + ".png")
        messagebox.showinfo("showinfo", "Saved in the same folder as py source")

def destroy_and_create():

    Msg = messagebox.askquestion("Verification", "Check input and click yes to begin prediction")
    if Msg=="yes":
        for wid in widget:
            wid.destroy()
        wait = tk.Label(window, text="Please wait...", font=("Ubuntu Mono", 30, 'bold'))
        wait.place(x=200, y=200)

        window.update()
        window.after_idle(predict)

def predict():
    stock = var.get()  # fb tsl btc oil gold eurusd
    period = days.get()  # 15 30 45
    optimiser = optimizer.get()  # adamax sgd adam adadelta
    if stock == 1:
        stock = "FB"
    elif stock == 2:
        stock = "TSLA"
    elif stock == 3:
        stock = "AAPL"
    elif stock == 4:
        stock = "CL=F"
    elif stock == 5:
        stock = "GC=F"
    else:
        stock = "EURUSD=X"

    if period == 1:
        period = 15
    elif period == 2:
        period = 30
    else:
        period = 45

    if optimiser == 1:
        optim = "adamax"
    elif optimiser == 2:
        optim = "sgd"
    elif optimiser == 3:
        optim = "adam"
    else:
        optim = "adadelta"

    print(stock)
    print(period)
    print(optim)

    company = stock

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 12, 12)
    data = web.DataReader(company, 'yahoo', start, end)

    # Prepare data
    # este o practica buna sa transformam datele inainte sa le folosim

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = period

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # prediction of next closing price

    model.compile(optimizer=optim, loss='mean_squared_error')
    hist = model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Load test data

    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make predictions on test data

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # plot the test prediction
    fig = plt.figure(figsize=(5, 5))
    plt.plot(actual_prices, color="black", label=f"Actual {company} price")
    plt.plot(predicted_prices, color="green", label="Predicted")
    plt.title(f"{company} share price")
    plt.xlabel('Time')
    plt.ylabel(f"{company} share price")

    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    rmse = mean_squared_error(actual_prices,predicted_prices,squared=False)
    loss = np.sqrt(np.mean(np.square(((actual_prices - predicted_prices) / actual_prices)), axis=0))
    loss = np.median(loss) * 100
    print(f"Loss {loss:.4f}%")
    print(f"Prediction {prediction}")

    window.geometry('1200x800')
    # specify the window as master
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, ipadx=40, ipady=20)

    # navigation toolbar
    toolbarFrame = tk.Frame(master=window)
    toolbarFrame.grid(row=0, column=0)
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    save_png = tk.Button(window, text="Save results as png", font=("Ubuntu Mono", 20, 'bold'),command=screen)
    save_png.place(x=700, y=550)

    exit = tk.Button(window, text="Exit", font=("Ubuntu Mono", 20, 'bold'),command=exxit)
    exit.place(x=700, y=750)

    result = tk.Label(window, text="Predicted price : " + str(prediction), font=("Ubuntu Mono", 20, 'bold'))
    result.place(x=700, y=600)

    loss_label = tk.Label(window, text="Error : " + str(round(loss, 2)) + " %", font=("Ubuntu Mono", 20, 'bold'))
    loss_label.place(x=700,y=650)

def clicked():
    window.geometry('1000x900')
    lbl.destroy()
    btn.destroy()
    select_stock = tk.Label(window, text="Select stock", font=("Ubuntu Mono", 30, 'bold'))
    select_stock.place(x=20, y=20)
    widget.append(select_stock)

    fb = tk.Radiobutton(window, text="Facebook", font=("Ubuntu Mono", 20), variable=var, value=1)
    fb.place(x=10, y=80)
    widget.append(fb)
    tsl = tk.Radiobutton(window, text="Tesla", font=("Ubuntu Mono", 20), variable=var, value=2)
    tsl.place(x=160, y=80)
    widget.append(tsl)
    btc = tk.Radiobutton(window, text="Apple", font=("Ubuntu Mono", 20), variable=var, value=3)
    btc.place(x=280, y=80)
    widget.append(btc)
    oil = tk.Radiobutton(window, text="Oil", font=("Ubuntu Mono", 20), variable=var, value=4)
    oil.place(x=10, y=140)
    widget.append(oil)
    gold = tk.Radiobutton(window, text="Gold", font=("Ubuntu Mono", 20), variable=var, value=5)
    gold.place(x=160, y=140)
    widget.append(gold)
    eur_usd = tk.Radiobutton(window, text="EUR/USD", font=("Ubuntu Mono", 20), variable=var, value=6)
    eur_usd.place(x=280, y=140)
    widget.append(eur_usd)

    select_day = tk.Label(window, text="Choose period of days to train the model", font=("Ubuntu Mono", 30, 'bold'))
    select_day.place(x=20, y=240)
    widget.append(select_day)
    d15 = tk.Radiobutton(window, text="15 days", font=("Ubuntu Mono", 20), variable=days, value=1)
    d15.place(x=20, y=320)
    widget.append(d15)
    d30 = tk.Radiobutton(window, text="30 days", font=("Ubuntu Mono", 20), variable=days, value=2)
    d30.place(x=160, y=320)
    widget.append(d30)
    d45 = tk.Radiobutton(window, text="45 days", font=("Ubuntu Mono", 20), variable=days, value=3)
    d45.place(x=280, y=320)
    widget.append(d45)

    select_optimizer = tk.Label(window, text="Select optimizer", font=("Ubuntu Mono", 30, 'bold'))
    select_optimizer.place(x=20, y=400)
    widget.append(select_optimizer)
    opt1 = tk.Radiobutton(window, text="Adamax", font=("Ubuntu Mono", 20), variable=optimizer, value=1)
    opt1.place(x=20, y=480)
    widget.append(opt1)
    opt2 = tk.Radiobutton(window, text="SGD (Gradient Descent with momentum)", font=("Ubuntu Mono", 20),
                          variable=optimizer, value=2)
    opt2.place(x=160, y=480)
    widget.append(opt2)
    opt3 = tk.Radiobutton(window, text="Adam", font=("Ubuntu Mono", 20), variable=optimizer, value=3)
    opt3.place(x=20, y=540)
    widget.append(opt3)
    opt4 = tk.Radiobutton(window, text="Adadelta", font=("Ubuntu Mono", 20), variable=optimizer, value=4)
    opt4.place(x=160, y=540)
    widget.append(opt4)

    rec_label = tk.Label(window,
                         text="Recommandations \nFor best results the optimizer Adam\nAnd for days choose 30",
                         font=("Ubuntu Mono", 16), background="#c4bdb7")
    rec_label.place(x=20, y=750)
    widget.append(rec_label)
    predict_btn = tk.Button(window, text="Continue", font=("Ubuntu Mono", 20, 'bold'),
                            command=destroy_and_create)
    predict_btn.place(x=700, y=640)
    widget.append(predict_btn)


window = tk.Tk()

window.title("Project")

window.geometry('500x300')

lbl = tk.Label(window, text="Stock price \n prediction tool", font=("Ubuntu Mono", 30))

lbl.place(x=100, y=20)

btn = tk.Button(window, text="Begin", command=clicked, font=("Ubuntu Mono", 20))

btn.place(x=200, y=200)

var = tk.IntVar()
days = tk.IntVar()
optimizer = tk.IntVar()

window.mainloop()
