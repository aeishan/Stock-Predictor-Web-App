from datetime import date
import streamlit as st 

import yfinance as yf
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predication App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks) # creates a drop down with the stocks tuple

n_years = st.slider("Years of prediction:", 1, 4) #creates a slider 
period = n_years*365 #for future predictions, we need it in days

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY) #downloads the data from START to TODAY of the stock ticker
    data.reset_index(inplace = True) #puts date in the first column
    
    return data #returns in a pandas dataframe

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading Data...Done!")

st.subheader('Raw Data')
st.write(data.tail()) #creates and outputs a table of the last 10 rows of the 'data'

def plot_data():
    fig = go.Figure() #creating graph object  (since the 'mode' attribute was not defined when adding traces, it will display both lines and points on graph)
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open')) #plots data from Open values  (like a scatter plot)
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close')) #plots data from Close values (like a scatter plot)
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True) # creates slider and updates values accordingly
    st.plotly_chart(fig) # plots the plotly figure (note: this graph is interactive)

plot_data()

# Forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  #how the prophet needs the data (standard) #renaming the columns (how its done with pandas since data is a
                                                                                                              #pandas dataframe)

model = Prophet() #creating a model using the Prophet() class
model.fit(df_train) #training the model using the revamped variable we made

future = model.make_future_dataframe(periods=period) #creating a dataframe that can hold the predictions of the selected stock   #we need the period of prediction to be in days
forecast = model.predict(future)  #now we have predicted the selected stock

st.subheader(f"Forecasted Data (predicting {period//365} years ahead of today):")
st.write(forecast.tail())

st.write(f"Forecasted Data (predicting {period//365} years ahead of today):")
figure1 = plot_plotly(model, forecast)  #when dealing with forecast, use plot_plotly() (returns an interactive plot graph) (automatically adds a range slider)
#figure1.show()
st.plotly_chart(figure1) #plots the interactive plot_plotly() graph

st.write('Forecast Components')
figure2 = model.plot_components(forecast) #returns a plot(s) that shows the diff components used to make the forecast predictions 
st.write(figure2) #since figure2 is not a plot_ploty() graph, we can use st.write()
