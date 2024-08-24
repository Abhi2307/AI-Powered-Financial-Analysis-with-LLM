### AI-Powered-Financial-Analysis-with-LLM

Project Description
This project combines Long Short-Term Memory (LSTM) networks and Large Language Models (LLMs) to predict the next day's closing price of a stock. The prediction leverages not only historical stock data but also sentiment analysis performed on recent news headlines related to the stock. The sentiment analysis is conducted using an LLM, enhancing the prediction accuracy by incorporating public sentiment. The project is built using Python, with a front-end interface created using Streamlit.
## Project Structure

├── aiapp.py                   # Main application file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── lstm_stock_model.keras     # Pre-trained LSTM model

## Installation Instructions
# Clone the Repository:
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
# Install Dependencies
Ensure you have Python 3.8+ installed. Install the required Python packages using pip:
pip install -r requirements.txt
# Download or Prepare the Dataset
Stock Price Data: Download historical stock price data from sources like Yahoo Finance.
News Headlines: Collect relevant headlines for sentiment analysis using financial news APIs or scraping.
# Run the Application
streamlit run aiapp.py

## Contributing
Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This project was inspired by various stock prediction models and sentiment analysis techniques.
The LSTM model and sentiment analysis are powered by keras and transformers libraries.
## Next Steps
Experiment with different LLMs for sentiment analysis.
Integrate real-time data fetching for continuous predictions.
Enhance the user interface with interactive visualizations and more detailed sentiment analysis.

