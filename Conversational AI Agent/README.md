# Conversational AI Agent

This project implements a conversational AI agent using LangChain and Panel, featuring tools for weather information, Wikipedia search, and custom text manipulation. The UI is built with Panel, providing an interactive chat experience.

## Screenshot

![Conversational AI Agent Screenshot](./assets/screenshot(121).png)
![Conversational AI Agent Screenshot](./assets/screenshot(122).png)

*(Note: You will need to create an `assets` folder in your project root and add a `screenshot.png` file for the image to display.)*


## Local Development

To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Conversational AI Agent
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up OpenAI API Key**:
    Create a `.env` file in the `Conversational AI Agent` directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

5.  **Run the application**:
    ```bash
    panel serve app.py --show
    ```
    This will open the application in your web browser.

## Features

*   **Weather Information**: Get current temperature for any location using the Open-Meteo API.
*   **Wikipedia Search**: Search and get summaries from Wikipedia.
*   **Custom Tools**: Example tool for text reversal.
*   **Memory**: The agent remembers conversation context.
