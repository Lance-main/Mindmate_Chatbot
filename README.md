
# MindMate 🤗

MindMate is an AI-driven mental health chatbot designed to provide mental health-related advice and help users book medical appointments. The chatbot leverages the power of OpenAI, LangChain, and vector-based search to answer questions based on a set of reference PDFs.

## Features

- **Mental Health Guidance**: Provides mental health advice for various conditions such as anxiety, depression, stress, and more.
- **PDF Referencing**: Answers questions based on content from specified PDF files. These pdf files are created by compiling material from official mental health platforms.
- **Appointment Booking**: Allows users to book appointments via Google Calendar.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Pip (Python package installer)
- A Google Cloud project with Calendar API enabled

### Step-by-Step Guide

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/mindmate-chatbot.git
   cd mindmate-chatbot
   ```

2. **Set Up the Environment**

   Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install the required Python packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Keys**

   - **OpenAI API Key**: Create an `.env` file in the root directory and add your OpenAI API key:

     ```bash
     echo "OPENAI_API_KEY=your-openai-api-key" > .env
     ```

   - **Google Calendar Credentials**: Download your `credentials.json` file from Google Cloud and place it in the root directory.

5. **Prepare Your PDFs**

   - Place all the reference PDFs in a folder named `Pdf` within the project directory. The chatbot will reference these PDFs to answer mental health-related questions.

6. **Run the Application**

   Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

7. **Access the Chatbot**

   Open your web browser and go to `http://localhost:8501` to start using the chatbot.

## Code Base

This repository includes all the necessary code files to implement the chatbot:

- **Prompt Engineering**: Tailor prompts for specific mental health-related queries.
- **RAG (Retrieval-Augmented Generation)**: Integrate RAG to improve the quality and relevance of chatbot responses by referecing pdf data.
- **LLMs (Large Language Models)**: Utilize OpenAI’s GPT model GPT-3.5 turbo for natural language understanding and generation.
- **LangChain**: Employ LangChain for efficient text splitting and document management.

Each code file is well-documented with comments to ensure clarity.

## SetUp Instructions

- **Dependencies**: Ensure you have the correct Python version and all dependencies listed in `requirements.txt`.
- **Libraries/Frameworks**: The project relies on Streamlit for the web interface, OpenAI for natural language processing, LangChain for document handling, FAISS for vector-based search, and Google Calendar API for appointment booking.
- **Configuration**: Set up your environment variables in the `.env` file and ensure your `credentials.json` file is properly configured.

## Video Demonstration & Code Explanation

Check out this YouTube video for a complete demonstration and code explanation of the chatbot:

[![AI-Driven Mental Health Chatbot](https://img.youtube.com/vi/dXnDA0U6YyA/0.jpg)](https://youtu.be/dXnDA0U6YyA)

[Watch the full video here](https://youtu.be/dXnDA0U6YyA).


## Usage

- **Asking Questions**: Enter your mental health-related queries in the text box.
- **Booking Appointments**: Use the sidebar to book an appointment with a medical professional.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is developed by me, Lance Main. I grant permission for peers to utilize this work.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://platform.openai.com/)
- [LangChain](https://python.langchain.com/)
