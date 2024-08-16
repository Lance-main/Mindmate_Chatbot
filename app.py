import streamlit as st
from dotenv import load_dotenv
import faiss
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime

# Predefined mental health advice
MENTAL_HEALTH_ADVICE = {
    "anxiety": "If you're feeling anxious, it's important to take deep breaths, try grounding exercises, and reach out to a friend or therapist if needed.",
    "anxious": "If you're feeling anxious, it's important to take deep breaths, try grounding exercises, and reach out to a friend or therapist if needed.",
    "depression": "If you're feeling depressed, consider talking to someone you trust, engage in activities that you enjoy, and don't hesitate to seek professional help.",
    "depressed": "If you're feeling depressed, consider talking to someone you trust, engage in activities that you enjoy, and don't hesitate to seek professional help.",
    "stress": "Managing stress can involve practicing relaxation techniques like meditation, ensuring you get enough sleep, and maintaining a balanced diet.",
    "loneliness": "When feeling lonely, try to connect with loved ones, engage in social activities, or explore hobbies that bring you joy.",
    "lonely": "When feeling lonely, try to connect with loved ones, engage in social activities, or explore hobbies that bring you joy.",
    "fear": "To manage fear, practice mindfulness, confront your fears gradually, and seek support from others or a professional.",
    "scared": "To manage fear, practice mindfulness, confront your fears gradually, and seek support from others or a professional.",
    "anger": "When dealing with anger, take a moment to breathe, express your feelings calmly, and engage in physical activity to release tension.",
    "grief": "During grief, allow yourself to feel emotions, seek support from others, and consider joining a support group to share your experience.",
    "mourning": "During grief, allow yourself to feel emotions, seek support from others, and consider joining a support group to share your experience.",
    "worry": "If you’re worrying excessively, try to focus on what you can control, practice relaxation techniques, and break tasks into smaller steps.",
    "panic": "In a panic situation, grounding techniques like focusing on your surroundings and deep breathing can help regain control.",
    "insomnia": "For insomnia, establish a regular sleep routine, create a relaxing bedtime environment, and avoid screens before bed.",
    "sadness": "When feeling sad, allow yourself to process emotions, engage in activities that uplift you, and talk to someone you trust.",
    "sad": "When feeling sad, allow yourself to process emotions, engage in activities that uplift you, and talk to someone you trust.",
    "burnout": "To combat burnout, prioritize self-care, set boundaries, and take breaks to recharge your energy.",
    "exhausted": "To combat burnout, prioritize self-care, set boundaries, and take breaks to recharge your energy.",
    "isolation": "If you’re feeling isolated, reach out to friends or family, join a community group, or explore online social platforms.",
    "overwhelm": "When overwhelmed, break tasks into smaller steps, prioritize what’s important, and ask for help when needed.",
    "overwhelmed": "When overwhelmed, break tasks into smaller steps, prioritize what’s important, and ask for help when needed.",
    "self-esteem": "Boost self-esteem by practicing self-compassion, setting achievable goals, and surrounding yourself with positive influences.",
    "worthless": "Boost self-esteem by practicing self-compassion, setting achievable goals, and surrounding yourself with positive influences.",
    "general": "For any mental health concerns, it's always beneficial to maintain a support system, practice self-care, and seek professional advice if necessary."

}

# Initialize TF-IDF vectorizer and fit it with keywords
vectorizer = TfidfVectorizer()
vectorizer.fit(MENTAL_HEALTH_ADVICE.keys())

# Function to find the best matching advice using TF-IDF and cosine similarity
def find_best_advice(query):
    query_vector = vectorizer.transform([query]).toarray()
    highest_similarity = 0
    best_advice = "general"

    for keyword, advice in MENTAL_HEALTH_ADVICE.items():
        keyword_vector = vectorizer.transform([keyword]).toarray()
        similarity = cosine_similarity(query_vector, keyword_vector)[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_advice = keyword

    return MENTAL_HEALTH_ADVICE[best_advice]

# Function to check if the query is asking for advice
def is_asking_for_advice(query):
    advice_keywords = ["should", "advice", "help", "cope", "deal with", "manage"]
    return any(keyword in query.lower() for keyword in advice_keywords)

# Function to authenticate and book an appointment
def book_appointment(date_time_str):
    try:
        # Load credentials from a file
        credentials = service_account.Credentials.from_service_account_file(
            'D:\\1_Full_Time_Job\\Project\\Chatbot_final\\credentials.json',
            scopes=["https://www.googleapis.com/auth/calendar"]
        )

        service = build('calendar', 'v3', credentials=credentials)

        # Ensure the date_time_str is in the correct ISO 8601 format
        event_start = datetime.datetime.fromisoformat(date_time_str)
        event_end = event_start + datetime.timedelta(hours=1)

        # Format the start and end times
        start_time = event_start.isoformat()
        end_time = event_end.isoformat()

        # Create an event with basic details
        event = {
            'summary': 'Medical Appointment',
            'location': 'Online',
            'description': 'Appointment with a medical professional',
            'start': {
                'dateTime': start_time,
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'America/New_York',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }

        # Insert the event into the calendar
        event = service.events().insert(calendarId='primary', body=event).execute()

        st.success("Your appointment has been booked successfully!")
        return event.get('htmlLink')
    
    except Exception as e:
        st.error(f"Failed to book appointment: {e}")
        return None

# Sidebar contents
with st.sidebar:
    st.title('MindMate🤗')
    st.markdown('''
    ## About
    This chatbot is designed to answer all your mental health-related questions by referencing official sources and articles.
                
    It Uses:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - Vector Store
    ''')
    add_vertical_space(5)
    st.write('Made by Lance Main')
    
    # Appointment Booking
    st.markdown("### Book an Appointment")
    appointment_date = st.date_input("Select the date for the appointment")
    appointment_time = st.time_input("Select the time for the appointment")
    if st.button("Confirm Booking"):
        date_time_str = f"{appointment_date}T{appointment_time}:00"
        appointment_link = book_appointment(date_time_str)
        if appointment_link:
            st.write(f"Your appointment has been booked. [View Appointment]({appointment_link})")

load_dotenv()

# Specify the folder where PDFs are stored
PDF_FOLDER = 'D:\\1_Full_Time_Job\\Project\\Chatbot_final\\Pdf'

# To keep track of conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def main():
    st.header("Know more about Mental Health💬")

    # List all PDFs in the specified folder
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Select the Pdf", pdf_files)

    if selected_pdf:
        pdf_path = os.path.join(PDF_FOLDER, selected_pdf)
        pdf_reader = PdfReader(pdf_path)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
            

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        documents = {i: Document(page_content=chunk) for i, chunk in enumerate(chunks)}

        store_name = selected_pdf[:-4]
        st.write(f'{store_name}')

        # Load or create FAISS index
        embeddings = OpenAIEmbeddings()
        if os.path.exists(f"{store_name}.index"):
            index = faiss.read_index(f"{store_name}.index")
            docstore = InMemoryDocstore(documents)
            index_to_docstore_id = {i: i for i in range(len(documents))}
            VectorStore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            faiss.write_index(VectorStore.index, f"{store_name}.index")

        # Conversational chatbot logic
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # Display the conversation history
        if st.session_state.conversation_history:
            st.markdown("### Conversation History")
            for entry in st.session_state.conversation_history:
                st.markdown(f"**User:** {entry['query']}")
                st.markdown(f"**Bot:** {entry['response']}")
                st.markdown("---")

        query = st.text_input("Ask questions about Mental health:")

        if query:
            # Store the conversation history
            st.session_state.conversation_history.append({"query": query})

            # Generate a conversation context string
            context = "\n".join([f"User: {item['query']}\nBot: {item.get('response', '')}" for item in st.session_state.conversation_history])

            # Check if the query is asking for advice
            if is_asking_for_advice(query):
                advice = find_best_advice(query)
                st.write(advice)
                st.session_state.conversation_history[-1]["response"] = advice  # Save the response
            elif "appointment" in query.lower():
                st.write("Let's book an appointment with a medical professional.")
                appointment_date = st.date_input("Select the date for the appointment")
                appointment_time = st.time_input("Select the time for the appointment")
                
                if st.button("Book Appointment"):
                    date_time_str = f"{appointment_date}T{appointment_time}:00"
                    appointment_link = book_appointment(date_time_str)
                    if appointment_link:
                        st.write(f"Your appointment has been booked. [View Appointment]({appointment_link})")
                    st.session_state.conversation_history[-1]["response"] = "Appointment booked."  # Save the response
            else:
                # Include the conversation context in the prompt
                full_query = f"{context}\nUser: {query}\nBot:"
                docs = VectorStore.similarity_search(query=full_query, k=5)
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=full_query)
                    print(cb)
                st.write(response)
                st.session_state.conversation_history[-1]["response"] = response  # Save the response

if __name__ == "__main__":
    main()
