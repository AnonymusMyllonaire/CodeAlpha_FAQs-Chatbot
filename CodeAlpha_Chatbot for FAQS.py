from flask import Flask, render_template_string, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# Download NLTK stopwords (only first time)
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# --------------------------
# FAQ DATA
# --------------------------
faq_data = [
    {"question": "How can I reset my password?", "answer": "Go to settings and click on 'Reset Password'."},
    {"question": "What is your refund policy?", "answer": "You can request a refund within 30 days of purchase."},
    {"question": "How do I contact customer support?", "answer": "You can email us at support@example.com."},
    {"question": "Where can I download the app?", "answer": "The app is available on both App Store and Google Play."},
    {"question": "Is there a free trial available?", "answer": "Yes, we offer a 7-day free trial for all new users."},
    {"question": "How do I change my email address?", "answer": "Go to account settings and update your email address."},
    {"question": "Can I use the app offline?", "answer": "Some features are available offline, but syncing requires internet."},
    {"question": "Do you offer student discounts?", "answer": "Yes, students get 20% off with a valid student ID."},
    {"question": "How can I cancel my subscription?", "answer": "You can cancel anytime from the billing section in your account."},
    {"question": "Is my data secure?", "answer": "Yes, we use encryption and follow strict security protocols to protect your data."},
    {"question": "Can I upgrade my plan later?", "answer": "Yes, you can upgrade or downgrade your plan anytime in settings."},
    {"question": "Do you support multiple languages?", "answer": "Currently, we support English, Spanish, French, and German."},
    {"question": "What payment methods do you accept?", "answer": "We accept credit cards, debit cards, and PayPal."},
    {"question": "Can I use the same account on multiple devices?", "answer": "Yes, your account can be accessed on multiple devices."},
    {"question": "Do you provide tutorials for beginners?", "answer": "Yes, we have step-by-step guides and video tutorials on our website."}
]


# Extract only the questions
faq_questions = [faq["question"] for faq in faq_data]
faq_answers = [faq["answer"] for faq in faq_data]

# --------------------------
# NLP PREPROCESSING
# --------------------------
stop_words = set(stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)

def preprocess(text):
    text = text.lower().translate(translator)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

preprocessed_questions = [preprocess(q) for q in faq_questions]

# Fit TF-IDF
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

# --------------------------
# MATCHING FUNCTION
# --------------------------
def find_best_answer(user_query):
    user_query_processed = preprocess(user_query)
    user_vector = vectorizer.transform([user_query_processed])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0, best_match_index]

    # Threshold to avoid nonsense matches
    if best_score < 0.3:
        return "Sorry, I couldn't find a relevant answer. Please contact support."
    return faq_answers[best_match_index]

# --------------------------
# HTML TEMPLATE
# --------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FAQ Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 0; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h2 { text-align: center; }
        #chat-box { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 8px; background: #fafafa; }
        .user { text-align: right; color: blue; margin: 5px 0; }
        .bot { text-align: left; color: green; margin: 5px 0; }
        input, button { padding: 10px; border-radius: 8px; border: 1px solid #ccc; width: 80%; }
        button { width: 18%; background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🤖 FAQ Chatbot</h2>
        <div id="chat-box"></div>
        <div>
            <input type="text" id="user-input" placeholder="Ask a question..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        function sendMessage() {
            const input = document.getElementById("user-input");
            const message = input.value.trim();
            if (!message) return;

            const chatBox = document.getElementById("chat-box");
            chatBox.innerHTML += `<div class='user'><b>You:</b> ${message}</div>`;
            input.value = "";

            fetch("/get_answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: message })
            })
            .then(response => response.json())
            .then(data => {
                chatBox.innerHTML += `<div class='bot'><b>Bot:</b> ${data.answer}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            });
        }
    </script>
</body>
</html>
"""

# --------------------------
# ROUTES
# --------------------------
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/get_answer", methods=["POST"])
def get_answer():
    user_data = request.get_json()
    user_question = user_data.get("question", "")
    answer = find_best_answer(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
