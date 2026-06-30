from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQs
faq_data = {
    "What is Python?": "Python is a popular programming language.",
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Machine Learning?": "Machine Learning is a branch of AI.",
    "What is NLP?": "NLP stands for Natural Language Processing.",
    "Who developed Python?": "Python was developed by Guido van Rossum.",
    "What is ChatGPT?": "ChatGPT is an AI chatbot developed by OpenAI."
}

questions = list(faq_data.keys())

# NLP Preprocessing using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

print("===== FAQ Chatbot =====")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score > 0.2:
        print("Bot:", faq_data[questions[best_match_index]])
    else:
        print("Bot: Sorry, I couldn't find a relevant answer.")