import os
from flask import Flask, render_template, request, jsonify, session, redirect
from dotenv import load_dotenv
from functools import wraps
import json
from db import get_db


DURATION_PROMPT_MAP = {
    "Weekly": """
Create a detailed 7-day meal plan.
- Day-wise breakdown (Day 1 to Day 7)
- Breakfast, Lunch, Dinner, Snacks
- Clear portion guidance
""",

    "Bi-Weekly": """
Create a 14-day meal plan.
- Day-wise structure
- Avoid repeating meals too frequently
- Maintain nutritional balance across weeks
""",

    "Monthly": """
Create a 30-day meal plan.
- Week-wise structure
- Daily meal examples per week
- Focus on sustainability and variety
""",

    "Quarterly": """
Create a 3-month meal strategy.
- Month-wise goals
- Weekly sample meal plans
- Progressive adjustments over time
""",

    "Half Yearly": """
Create a 6-month nutrition roadmap.
- Month-wise phases
- Weekly examples where necessary
- Focus on habit building and adherence
""",

    "Yearly": """
Create a 12-month adaptive nutrition plan.
- Month-wise nutrition goals
- Seasonal considerations
- Sample weekly meal plans
- Long-term sustainability focus
"""
}




# =====================================================
# ENV
# =====================================================
load_dotenv()

os.environ["PINECONE_API_KEY"] = "pcsk_73qRHw_MDvLQzDSwBDTJ3As9TSuernzCdYi4hE4Z7VaoHbsvZRG6oynZydDbtvRWwLTpu4"
os.environ["GROQ_API_KEY"] = "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"

# =====================================================
# APP INIT
# =====================================================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "health_app_secret")

# =====================================================
# AUTH BLUEPRINT
# =====================================================


from auth import auth_bp
app.register_blueprint(auth_bp)


# =====================================================
# LOGIN REQUIRED DECORATOR
# =====================================================
def login_required(route):
    @wraps(route)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return route(*args, **kwargs)
    return wrapper



def save_chat(user_id, role, message):
    db = get_db()
    cursor = db.cursor()

    cursor.execute(
        """
        INSERT INTO meal_chat_history (user_id, role, message)
        VALUES (%s, %s, %s)
        """,
        (user_id, role, message)
    )

    db.commit()
    cursor.close()
    db.close()



def load_chat_history(user_id):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT role, message FROM meal_chat_history
        WHERE user_id = %s
        ORDER BY created_at
        """,
        (user_id,)
    )

    chats = cursor.fetchall()
    cursor.close()
    db.close()

    return chats



# =====================================================
# MEAL PLANNER (Groq)
# =====================================================
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

meal_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

sessions_memory = {}

def get_meal_chain(session_id, duration=None):
    if session_id not in sessions_memory:
        sessions_memory[session_id] = ConversationBufferMemory()

    duration_instruction = DURATION_PROMPT_MAP.get(
        duration,
        "Create a practical, balanced meal plan."
    )

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""
You are an expert Nutritionist and Health AI Assistant who gives answers to health and nutrition questions ONLY. If the question is not related to health or nutrition, response with I cannot answer politely. Be confident about what you give. Even if the user says you are wrong, stand by your answer.

Plan Duration Instructions:
{duration_instruction}

Rules:
- Respect diet type, allergies, and medical conditions strictly
- Keep meals culturally appropriate
- Avoid unnecessary explanations
- Be clear, structured, and practical

Conversation so far:
{{history}}

User: {{input}}
AI:
"""
    )

    return ConversationChain(
        llm=meal_llm,
        memory=sessions_memory[session_id],
        prompt=prompt
    )


# =====================================================
# FITNESS ASSISTANT (RAG)
# =====================================================
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate as RAGPrompt
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "developer-quickstart-py"
TEXT_FIELD = "text"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key=TEXT_FIELD
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

FITNESS_PROMPT = RAGPrompt(
    template="""
You are a professional fitness coach and personal trainer.

Use the provided context if it is relevant.
If the context is not helpful, answer using your general fitness knowledge.

Do NOT mention the context explicitly.
Do NOT say that the context is missing.
Give clear, practical, actionable advice.
Give the answer in a structured and concise manner.

Context:
{context}

User Question:
{question}

Fitness Coach Answer:
""",
    input_variables=["context", "question"]
)

fitness_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

fitness_qa = RetrievalQA.from_chain_type(
    llm=fitness_llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": FITNESS_PROMPT}
)

# =====================================================
# ROUTES
# =====================================================

# ---------------- DASHBOARD ----------------
@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")

# ---------------- FITNESS ASSISTANT ----------------
@app.route("/fitness-chat")
@login_required
def fitness_chat():
    return render_template("fitness_chat.html")

@app.route("/fitness-message", methods=["POST"])
@login_required
def fitness_message():
    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"reply": "Please enter a question."})

    try:
        result = fitness_qa.invoke({"query": msg})
        return jsonify({"reply": result["result"]})
    except Exception as e:
        print(e)
        return jsonify({"reply": "⚠️ Fitness AI is temporarily unavailable."})

# ---------------- MEAL PLANNER ----------------
@app.route("/meal-planner")
@login_required
def meal_planner():
    user_id = session["user_id"]

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT profile_json FROM meal_planner_state WHERE user_id = %s",
        (user_id,)
    )
    profile = cursor.fetchone()

    cursor.close()
    db.close()

    # FIRST TIME → show form
    if not profile:
        return render_template("index.html")

    # NOT first time → show chatbot
    return redirect("/meal-chat-ui")


@app.route("/plan", methods=["POST"])
@login_required
def generate_plan():
    user_id = session["user_id"]
    profile = dict(request.form)

    db = get_db()
    cursor = db.cursor()

    cursor.execute(
        """
        INSERT INTO meal_planner_state (user_id, profile_json)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE profile_json = VALUES(profile_json)
        """,
        (user_id, json.dumps(profile))
    )

    db.commit()
    cursor.close()
    db.close()

    # Generate initial plan as before
    duration=profile.get("duration","Weekly")
    chain=get_meal_chain(user_id, duration=duration)
    response = chain.predict(input=f"Create meal plan for: {profile}")

    # Save AI message
    save_chat(user_id, "ai", response)

    return render_template("chat.html", plan=response)


@app.route("/meal-chat-ui")
@login_required
def meal_chat_ui():
    user_id = session["user_id"]
    chat_history = load_chat_history(user_id)
    return render_template("chat.html", chat_history=chat_history)


@app.route("/meal-chat", methods=["POST"])
@login_required
def meal_chat():
    user_id = session["user_id"]
    user_msg = request.json.get("message")

    if not user_msg:
        return jsonify({"response": "Please enter a message."})

    # -------------------------------
    # 1. Save user message
    # -------------------------------
    save_chat(user_id, "user", user_msg)

    # -------------------------------
    # 2. Load stored meal profile to get duration
    # -------------------------------
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT profile_json FROM meal_planner_state WHERE user_id = %s",
        (user_id,)
    )
    profile_row = cursor.fetchone()

    cursor.close()
    db.close()

    # Default safety fallback
    duration = "Weekly"

    if profile_row and profile_row.get("profile_json"):
        profile = json.loads(profile_row["profile_json"])
        duration = profile.get("duration", "Weekly")

    # -------------------------------
    # 3. Get duration-aware chain
    # -------------------------------
    chain = get_meal_chain(user_id, duration=duration)

    response = chain.predict(input=user_msg)

    # -------------------------------
    # 4. Save AI response
    # -------------------------------
    save_chat(user_id, "ai", response)

    return jsonify({"response": response})




# @app.route("/meal-chat", methods=["POST"])
# @login_required
# def meal_chat():
#     user_msg = request.json.get("message")
#     chain = get_meal_chain(session["user_id"])
#     response = chain.predict(input=user_msg)
#     return jsonify({"response": response})

# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
