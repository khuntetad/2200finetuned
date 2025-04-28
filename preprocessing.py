import os
import uuid
import io
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from PIL import Image
from time import perf_counter  # use perf_counter for better precision
import pytesseract

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGES_FOLDER'] = 'uploads/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'jpg', 'jpeg', 'png', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)


vector_store = None
qa_chain = None
latest_image_text = ""

latency_log = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_pdf_text(file_path):
    startTimer = time.time()
    print("[DEBUG] Loading PDF from:", file_path)
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        endTimer = time.time()
        print(f"[UPLOAD] Textbook: {endTimer - startTimer} seconds") 
        return text
        

    except Exception as e:
        print(f"[ERROR] Extracting text from PDF: {e}")
        return ""

def extract_text_from_image(image_path):
    
    print("[DEBUG] Processing image for OCR:", image_path)
    try:
        startTime = time.time()
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print(f"[DEBUG] OCR snippet ({image_path}): {text[:200]}")
        endTime = time.time()
        print(f"[UPLOAD] OCR time: {endTime - startTime} seconds") 
        return text
    except Exception as e:
        print(f"[ERROR] Extracting text from image: {e}")
        return ""
    

def create_vector_store(texts, existing_store=None):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        keep_separator=True
    )

    all_chunks = []
    for text in texts:
        if text.strip():
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
    if not all_chunks:
        return existing_store or None

    embeddings = OpenAIEmbeddings()
    if existing_store:
        existing_store.add_texts(all_chunks)
        return existing_store
    else:
        return FAISS.from_texts(all_chunks, embeddings)

def initialize_qa_chain(vstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    custom_prompt = PromptTemplate.from_template(
        """
        You are a helpful and knowledgeable TA for a college course using a textbook.

        Your goal is to guide the student through thinking and understanding — **not** to provide direct answers.

        Follow these rules carefully:
        - Do **not** provide final numeric or factual answers, even if the student asks for it directly.
        - Instead, explain how the student should think through the problem.
        - Use definitions, examples, and step-by-step reasoning.
        - If formulas or calculations are needed, show how to set them up, but do **not** solve them fully.

        Special instruction:
        If the input includes "IMAGE TEXT:", that content is from an uploaded image and should take priority over other context if relevant.

        {context}

        Question: {question}
        Response:
        """
    )


    llm = ChatOpenAI(temperature=0.15, model="gpt-4o")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(search_type="mmr", search_kwargs={"k": 6}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose=True
    )
    return chain

try:
    text_from_default_pdf = extract_pdf_text("2200-textbook.pdf")
    print("[DEBUG] Default PDF text length:", len(text_from_default_pdf))
    vector_store = create_vector_store([text_from_default_pdf])
    qa_chain = initialize_qa_chain(vector_store)
    print("[INIT] QA chain initialized with default textbook.")
except Exception as e:
    print("[ERROR] Initializing QA chain with default textbook:", e)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "has_vector_store": vector_store is not None,
        "has_qa_chain": qa_chain is not None
    })

from time import perf_counter  # use perf_counter for better precision

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store, qa_chain, latency_log

    if not qa_chain:
        return jsonify({"answer": "No documents have been uploaded yet or the chain was not initialized."}), 400

    question_text = ""
    if request.content_type and "multipart/form-data" in request.content_type:
        question_text = request.form.get("question", "").strip()
    elif request.is_json:
        data = request.get_json()
        question_text = data.get("question", "").strip() if data else ""

    if not question_text:
        return jsonify({"answer": "No question provided."}), 400

    try:
        total_start = perf_counter()

        # Step 1: Retrieval step timing
        retrieval_start = perf_counter()
        retriever = qa_chain.retriever
        _ = retriever.invoke(question_text)  # <-- just invoke to measure retrieval time
        retrieval_end = perf_counter()

        # Step 2: LLM Generation step timing
        llm_start = perf_counter()
        result = qa_chain({"question": question_text})  # <-- do NOT touch internals, call as normal
        llm_end = perf_counter()

        total_end = perf_counter()

        latency_log.append(total_end - total_start)
        avg_latency = sum(latency_log) / len(latency_log)

        answer = result.get("answer", "I couldn't find an answer.")
        token_usage = result.get("usage", "Not available")

        response = {
            "answer": answer,
            "step_latencies": {
                "retrieval_time": round(retrieval_end - retrieval_start, 4),
                "llm_generation_time": round(llm_end - llm_start, 4),
                "total_latency": round(total_end - total_start, 4)
            },
            "average_latency": round(avg_latency, 4),
            "token_usage": token_usage
        }

        print(f"[ASK] Detailed Timing: {response['step_latencies']}")

        return jsonify(response)
    except Exception as e:
        print(f"[ASK] Error during chain execution: {e}")
        return jsonify({"answer": "An error occurred while processing your question."}), 500

@app.route("/latency", methods=["GET"])
def get_latency():
    if latency_log:
        avg = round(sum(latency_log) / len(latency_log), 2)
    else:
        avg = 0.0
    return jsonify({
        "average_latency": avg,
        "requests_count": len(latency_log)
    })

@app.route("/upload", methods=["POST"])
def upload_material():
    global vector_store, qa_chain

    upload_start_time = time.time()  # start timer

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported extension"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = f"{uuid.uuid4()}.{ext}"
    folder = app.config["IMAGES_FOLDER"] if ext in ["jpg", "jpeg", "png", "gif"] else app.config["UPLOAD_FOLDER"]
    path = os.path.join(folder, uid)
    file.save(path)

    if ext == "pdf":
        new_text = extract_pdf_text(path)
    else:
        new_text = extract_text_from_image(path)

    if not new_text.strip():
        return jsonify({"error": "Could not extract any text"}), 422

    if vector_store:
        create_vector_store([new_text], existing_store=vector_store)
    else:
        vector_store = create_vector_store([new_text])

    qa_chain = initialize_qa_chain(vector_store)

    upload_end_time = time.time()  # end timer
    upload_duration = round(upload_end_time - upload_start_time, 2)

    print(f"[UPLOAD] Upload and processing time: {upload_duration} seconds") 

    return jsonify({
        "message": "Material added to knowledge base.",
        "chars_ingested": len(new_text)
    })

@app.route("/clear", methods=["POST"])
def clear_conversation():
    global qa_chain, vector_store
    if vector_store:
        qa_chain = initialize_qa_chain(vector_store)
        return jsonify({"message": "Conversation memory cleared."})
    else:
        return jsonify({"error": "No active vector store"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)