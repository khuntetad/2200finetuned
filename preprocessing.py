import os
import uuid
import time
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

os.environ[
    "OPENAI_API_KEY"] = "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"

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


# Function extracts text from each page in pdf and appends to a string
def extract_pdf_text(file_path):
    startTimer = time.time()
    # print("[DEBUG] Loading PDF from:", file_path)
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


# Function uses pytesseract to extract text from image and returns a string
def extract_text_from_image(image_path):
    print("[DEBUG] Processing image for OCR:", image_path)
    try:
        startTime = time.time()
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        # print(f"[DEBUG] OCR snippet ({image_path}): {text[:200]}")
        endTime = time.time()
        print(f"[UPLOAD] OCR time: {endTime - startTime} seconds")
        return text
    except Exception as e:
        print(f"[ERROR] Extracting text from image: {e}")
        return ""


# Function splits string variable containing text from whole pdf into chunks and creates a FAISS vector store
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
    if all_chunks:
        embeddings = OpenAIEmbeddings()
        if not existing_store:
            return FAISS.from_texts(all_chunks, embeddings)
        else:
            existing_store.add_texts(all_chunks)
            return existing_store
    else:
        return existing_store or None


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


# Creating initial vector store from 2200 textbook
try:
    text_from_default_pdf = extract_pdf_text("2200-textbook.pdf")
    # print("[DEBUG] Default PDF text length:", len(text_from_default_pdf))
    vector_store = create_vector_store([text_from_default_pdf])
    qa_chain = initialize_qa_chain(vector_store)
    print("[INIT] QA chain initialized with default textbook.")
except Exception as e:
    print("[ERROR] Initializing QA chain with default textbook:", e)


# A wrapper function to send data to the frontend
def res_template(message_type, message, status_code):
    return jsonify({message_type: message}), status_code


@app.route("/")
def index():
    return render_template("index.html")


from time import perf_counter


# This route takes the user's prompt and queries the vector store for useful info then sends the info along
# with the query to the OpenAI GPT4o model
@app.route("/ask", methods=["POST"])
def ask():
    global vector_store, qa_chain, latency_log

    if not qa_chain:
        return jsonify({"answer": "No documents have been uploaded yet or the chain was not initialized."}), 400

    question_text = ""
    if request.content_type and "multipart/form-data" in request.content_type:
        question_text = request.form.get("question", "").strip()
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            file_id = str(uuid.uuid4())
            unique_filename = f"{file_id}.{file_ext}"

            if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
                file_path = os.path.join(app.config['IMAGES_FOLDER'], unique_filename)
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(file_path)
            image_text = extract_text_from_image(file_path)

            if image_text.strip():
                image_text = f"IMAGE TEXT:\n{image_text}"
                if vector_store:
                    create_vector_store([image_text], vector_store)
                else:
                    vector_store = create_vector_store([image_text])
                qa_chain = initialize_qa_chain(vector_store)
                question_text = f"{image_text}\n\nUSER QUESTION:\n{question_text}"
    elif request.is_json:
        data = request.get_json()
        question_text = data.get("question", "").strip() if data else ""

    if not question_text:
        return res_template("answer", "No question provided.", 400)

    try:
        total_start = perf_counter()

        # This is where the retrieval occurs
        retrieval_start = perf_counter()
        retriever = qa_chain.retriever
        _ = retriever.invoke(question_text)
        retrieval_end = perf_counter()

        # This is where we pass on the retrieved data along with user prompt to GPT4o
        llm_start = perf_counter()
        result = qa_chain({"question": question_text})
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
        return res_template("answer", "An error occurred while processing your question.", 500)


# This route handles uploading additional pdf files and adding the text in them to the vector store
@app.route("/upload", methods=["POST"])
def upload_material():
    global vector_store, qa_chain

    upload_start_time = time.time()

    if "file" not in request.files:
        return res_template("error", "No file part", 400)

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return res_template("error", "Invalid file", 400)

    ext = file.filename.rsplit(".", 1)[1].lower()
    folder = app.config["IMAGES_FOLDER"] if ext in ["jpg", "jpeg", "png", "gif"] else app.config["UPLOAD_FOLDER"]
    path = os.path.join(folder, f"{uuid.uuid4()}.{ext}")
    file.save(path)

    if ext == "pdf":
        new_text = extract_pdf_text(path)
    else:
        new_text = extract_text_from_image(path)

    if not new_text.strip():
        return res_template("error", "Could not extract any text", 422)

    if vector_store:
        create_vector_store([new_text], existing_store=vector_store)
    else:
        vector_store = create_vector_store([new_text])

    qa_chain = initialize_qa_chain(vector_store)

    upload_end_time = time.time()
    upload_duration = round(upload_end_time - upload_start_time, 2)

    print(f"[UPLOAD] Upload and processing time: {upload_duration} seconds")

    return jsonify({
        "message": "Material added to knowledge base.",
        "chars_ingested": len(new_text)
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
