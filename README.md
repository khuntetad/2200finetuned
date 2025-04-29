# CS 2200 AI Teaching Assistant

A lightweight Flask‑based web application that lets students query a textbook, PDFs, and images using an OpenAI‑powered conversational agent.  Uploaded material is chunked, embedded with FAISS, and retrieved with LangChain so the bot can guide the learner without revealing full solutions.

---

##  Project Structure

```
project/
│ preprocessing.py                # ⇐ main Flask application 
│ 2200-textbook.pdf     # default knowledge base (optional)
│ requirements.txt
│ uploads/              # runtime‑created (PDFs)
│ uploads/images/       # runtime‑created (images)
└─ templates/
   └─ index.html        # ⇐ chat frontend 
```

---

##  Installation

```bash
# 1. Clone & enter the folder
$ git clone <repo‑url> cs2200‑ai‑ta && cd cs2200‑ai‑ta

# 2. Create and activate a virtual‑env (recommended)
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Make a requirements.txt with the dependencies below
# and Install Python dependencies
$ pip install -r requirements.txt
```

### `Dependencies`

```text
Flask>=2.3
PyPDF2>=3.0.0
Pillow
pytesseract
faiss-cpu           # or faiss-gpu if you have CUDA
langchain           # core orchestration
langchain-community # vector stores, text splitters, etc.
langchain-openai    # OpenAI wrappers
openai              # OpenAI API client
python-dotenv       # optional, for .env files
Flask
PyPDF2
Pillow
pytesseract
faiss-cpu          # or faiss-gpu if you have CUDA
langchain          # pulls langchain-community & langchain-openai
openai             # OpenAI API
python-dotenv      # optional, for .env files
```

---

##  Running the app

```bash
$ python preprocessing.py
```

Then open [http://localhost:5001](http://localhost:5001) in a browser.

---

##  Using the UI

1. Ask a natural‑language question in the chat box.
2. *Optionally* attach images (JPEG/PNG/GIF); they’re OCR‑scanned and appended as context.
3. Use the **Upload Study Materials** drawer to drop PDFs. Press **Process Files** to ingest them into the vector store.
4. The TA responds with guidance – not full answers – per the prompt rules.


---

##  License

MIT License

---
