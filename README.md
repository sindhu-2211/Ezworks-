# EZWorks Document Intelligence App 📄🤖

A Streamlit-powered intelligent document interface. This application allows users to upload documents and either:

- 🧠 Ask questions freely (Ask Anything Mode)
- 🎯 Get quizzed by the system and receive evaluations (Challenge Me Mode)

---

## 🏗 Architecture

```text
          ┌────────────────────┐
          │  Document Upload   │
          └────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │   Text Extraction   │
         │  (PDF/DOCX/TXT etc) │
         └────────┬────────────┘
                  ▼
       ┌───────────────────────────┐
       │     Summarization         │
       │   (via LLM API like GPT)  │
       └──────┬──────────────┬─────┘
              │              │
              ▼              ▼
   ┌────────────────┐  ┌──────────────────────┐
   │ Ask Anything    │  │  Challenge Me Mode   │
   │ Mode            │  │                      │
   │ - QA interface  │  │ - LLM generates Qs   │
   │ - LLM answers   │  │ - User answers       │
   │   from context  │  │ - LLM evaluates      │
   └────────────────┘  └──────────────────────┘

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.10+
- Git
- OpenAI (or Gemini/Cohere) API Key

---

### 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ezworks-.git
   cd ezworks-
Create a Virtual Environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # For Windows: venv\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set Environment Variables

Option 1: Create a .env file in the root directory:

ini
Copy
Edit
OPENAI_API_KEY=your-api-key-here
Option 2: Set it inside your script (for example, in app.py):

python
Copy
Edit
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
Run the App

bash
Copy
Edit
streamlit run app.py
