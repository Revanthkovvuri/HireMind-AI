# 🤖 HireMind AI - Intelligence-Driven ATS

**HireMind AI** is an enterprise-grade Applicant Tracking System (ATS) designed to eliminate recruitment bottlenecks and hiring bias. By combining local semantic embeddings with cloud-based LLM reasoning, it provides a fast, fair, and data-driven recruitment pipeline.

## 🌟 Key Features

1.  **📄 Bulk Resume Processing:** Rapidly upload and process dozens of resumes simultaneously.
2.  **🔍 Universal Parsing:** Full support for extracting and analyzing text from both **PDF** and **DOCX** formats.
3.  **🧠 Semantic Match Engine:** Moves beyond keyword matching to understand the deep contextual relevance between candidates and Job Descriptions.
4.  **📝 AI-Generated Candidate Summaries:** Leverages LLMs to generate concise, high-level snapshots of a candidate's professional profile.
5.  **⚖️ Explainable Scoring:** Provides a transparent breakdown of match percentages across Education, Experience, and Skills.
6.  **🏆 Intelligent Ranking:** Automatically ranks candidates based on a multi-dimensional semantic match score.
7.  **🕵️ Bias Masking:** Redacts PII (Names, Emails, Locations) and neutralizes pronouns to ensure a purely merit-based screening process.
8.  **📊 Diversity & Analytics Dashboard:** Real-time visualization of applicant distribution, score tiers, and hiring metrics.
9.  **🚀 Multi-Role Management:** Seamlessly handle multiple job openings with dedicated, filtered data and leaderboards for each role.
10. **🔄 Hiring Cycle Tracking:** End-to-end management of the recruitment pipeline—manually advance candidates through stages from *CV Screening* to *Technical Interviews* and *Offer Letters*.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom Glass-morphism UI)
* **NLP Engine:** spaCy (NER for Bias Masking)
* **Vector Embeddings:** SBERT (all-MiniLM-L6-v2)
* **Reasoning LLM:** Groq (Llama-3.3-70b-versatile)
* **Visualization:** Plotly Express

## 📦 Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/OptCamp/HiremindAi.git](https://github.com/OptCamp/HiremindAi.git)
    cd HiremindAi
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download NLP Models:**
    ```bash
    python -m spacy download en_core_web_sm
    ```
4.  **Configure Secrets:**
    Add your `GROQ_API_KEY` to `.streamlit/secrets.toml`.
5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---
**👨‍💻 Developer:** [Revanth Kovvuri](https://github.com/Revanthkovvuri)