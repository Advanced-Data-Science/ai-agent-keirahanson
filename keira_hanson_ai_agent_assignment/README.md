# **AI-Powered Rental Data Collection Agent** **Project Overview**

This project implements an AI-powered data collection agent that automatically gathers, processes, and documents rental data according to a Data Management Plan (DMP). The system was designed to demonstrate best practices in API usage, data management, and intelligent collection strategies.

The primary scenario chosen for this assignment is **Rental Prices Analysis**, focusing on rental property data in San Francisco. The agent interacts with the **RentCast API**  to collect rental property listings and metadata, process them into structured formats, and generate quality and metadata reports.

The agent is designed to:

* **Plan:** Read collection requirements from the DMP.  
* **Collect:** Gather data using APIs with rate limiting and adaptive strategies.  
* **Process:** Normalize, validate, and clean the data.  
* **Document:** Generate automated metadata, quality reports, and collection summaries.  
* **Adapt:** Adjust collection speed and strategy based on success rates and data quality.

This project demonstrates ethical and respectful data collection practices, code quality requirements, and the creation of reproducible workflows.

**Learning Objectives**  
By completing this project, I gained experience in:

* Understanding and implementing API interactions from scratch.  
* Applying Data Management Plan principles to practice.  
* Building an intelligent agent that adapts collection strategies dynamically.  
* Practicing ethical and respectful data collection with rate limiting and error handling.  
* Creating automated documentation and quality checks to ensure dataset usability.

## **Data Management Plan Summary** **Scenario:** Rental Prices Analysis

* **Objective:** Collect rental home data to analyze rent prices across varying proximity to high noise pollution areas.  
* **Data Sources:** RentCast Property Data API (mocked for this assignment).  
* **Data Types:** Address components, property types, HOA fees, tax amounts, sale transaction history (dates, prices, interest rates).  
* **Geographic Scope:** San Francisco.  
* **Time Range:** Past 5–10 years.

The DMP ensures data collection aligns with ethical, reproducible, and well-documented practices.

## **File Structure**

`your_name_ai_agent_assignment/`  
`├── README.md                    # Project overview, setup instructions, usage guide`  
`├── data_management_plan.pdf     # Mini DMP`  
`├── agent/`  
`│   ├── data_collection_agent.py # Main agent class`  
`│   ├── config.json              # API + collection config`  
`│   ├── requirements.txt         # Dependencies`  
`│   └── tests/`  
`│       └── test_agent.py        # Unit tests`  
`├── data/`  
`│   ├── raw/                     # Raw collected data`  
`│   ├── processed/               # Cleaned data`  
`│   └── metadata/                # Generated metadata`  
`├── logs/`  
`│   └── collection.log           # Execution logs`  
`├── reports/`  
`│   ├── quality_report.html      # Human-readable quality report`  
`│   └── collection_summary.pdf   # Final summary`  
`└── demo/`  
    `├── api_exercises.py         # API fundamentals exercises`  
    `└── demo_screenshots/        # Screenshots of the agent running`

---

## **Setup Instructions**

**Clone Repository**

 `git clone <repo-url>`  
`cd your_name_ai_agent_assignment`

1. **Create Virtual Environment (recommended)**

    `python3 -m venv venv`

`source venv/bin/activate   # macOS/Linux`  
`venv\Scripts\activate      # Windows`

2. **Install Dependencies**

    `pip install -r agent/requirements.txt`  
3. **Configure API Keys**  
   Create a `.env` file in the project root:

    `RENT_API_KEY=your_actual_key_here`  
   *   
   * Update `agent/config.json` with your API settings.

4. \! For this assignment, the agent defaults to **mock mode**, so it can run without a real API key.

## **Usage Guide**

### **Run the Agent:** From the `agent/` folder:

`python3 data_collection_agent.py`

### **Expected Outputs**

* **Raw Data:** Saved to `data/raw/collected.json`.  
* **Processed Data:** Saved to `data/processed/processed.json`.  
* **Logs:** Written to `logs/collection.log`.  
* **Reports:**

  * Metadata → `reports/dataset_metadata.json`  
  * Quality Report → `reports/quality_report.json` \+ `reports/quality_report.html`  
  * Collection Summary → `reports/collection_summary.json` \+ `reports/collection_summary.pdf`

### **Example Terminal Output**

`INFO - Starting Rent Data Collection Agent`  
`Stored listing 12345 in San Francisco at $2577`  
`- Raw data saved to data/raw/collected.json`  
`- Processed data saved to data/processed/processed.json`  
`INFO - Final report generated: reports/collection_summary.json`

### **Testing the Agent** From the `agent/` folder, run: `pytest tests/`

---

**Key Features**

* **Configuration Management:** Uses `config.json` and `.env` for API settings (no hardcoding).  
* **Adaptive Collection:** Dynamically adjusts delays if success rates drop.  
* **Respectful API Usage:** Respects rate limits and includes random jitter to avoid overloading APIs.  
* **Data Quality Assessment:** Evaluates completeness, accuracy, consistency, and timeliness.  
* **Documentation:** Automatically generates metadata, quality reports, and summaries.  
* **Error Handling:** Includes try/except with informative logging for failed API calls.

---

## **Lessons Learned**

* **APIs:** Learned how to send requests, handle parameters, parse JSON responses, and incorporate error handling.

* **Data Management:** Reinforced the importance of DMPs in structuring collection projects.

* **Agent Design:** Gained experience building an agent that not only collects data but *adapts* based on performance and data quality.

* **Documentation:** Discovered the value of automated metadata and quality reporting for reproducibility.


