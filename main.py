from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import json
import uvicorn
import qdrant_client
from groq import Groq
from qdrant_client.models import PointStruct
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()  

# Initialize FastAPI
app = FastAPI()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Initialize Groq Client (LLaMA-3)
groq_client = Groq(api_key=GROQ_API_KEY)  # Replace with your actual API key

# Initialize Qdrant Client (Cloud)
qdrant_client = qdrant_client.QdrantClient(
    url=QDRANT_URL,  # Replace with your Qdrant URL
    api_key=QDRANT_API_KEY # Replace with your Qdrant API key
)
collection_name = "documents"

# Replace with your Weather API key

class FarmInfo(BaseModel):
    id: int
    totalLandArea: float
    farmLocation: str
    latitude: str
    longitude: str
    deviceId: str
    soilType: str
    waterSource: str
    crop: str
    sowingDate: str
    currentGrowthStage: str
    idealGrowingConditions: str
    pastPestIssues: bool
    preferredMoistureLevel: str
    irrigationType: str
    waterAvailabilityStatus: str
    fertilizersUsed: List[str]

class SensorData(BaseModel):
    id: int
    deviceId: str
    nitrogen: float
    potassium: float
    phosphorus: float
    conductivity: float
    pH: float
    humidity: float
    temperature: float
    userId: int
    createdAt: str  

class Tasks(BaseModel):
    id: int
    taskTitle: str
    taskDescription: str
    taskSeverity: str
    taskStatus: str
    deviceId: str
    deadliestDeadline: str
    createdAt: str

class Advisories(BaseModel):
    title: str
    precaution: str
    risk_factors: str
    recommended_action: str
    createdAt: str

class FarmRequest(BaseModel):
    farm_info: FarmInfo
    npk_data: List[SensorData]

class FarmRequestTasks(BaseModel):
    farm_info: FarmInfo
    npk_data: List[SensorData]
    advisories: List[Advisories]

class FarmRequestUpdatedTasks(BaseModel):
    tasks: List[Tasks]
    farm_info: FarmInfo
    npk_data: List[SensorData]
    advisories: List[Advisories]

# Load the correct embedding model (384-dimension)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-D vectors

def truncate_text(text, max_chars=4500):
    """Truncate text to the specified character limit."""
    if isinstance(text, str):
        return text[:max_chars] + "..." if len(text) > max_chars else text
    return text  # If it's not a string, return as is

def search_qdrant_advisories(npk_data: List[dict], crop: str, soil_type: str):
    try:
        # Convert NPK data into a structured sentence
        npk_data_texts = []
        for npk_entry in npk_data:
            npk_text = (
                f"Nitrogen: {npk_entry['nitrogen']}, "
                f"Phosphorus: {npk_entry['phosphorus']}, "
                f"Potassium: {npk_entry['potassium']}, "
                f"Soil Moisture: {npk_entry['humidity']}, "
                f"Soil Temperature: {npk_entry['temperature']}, "
                f"Conductivity: {npk_entry['conductivity']}, "
                f"pH Level: {npk_entry['pH']}"
            )
            npk_data_texts.append(npk_text)

        # Join all NPK entries into a single text
        npk_combined_text = " | ".join(npk_data_texts)

        # Formulate search query as a natural text prompt
        search_text = (
            f"Crop Type: {crop}. Soil Type: {soil_type}. "
            f"Soil Nutrient Levels: {npk_combined_text}."
        )

        # Generate the correct 384-D embedding
        query_vector = embedding_model.encode(search_text).tolist()

        # Perform the Qdrant vector search
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,  # ✅ Correct 384-D vector
            limit=1,
        )

        json_results = [
            {
                "id": result.id,
                "score": result.score,
                "payload": {
                    key: truncate_text(value) if isinstance(value, str) else value  # Apply truncation only on text values
                    for key, value in result.payload.items()
                }
            }
            for result in search_results
        ]

        return json_results

    except Exception as e:
        return []

def fetch_weather_forecast(latitude: float, longitude: float):
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}/next3days?key={WEATHER_API_KEY}&contentType=json&include=days"
        response = requests.get(url)
        response.raise_for_status()
        
        weather_data = response.json().get("days", [])

        return weather_data
    except requests.RequestException as e:
        return None

def generate_advisories(farm_request: dict, weather_forecast: list, qdrant_advisories: list):
    """
    Generate structured advisories using LLM with Qdrant search results as additional context.
    """
    qdrant_context = json.dumps(qdrant_advisories) if qdrant_advisories else "No similar past advisories found."
    
    prompt = f"""
### 🌱 **Agricultural Advisory Generation** 🌱  
You are an **elite precision farming expert** specializing in **crop-specific, growth-stage-aware, and data-driven advisories.**  

🔹 **Your Task:**  
Using the provided **farm data, soil conditions, weather forecast, past advisories, and nutrient analysis**, generate **highly structured, expert-level advisories** in JSON format.  

---

### 📌 **Farm & Environmental Data (Strictly Consider All Factors)**
- **Crop Type:** {farm_request['farm_info']['crop']}  
- Farm info {farm_request['farm_info']}
- **Current Growth Stage:** {farm_request['farm_info']['currentGrowthStage']}  
- **Soil Type:** {farm_request['farm_info']['soilType']}  
- **Fertilizers Applied:** {farm_request['farm_info']['fertilizersUsed']}  
- **Weather Forecast (Next 7 Days):** {json.dumps(weather_forecast, indent=2)}  
- **Soil Nutrient Profile (NPK, pH, Conductivity, Moisture, etc.):** {json.dumps(farm_request['npk_data'], indent=2)}  
- **Past Relevant Advisories from Vector DB:** {qdrant_context}  

Units of NKP are in mg/L, temperature in °C, and humidity in %.
---

### 🚀 **Rules for Generating Advisories**  
🔹 **Ultra-Specific Insights** → Every advisory must be **data-driven** and directly relevant to the crop, soil, and weather conditions.  
🔹 **Deep Analysis Required** → AI must **correlate soil health, growth stage, and forecasted weather trends** to determine **optimal actions.**  
🔹 **Preventative & Corrective Approach** → If conditions are suboptimal, provide **clear, actionable measures** to optimize yield.  
🔹 **No Generic Advice** → Responses must be **scientific, precise, and field-applicable.**  
🔹 **Risk-Based Reasoning** → Highlight **consequences of inaction** and potential **threats to yield or soil health.**  
🔹 **Urgency Classification Required** → If the situation is critical (e.g., severe nutrient deficiency, extreme weather risks), tag it as **"High-Priority Advisory."**  
🔹 **Adhere to JSON Output Format** → AI must return **only structured data**—no explanations, just actionable content.  
🔹 **Limit to only most critical Advisories** → If more advisories are necessary, merge related recommendations into a single advisory with multiple key points.  

---

### ⚠️ **Critical Factors That Must Trigger an Advisory**
✔ **Extreme Weather Alerts:** Drought, frost, heavy rain, heatwaves.  
✔ **Soil Deficiencies:** Low or zero levels of NPK, imbalanced pH.  
✔ **Pest/Disease Risk:** If environmental conditions favor disease outbreaks.  
✔ **Water Management Issues:** Overwatering or drought risks based on soil moisture and weather data.  
✔ **Fertilization Adjustments:** Detect trends in nutrient depletion and recommend corrective action.  

---

### 🔹 **Strict JSON Output Format (No Extra Words)**
```json
[
  {{
    "title": "<Short, highly relevant advisory title>",
    "precaution": "<Precise precautionary steps based on current conditions>",
    "risk_factors": "<Potential threats if this advisory is ignored>",
    "recommended_action": "<Exact scientific action the farmer must take>",
  }},
  ...
]

Ensure each advisory is **relevant** to the crop, growth stage, and upcoming weather conditions.
Generate most critical advisories, merging related points where necessary.
generate max 3 advisories daily and no other text other than these tasks , return only json data nothing else  return only json data nothing else not a single line
"""



    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type":"json_object"}
        )

        if not response.choices or not response.choices[0].message.content:
            return {"advisories": []}

        llm_output = response.choices[0].message.content.strip()

        try:
            return {"advisories": llm_output}
        except json.JSONDecodeError:
            return {"advisories": []}

    except Exception as e:
        return {"advisories": []}

def generate_tasks_func(farm_request: dict, weather_forecast: list, qdrant_advisories: list):
    """
    Generate structured advisories using LLM with Qdrant search results as additional context.
    """
    qdrant_context = json.dumps(qdrant_advisories) if qdrant_advisories else "No similar past advisories found."
    prompt = f"""
    ### 🚜 **Precision Agriculture Task Generator** 🌱  
You are an **elite agricultural task management AI**, specializing in **data-driven, urgency-aware, and precision-farming-focused task generation**.  

🔹 **Your Task:**  
Using **real-time farm data, environmental conditions, advisories, and past tasks**, generate **highly structured, quantitative, and fully actionable farming tasks** in JSON format. Each task must have a **priority level (High, Medium, Low), an impact-based deadline, and an explicit action** with precise quantities and methods suited to the farm's conditions.  

---

### 📌 **🌾 Farm & Advisory Context (Strictly Consider All Factors)**
- **Crop Type:** {farm_request['farm_info']['crop']}  
- **Current Growth Stage:** {farm_request['farm_info']['currentGrowthStage']}  
- **Soil Type:** {farm_request['farm_info']['soilType']}  
 Farm info {farm_request['farm_info']}
- **Fertilizers Applied:** {farm_request['farm_info']['fertilizersUsed']}  
- **Weather Forecast (Next 7 Days):** {json.dumps(weather_forecast, indent=2)}  
- **Soil Nutrient Profile (NPK, pH, Conductivity, Moisture, etc.):** {json.dumps(farm_request['npk_data'], indent=2)}  
- **Irrigation System:** {farm_request['farm_info']['irrigationType']}  
- **Recent Advisories (Last 48 Hours):** {json.dumps(farm_request['advisories'], indent=2)}  
Past Relevant Advisories from Vector DB:** {qdrant_context}  
📌 *Units: NPK in mg/L, temperature in °C, humidity in %.*

---

### 🔥 **Task Generation Rules**
🔹 **Data-Driven & Ultra-Specific** → Every task must be derived from **real-time farm data and trends** with **exact values & actions**.  
🔹 **Severity Classification** → Each task must be tagged as:  
  - **HIGH:** Requires **immediate action** (e.g., extreme weather protection, severe nutrient deficiency).  
  - **MEDIUM:** Important but **can be delayed slightly** (e.g., moderate irrigation adjustment, minor pest risk).  
  - **LOW:** Routine optimizations that **improve yield over time** (e.g., gradual soil enhancement).  
🔹 **Deadline-Driven** → Define **deadliest deadline**, the **absolute latest** time before the task must be completed to prevent yield loss.  
🔹 **Avoid Redundancy** → If a task was recently assigned, **update it instead of duplicating** unless urgency has changed.  
🔹 **Actionable & Quantitative** → Tasks must **specify** amounts, methods, and equipment to use.  
🔹 **Ultra-Specific & Impact-Based** → Every task must be **directly derived from farm data, advisories, and environmental trends.**  
🔹 **Actionable, No Fluff** → No vague tasks. Every task must include **what, why, and how.**  


---

### ⚠️ **Conditions That Must Trigger Tasks**
✔ **Extreme Weather Alerts:** Activate countermeasures for drought, frost, storms, and heatwaves based on forecast.  
✔ **Soil Moisture & Irrigation:** If moisture drops below threshold for that crop, initiate irrigation **specific to system type (drip, flood, sprinkler, etc.)**.  
✔ **Nutrient Deficiencies:** Apply the **exact quantity of fertilizer** required per hectare to correct soil imbalance.  
✔ **Pest & Disease Risk:** Deploy **recommended pesticide/fungicide with proper dosage** based on risk level.  
✔ **Soil Health Optimization:** Adjust **pH or conductivity** using specific soil amendments.  
✔ **Growth-Stage-Based Actions:** Ensure **stage-appropriate** interventions (e.g., nitrogen boost for vegetative stage).  
✔ **Extreme Weather Alerts:** Prepare farm for drought, frost, heatwaves, storms.  
✔ **Pest & Disease Risk:** Take **preventative actions** based on environmental risk factors.  
✔ **Water & Irrigation Management:** Adjust watering schedules **based on soil moisture and forecast**.  
✔ **Fertilization & Soil Health:** Identify **upcoming nutrient needs** based on trends.  

---

### 📜 **Strict JSON Output Format**
Ensure each task is **relevant**, **quantitative**, and **specific to the farm’s exact conditions** with **no vague descriptions**. Each task should be clearly different from advisories and contain **direct actions** that must be executed.  


### 📜 **Strict JSON Output Format (No Extra Words)**
```json
[
  {{
    "taskTitle": "<very Concise, high-impact task title>",
    "taskDescription": "<Detailed explanation of the task, why it's necessary, and the specific action required>",
    "taskSeverity": "<HIGH / MEDIUM / LOW>",
    "deadliestDeadline": "<YYYY-MM-DDTHH:MM:SSZ - absolute latest time before the task becomes critical>",
  }},
  ...
]
generate max 2 tasks daily and no other text other than these tasks, return only json data nothing else  return only json data nothing else not a single line
"""
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type":"json_object"}
        )

        if not response.choices or not response.choices[0].message.content:
            return {"tasks": []}

        llm_output = response.choices[0].message.content.strip()

        try:
            return {"tasks": llm_output}
        except json.JSONDecodeError:
            return {"tasks": []}

    except Exception as e:
        return {"tasks": []}

def update_tasks(farm_request: dict, weather_forecast: list):
    """
    Generate structured advisories using LLM with Qdrant search results as additional context.
    """
    
    prompt = f"""

### 🚜 **Precision Agriculture Task Updater** 🌱  
You are an **elite agricultural automation AI**, specializing in **real-time task optimization** for **smart farming devices**. Your role is to analyze active agricultural tasks and ensure they remain relevant, efficient, and aligned with real-time farm conditions. The goal is to eliminate redundant or outdated tasks while making precise updates to those that need modification.  

---  

🔹 **Your Objective:**  
You will process **real-time farm data, environmental conditions, advisories, and past tasks** to:  
✔ **Modify only necessary tasks** based on changing farm conditions.  
✔ **Cancel redundant or obsolete tasks** (e.g., outdated duplicates, unnecessary actions).  
✔ **Ensure logical consistency**—modifications should align with environmental realities.  
✔ **Provide a clear rationale** for each change using a `notes` field.  

**💡 DO NOT update tasks that don’t require modifications. Only return necessary changes.**  

---  

## 📌 **Data Inputs for Task Optimization**  
These data sources will guide your decision-making:  

### 🟢 **Farm Details**  
- **Crop Type:** {farm_request['farm_info']['crop']}  
- **Current Growth Stage:** {farm_request['farm_info']['currentGrowthStage']}  
- **Soil Type:** {farm_request['farm_info']['soilType']}  
- **Fertilizers Applied:** {farm_request['farm_info']['fertilizersUsed']}  

### 🌦 **Environmental Conditions**  
- **Weather Forecast (Next 7 Days):** {json.dumps(weather_forecast, indent=2)}  
- **Farm Sensor Data (NPK, pH, Moisture, Temperature, etc.):** {json.dumps(farm_request['npk_data'], indent=2)}  
- **Active Farm Advisories & Climate Alerts:** {json.dumps(farm_request['advisories'], indent=2)}  

### 📋 **Pending Farm Tasks**  
- **Unfinished Tasks:** {json.dumps(farm_request['tasks'], indent=2)}  

📌 *Units: NPK in mg/L, temperature in °C, moisture in %.*  

---  

## 🔥 **Task Optimization Rules**  

### 🔴 **Task Cancellation (`taskStatus: "Cancelled"`)**  
Cancel tasks only if:  
- **They are redundant** *(e.g., an outdated duplicate exists).*  
- **They are no longer necessary due to environmental conditions** *(e.g., scheduled irrigation is no longer needed due to recent rainfall).*  
- **A newer task supersedes the current one**, and the older task has an earlier `createdAt` timestamp.  
🚨 **If two tasks are similar, only cancel the older one and retain the most recent task.**  

### 🟠 **Task Severity Updates (`taskSeverity`)**  
Modify severity only when:  
- **An urgent condition develops**, requiring immediate attention *(e.g., critically low soil moisture or extreme weather conditions).*  
- **A previously critical condition has stabilized**, allowing for a lower priority.  

### 🟡 **Task Description Updates (`taskDescription`)**  
Modify descriptions only if:  
- **The task execution method needs adjustment** due to environmental changes *(e.g., switching from nitrogen-based to phosphorus-based fertilization).*  
- **The task requires clearer execution details**, while still being valid.  

### 🔵 **Deadline Adjustments (`deadliestDeadline`)**  
Modify deadlines only if:  
- **The urgency of the task has changed** due to real-time conditions.  
- **The task should be rescheduled to a more optimal time** based on farm priorities.  

---  

## ⚠️ **Strict Task Modification Guidelines**  
✔ **Modify only tasks that require updates. Leave others unchanged.**  
✔ **Each task should have only one modification at a time (e.g., changing severity OR canceling, not both).**  
✔ **Each update must include a `notes` field explaining the reason for modification.**  
✔ **If duplicate tasks exist, remove only the older one, not both.**  

---  

## 📜 **Strict JSON Output Format**  
Return an **array of objects**, where each object contains only the modified fields and a `notes` field:  

```json
[
  {{
    "id": "<taskid>",
    "<modified_field>": "<new_value>",
    "notes": "<brief explanation of the modification>"
  }},
  .....
]
Return only json array nothing else no optimization rules at the bottom, also new_value must be in Capitalize format , return only json data nothing else not a single line
"""

    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type":"json_object"}
        )

        if not response.choices or not response.choices[0].message.content:
            return {"updatedTasks": []}

        llm_output = response.choices[0].message.content.strip()

        try:
            return {"updatedTasks": llm_output}
        except json.JSONDecodeError:
            return {"updatedTasks": []}

    except Exception as e:
        return {"updatedTasks": []}

def summary_report(farm_request: dict, weather_forecast: list):
    """
    Generate structured advisories using LLM with Qdrant search results as additional context.
    """
    prompt = f"""
### 📊 **Weekly Farm Health, Yield Forecast & Sustainability Report** 🌱  

You are an **AI agronomist and precision farming specialist**, tasked with analyzing the farm’s **weekly performance** and providing an **expert-level strategic report**. Your goal is to assess **soil health, environmental conditions, emerging risks, and yield projections** to ensure optimal farm productivity and sustainability.  

---  

🔹 **Your Objective:**  
Using the provided farm advisories, environmental data, and operational insights:  
✔ **Evaluate overall farm health**, including soil quality, crop conditions, and resource utilization.  
✔ **Identify inefficiencies or risk factors** affecting crop growth and sustainability.  
✔ **Forecast future yield trends** based on soil, climate, and resource data.  
✔ **Provide data-driven recommendations** to optimize short-term productivity and long-term farm resilience.  

---  

## 📌 **Input Data for Analysis**  
You will receive:  
- **Farm Advisories & Environmental Trends:** {json.dumps(farm_request['advisories'], indent=2)}  
- **Soil & Crop Health Data (NPK, Moisture, pH, EC, Temperature):** {json.dumps(farm_request['npk_data'], indent=2)}  
- **Weather Conditions & Seasonal Patterns:** (if applicable)  

📌 *Units: NPK in mg/L, pH in standard units, EC in dS/m, temperature in °C, soil moisture in %.*  

---  

## 🔍 **Your Analysis Must Include:**  

### 🌿 **1. Farm Condition & Soil Health Assessment**  
- **Soil nutrient balance trends** (Nitrogen, Phosphorus, Potassium, pH stability).  
- Detection of **soil deficiencies or excesses** impacting crop growth.  
- **Environmental factors** affecting soil quality (e.g., extreme temperatures, excess rainfall, salinity).  
- **Early warning indicators** of potential stress, disease vulnerability, or nutrient imbalances.  

### 📉 **2. Risks, Inefficiencies & Crop Productivity Challenges**  
- Identification of **emerging threats** (pest outbreaks, nutrient leaching, irrigation mismanagement).  
- **Yield-impacting trends** based on recent soil and environmental shifts.  
- **Efficiency gaps** in fertilization, irrigation, or resource utilization.  
- **How current farm conditions could impact next season's harvest** and necessary mitigations.  

### 🚀 **3. Strategic Yield Optimization & Sustainability Plan**  
- **Immediate Interventions:** Critical actions to mitigate potential yield losses.  
- **Soil & Resource Optimization:** Adjustments in fertilization, irrigation, or crop rotation strategies.  
- **Yield Forecast:** Projected productivity trends and expected impact of current farm conditions.  
- **Climate-Resilient Strategies:** Proactive measures to sustain long-term soil health and resource efficiency.  

---  

⚠️ **Strict Output Format:**  
Return a structured **JSON response** in the following format:  

```json
[
  { {
      "farm_health": [
        "Soil condition insight 1"
      ],
      "risk_analysis": [
        "Identified risk 1"
      ],
      "yield_forecast": [
        "Projected yield trend 1"
      ]
    }
  }
]

📌 **Return only the JSON array**—no additional text, explanations, or optimization rules at the bottom. Also, ensure `new_value` is always in **Capitalized Format**.  
 return only json data nothing else not a single line
"""

    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type":"json_object"}
        )

        if not response.choices or not response.choices[0].message.content:
            return {"weeklySummary": []}

        llm_output = response.choices[0].message.content.strip()

        try:
            return {"weeklySummary": llm_output}
        except json.JSONDecodeError:
            return {"weeklySummary": []}

    except Exception as e:
        return {"weeklySummary": []}

@app.post("/events")
async def generate_farm_advisory(request: FarmRequest):
    try:
        farm_data = request.dict()
        latitude, longitude = farm_data["farm_info"]["latitude"], farm_data["farm_info"]["longitude"]

        # Fetch weather forecast
        weather_forecast = fetch_weather_forecast(latitude, longitude)
        if not weather_forecast:
            raise HTTPException(status_code=500, detail="Weather API fetch failed.")

        # Fetch past relevant advisories from Qdrant
        qdrant_advisories = search_qdrant_advisories(
            npk_data=farm_data["npk_data"],
            crop=farm_data["farm_info"]["crop"],
            soil_type=farm_data["farm_info"]["soilType"]
        )

        # Generate advisories using LLM with Qdrant context
        advisories = generate_advisories(farm_data, weather_forecast, qdrant_advisories)

        return advisories

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-tasks")
async def generate_tasks(request: FarmRequestTasks):
    try:
        farm_data = request.dict()
        latitude, longitude = farm_data["farm_info"]["latitude"], farm_data["farm_info"]["longitude"]

        # Fetch weather forecast
        weather_forecast = fetch_weather_forecast(latitude, longitude)
        if not weather_forecast:
            raise HTTPException(status_code=500, detail="Weather API fetch failed.")

        # Fetch past relevant advisories from Qdrant
        qdrant_advisories = search_qdrant_advisories(
            npk_data=farm_data["npk_data"],
            crop=farm_data["farm_info"]["crop"],
            soil_type=farm_data["farm_info"]["soilType"]
        )

        # Generate advisories using LLM with Qdrant context
        advisories = generate_tasks_func(farm_data, weather_forecast, qdrant_advisories)

        return advisories

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/updated-tasks")
async def generate_updated_tasks(request: FarmRequestUpdatedTasks):
    try:
        farm_data = request.dict()
        latitude, longitude = farm_data["farm_info"]["latitude"], farm_data["farm_info"]["longitude"]

        # Fetch weather forecast
        weather_forecast = fetch_weather_forecast(latitude, longitude)
        if not weather_forecast:
            raise HTTPException(status_code=500, detail="Weather API fetch failed.")

       

        # Generate advisories using LLM with Qdrant context
        advisories = update_tasks(farm_data, weather_forecast)

        return advisories

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
async def generate_updated_tasks(request: FarmRequestUpdatedTasks):
    try:
        farm_data = request.dict()
        latitude, longitude = farm_data["farm_info"]["latitude"], farm_data["farm_info"]["longitude"]

        # Fetch weather forecast
        weather_forecast = fetch_weather_forecast(latitude, longitude)
        if not weather_forecast:
            raise HTTPException(status_code=500, detail="Weather API fetch failed.")

       

        # Generate advisories using LLM with Qdrant context
        advisories = summary_report(farm_data, weather_forecast)

        return advisories

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)