from flask import Flask, jsonify, render_template, request
import requests
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

# ----------------- CONFIG -----------------
FHIR_BASE = "https://hapi.fhir.org/baseR4"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------- FUNCTIONS -----------------
def fetch_lab_results(patient_id):
    """
    Retrieve all Observation resources (lab results) for a specific patient.
    """
    url = f"{FHIR_BASE}/Observation?patient={patient_id}&_count=20"
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("entry", [])


def sanitize_observation(obs):
    """
    Simplify Observation data into a readable format.
    """
    code_info = obs.get("code", {}).get("coding", [{}])[0]
    return {
        "test_name": code_info.get("display", "Unknown test"),
        "code": code_info.get("code", ""),
        "value": obs.get("valueQuantity", {}).get("value", "N/A"),
        "unit": obs.get("valueQuantity", {}).get("unit", ""),
        "status": obs.get("status", ""),
        "effectiveDateTime": obs.get("effectiveDateTime", ""),
    }


def summarize_lab(obs):
    """
    Use ChatGPT (OpenAI API) to generate a natural-language summary of the lab result.
    """
    test = obs.get("code", {}).get("coding", [{}])[0].get("display", "Unknown test")
    value = obs.get("valueQuantity", {}).get("value", "")
    unit = obs.get("valueQuantity", {}).get("unit", "")
    prompt = f"Explain in simple terms what it means if {test} is {value}{unit}. Keep it concise and helpful for a patient."

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant explaining lab test results to a patient in plain, kind, and accurate language."
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating summary: {e}"


# ----------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/labs", methods=["GET"])
def get_labs():
    """
    Endpoint: /labs?patient_id=12345
    Fetches all lab results for a given patient and summarizes them.
    """
    patient_id = request.args.get("patient_id", "").strip()
    if not patient_id:
        return jsonify({"error": "Missing required parameter: patient_id"}), 400

    try:
        labs = fetch_lab_results(patient_id)
        if not labs:
            return jsonify({"error": f"No lab results found for patient {patient_id}."}), 404

        summaries = []
        for entry in labs:
            obs = entry.get("resource", {})
            lab = sanitize_observation(obs)
            lab["summary"] = summarize_lab(obs)
            summaries.append(lab)

        return jsonify(summaries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
