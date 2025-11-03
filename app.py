from flask import Flask, jsonify, render_template
import requests

app = Flask(__name__)

# ----------------- CONFIG -----------------
FHIR_BASE = "https://hapi.fhir.org/baseR4"
HF_API_URL = "https://router.huggingface.co/hf-inference"
HF_HEADERS = {}  # You can leave this empty for public/free access


# ----------------- FUNCTIONS -----------------
def fetch_lab_results():
    """Retrieve Observation resources (lab results)"""
    url = f"{FHIR_BASE}/Observation?code=2339-0&_count=3"  # Glucose [mg/dL]
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("entry", [])


def sanitize_observation(obs):
    """Simplify Observation data"""
    return {
        "test_name": obs.get("code", {}).get("coding", [{}])[0].get("display", "Unknown test"),
        "value": obs.get("valueQuantity", {}).get("value", "N/A"),
        "unit": obs.get("valueQuantity", {}).get("unit", ""),
        "status": obs.get("status", ""),
        "effectiveDateTime": obs.get("effectiveDateTime", "")
    }


def summarize_lab(obs):
    """Use Hugging Face model to summarize a lab result"""
    test = obs.get("code", {}).get("coding", [{}])[0].get("display", "Unknown test")
    value = obs.get("valueQuantity", {}).get("value", "")
    unit = obs.get("valueQuantity", {}).get("unit", "")
    prompt = f"Explain in plain language what it means if {test} is {value}{unit}."

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 80}}
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        result = response.json()

        # Different HF models return text differently
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"Model error: {result['error']}"
        else:
            return str(result)
    except Exception as e:
        return f"Error generating summary: {e}"


# ----------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/labs", methods=["GET"])
def get_labs():
    try:
        labs = fetch_lab_results()
        if not labs:
            return jsonify({"error": "No lab results found."}), 404

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
