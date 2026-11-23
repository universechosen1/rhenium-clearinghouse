import json
import os
import re
from openai import APIError, OpenAI
import streamlit as st

# 1. Initialize Inception Mercury Client (Reads from Streamlit Secrets or Env Vars)
INCEPTION_API_KEY = st.secrets.get(
    "INCEPTION_API_KEY", os.getenv("INCEPTION_API_KEY", "")
)
INCEPTION_BASE_URL = st.secrets.get(
    "INCEPTION_BASE_URL",
    os.getenv("INCEPTION_BASE_URL", "https://inceptionlabs.ai"),
)
MERCURY_MODEL_SLUG = st.secrets.get(
    "MERCURY_MODEL_SLUG", os.getenv("MERCURY_MODEL_SLUG", "mercury-3")
)

client = OpenAI(
    api_key=INCEPTION_API_KEY or "MISSING_KEY",
    base_url=INCEPTION_BASE_URL,
)

# 2. System Prompt & JSON Schema
SYSTEM_PROMPT = """You are the core extraction engine for an industrial Rhenium (Re) and Platinum-Group Metal (PGM) Catalyst & Superalloy Clearinghouse. Users will interact via chat by pasting raw ERP text, lab assay snippets, or typing informal descriptions of spent catalysts and turbine superalloys.

Your ONLY job is to parse the user's message into strict, valid JSON. Do not output markdown blocks, explanations, or conversational text outside the JSON object.

Rules for Extraction:
1. Classify `material_type` as: "Spent_Reforming_Catalyst", "Superalloy_Scrap", "APR_or_Metal_Powder", or "Unknown".
2. Identify `grade_name` (e.g., "Pt-Re Alumina Catalyst", "CMSX-4", "Rene N5", "Rene N6", "PWA 1484", "Inconel 718").
3. Convert `net_weight_kg` to float kilograms (1 metric ton = 1000 kg; 1 lb = 0.453592 kg). If missing or <= 0, set to null.
4. Extract `re_wt_percent` and `pt_wt_percent` as floats (e.g., 0.35% = 0.35; 3000 ppm = 0.30).
   - Fallback Rule: If the user specifies a known superalloy grade without wt%, apply nominal industry Re content and set `is_estimated_grade` to true:
     * CMSX-4 = 3.0% Re
     * Rene N5 = 3.0% Re
     * PWA 1484 = 3.0% Re
     * Rene N6 = 6.0% Re
5. Extract `contaminants`: `sulfur_pct` (float), `carbon_pct` (float), `moisture_pct` (float), and `arsenic_ppm` (float). Set each to 0.0 if unmentioned.
6. Check `missing_required_fields`: Include "net_weight_kg" if null; include "grade_or_re_content" if both `re_wt_percent` and `pt_wt_percent` are null. If valid, return an empty array [].
7. Generate `next_chat_prompt`: If `missing_required_fields` is not empty, write one concise, technical question asking the user for the missing info. If complete, set to "READY_FOR_VALUATION".

Output strictly in this JSON schema:
{
  "material_type": "string",
  "grade_name": "string_or_null",
  "net_weight_kg": "float_or_null",
  "re_wt_percent": "float_or_null",
  "pt_wt_percent": "float_or_null",
  "is_estimated_grade": true,
  "contaminants": {
    "sulfur_pct": 0.0,
    "carbon_pct": 0.0,
    "moisture_pct": 0.0,
    "arsenic_ppm": 0.0
  },
  "missing_required_fields": [],
  "next_chat_prompt": "string"
}"""

# 3. Reference Pricing & Clearinghouse Parameters
RE_SPOT_USD_PER_KG = 2750.00
PT_SPOT_USD_PER_KG = 32500.00
CLEARINGHOUSE_FEE_PCT = 0.04


def _to_float(val, default=None):
  if val is None or isinstance(val, bool):
    return default
  try:
    return float(val)
  except (TypeError, ValueError):
    return default


def parse_and_validate(raw_content: str) -> dict:
  """Strips markdown fences, parses JSON, and enforces schema defaults."""
  cleaned = re.sub(
      r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.IGNORECASE
  )
  raw = json.loads(cleaned)
  if not isinstance(raw, dict):
    raise ValueError("Model response must be a JSON object.")

  raw_contam = (
      raw.get("contaminants")
      if isinstance(raw.get("contaminants"), dict)
      else {}
  )
  contaminants = {
      "sulfur_pct": _to_float(raw_contam.get("sulfur_pct"), 0.0),
      "carbon_pct": _to_float(raw_contam.get("carbon_pct"), 0.0),
      "moisture_pct": _to_float(raw_contam.get("moisture_pct"), 0.0),
      "arsenic_ppm": _to_float(raw_contam.get("arsenic_ppm"), 0.0),
  }

  weight = _to_float(raw.get("net_weight_kg"), None)
  re_wt = _to_float(raw.get("re_wt_percent"), None)
  pt_wt = _to_float(raw.get("pt_wt_percent"), None)

  missing = (
      list(raw.get("missing_required_fields"))
      if isinstance(raw.get("missing_required_fields"), list)
      else []
  )
  if (weight is None or weight <= 0) and "net_weight_kg" not in missing:
    missing.append("net_weight_kg")
  if (
      (re_wt is None or re_wt <= 0)
      and (pt_wt is None or pt_wt <= 0)
      and "grade_or_re_content" not in missing
  ):
    missing.append("grade_or_re_content")

  return {
      "material_type": str(raw.get("material_type") or "Unknown"),
      "grade_name": raw.get("grade_name") or "Unspecified Grade",
      "net_weight_kg": weight,
      "re_wt_percent": re_wt,
      "pt_wt_percent": pt_wt,
      "is_estimated_grade": bool(raw.get("is_estimated_grade", False)),
      "contaminants": contaminants,
      "missing_required_fields": missing,
      "next_chat_prompt": str(
          raw.get("next_chat_prompt")
          or "Please provide the net weight (kg/MT) and alloy grade or Re/Pt"
          " percentage."
      ),
  }


def calculate_valuation(data: dict) -> dict:
  """Deterministic valuation engine with strict input validation."""
  weight = data.get("net_weight_kg")
  if weight is None or not isinstance(weight, (int, float)) or weight <= 0:
    raise ValueError(
        "Valid positive net_weight_kg is required before running valuation."
    )

  re_wt = data.get("re_wt_percent")
  pt_wt = data.get("pt_wt_percent")
  if (re_wt is None or re_wt <= 0) and (pt_wt is None or pt_wt <= 0):
    raise ValueError(
        "At least one payable metal (Re wt% or Pt wt%) must be greater than"
        " zero."
    )

  re_pct = (re_wt or 0.0) / 100.0
  pt_pct = (pt_wt or 0.0) / 100.0
  mat_type = data.get("material_type", "Unknown")

  if mat_type == "Spent_Reforming_Catalyst":
    re_recovery, pt_recovery = 0.88, 0.95
  elif mat_type == "Superalloy_Scrap":
    re_recovery, pt_recovery = 0.85, 0.90
  else:
    re_recovery, pt_recovery = 0.90, 0.95

  gross_re_kg = weight * re_pct
  gross_pt_kg = weight * pt_pct

  payable_re_usd = gross_re_kg * RE_SPOT_USD_PER_KG * re_recovery
  payable_pt_usd = gross_pt_kg * PT_SPOT_USD_PER_KG * pt_recovery
  gross_payout = payable_re_usd + payable_pt_usd

  contaminants = data.get("contaminants") or {}
  penalty_usd = 0.0
  penalty_notes = []

  if contaminants.get("sulfur_pct", 0.0) > 2.0:
    penalty_usd += weight * 0.15
    penalty_notes.append("High Sulfur (>2.0%)")
  if contaminants.get("moisture_pct", 0.0) > 5.0:
    penalty_usd += gross_payout * 0.02
    penalty_notes.append("High Moisture (>5.0%)")
  if contaminants.get("arsenic_ppm", 0.0) > 50.0:
    penalty_usd += weight * 0.25
    penalty_notes.append("Arsenic Penalty (>50 ppm)")

  net_before_fee = max(0.0, gross_payout - penalty_usd)
  clearing_fee_usd = net_before_fee * CLEARINGHOUSE_FEE_PCT
  final_seller_quote = net_before_fee - clearing_fee_usd

  return {
      "gross_re_kg": round(gross_re_kg, 3),
      "gross_pt_kg": round(gross_pt_kg, 3),
      "gross_payout_usd": round(gross_payout, 2),
      "penalty_usd": round(penalty_usd, 2),
      "penalty_notes": ", ".join(penalty_notes) if penalty_notes else "None",
      "final_seller_quote_usd": round(final_seller_quote, 2),
      "clearing_fee_usd": round(clearing_fee_usd, 2),
  }


# 4. Streamlit Chat Interface
st.set_page_config(page_title="Rhenium & Catalyst Clearinghouse", page_icon="⚙️")
st.title("⚙️ Industrial Rhenium & Catalyst Clearinghouse")
st.caption(
    "Instant assay valuation for spent Pt-Re catalysts & turbine superalloys."
    " Powered by Mercury 3."
)

if "messages" not in st.session_state:
  st.session_state.messages = [{
      "role": "assistant",
      "content": (
          "Paste your raw lab assay snippet or type your batch details (e.g.,"
          " *'4.5 metric tons of spent Pt-Re catalyst, 0.32% Re, 0.28% Pt, 80"
          " ppm As'* or *'850 kg CMSX-4 turbine scrap'*)."
      ),
  }]

for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])

if user_input := st.chat_input(
    "Enter batch weight, alloy grade, or Re/Pt assay..."
):
  st.session_state.messages.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.markdown(user_input)

  conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}] + [
      {"role": m["role"], "content": m["content"]}
      for m in st.session_state.messages[-4:]
  ]

  try:
    response = client.chat.completions.create(
        model=MERCURY_MODEL_SLUG,
        messages=conversation_history,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    parsed_json = parse_and_validate(raw_content)

    if parsed_json.get("missing_required_fields"):
      reply = parsed_json.get(
          "next_chat_prompt",
          "Please provide the batch weight (kg/MT) and alloy grade or Re/Pt"
          " percentage.",
      )
    else:
      val = calculate_valuation(parsed_json)
      est_note = (
          " *(Nominal grade estimate)*"
          if parsed_json.get("is_estimated_grade")
          else ""
      )
      reply = f"""### ✅ Indicative Clearinghouse Quote
* **Material:** `{parsed_json.get('material_type')}` — **{parsed_json.get('grade_name')}**{est_note}
* **Net Weight:** `{parsed_json['net_weight_kg']:,.1f} kg`
* **Contained Metal:** `{val['gross_re_kg']} kg Re` | `{val['gross_pt_kg']} kg Pt`
* **Contaminant Deductions:** `-${val['penalty_usd']:,.2f}` *({val['penalty_notes']})*
* **Estimated Net Seller Payout:** **${val['final_seller_quote_usd']:,.2f} USD**

*Reply with your **corporate email** to lock this indicative quote and route this batch to our Tier-1 refiner desk.*"""

  
  except APIError as e:
    reply = (
        f"⚠️ **API Connection Error:** Unable to reach Inception Mercury ({e})."
        " Please verify your `INCEPTION_API_KEY` and endpoint URL."
    )
  except json.JSONDecodeError:
    reply = (
        "⚠️ **Parsing Error:** The model returned an unexpected format. Please"
        " re-enter your batch weight and Re/Pt percentages clearly."
    )
  except ValueError as e:
    reply = f"⚠️ **Validation Check:** {e}"
  except Exception as e:
    reply = f"⚠️ **Unexpected Error:** {e}"

  st.session_state.messages.append({"role": "assistant", "content": reply})
  with st.chat_message("assistant"):
    st.markdown(reply)
      
      
