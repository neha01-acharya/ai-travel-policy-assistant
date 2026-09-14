from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT AI AGENT
# ============================================================

from src.agent import TravelPolicyAgent


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = TravelPolicyAgent()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return render_template("index.html")


# ============================================================
# ASK ASSISTANT
# ============================================================

@app.post("/ask")
def ask():

    try:
        data = request.get_json(silent=True) or {}

        question = str(
            data.get("question", "")
        ).strip()

        employee_id = str(
            data.get("employee_id", "")
        ).strip()

        # ----------------------------------------------------
        # VALIDATE QUESTION
        # ----------------------------------------------------

        if not question:
            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        result = agent.run(
            query=question,
            employee_id=employee_id or None,
        )

        return jsonify({
            "success": True,
            "answer": result.get("response", ""),
            "sources": result.get("sources", []),
            "route": result.get("route", ""),
        })

    except Exception as exc:

        print(
            f"Application error: {exc}",
            file=sys.stderr,
        )

        return jsonify({
            "success": False,
            "error": (
                "Sorry, something went wrong "
                "while processing your question."
            ),
        }), 500


# ============================================================
# CLEAR CONVERSATION
# ============================================================

@app.post("/clear")
def clear():

    try:

        agent.clear_memory()

        return jsonify({
            "success": True,
            "message": "Conversation cleared."
        })

    except Exception as exc:

        print(
            f"Clear error: {exc}",
            file=sys.stderr,
        )

        return jsonify({
            "success": False,
            "error": "Unable to clear the conversation."
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "healthy",
        "service": "AI Travel Policy Assistant"
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )