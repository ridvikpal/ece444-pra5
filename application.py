import os
import logging
import threading
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

# Flask app (Elastic Beanstalk Procfile expects "application:application")
application = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve artifact paths relative to this file; allow env overrides (empty env won't override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(BASE_DIR, "basic_classifier.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or os.path.join(BASE_DIR, "count_vectorizer.pkl")

# Log resolved paths
logger.info("CWD: %s", os.getcwd())
logger.info("Resolved MODEL_PATH: %s", MODEL_PATH)
logger.info("Resolved VECTORIZER_PATH: %s", VECTORIZER_PATH)

# Global variables for loaded artifacts
_loaded_model: Optional[object] = None
_vectorizer: Optional[object] = None
_artifact_lock = threading.Lock()

# Artifact loading
def _load_artifacts_once() -> None:
    """Lazily load model and vectorizer once per process."""
    global _loaded_model, _vectorizer
    if _loaded_model is not None and _vectorizer is not None:
        return
    with _artifact_lock:
        if _loaded_model is None or _vectorizer is None:
            import pickle
            logger.info("Loading artifacts...")
            with open(MODEL_PATH, "rb") as mf:
                _loaded_model = pickle.load(mf)
            with open(VECTORIZER_PATH, "rb") as vf:
                _vectorizer = pickle.load(vf)
            logger.info("Artifacts loaded.")

# Inference function
def _predict_text(message: str) -> str:
    """Run inference and return the predicted class as a string label."""
    _load_artifacts_once()
    X = _vectorizer.transform([message])
    pred = _loaded_model.predict(X)
    # pred[0] could be a numpy scalar; normalize to native str
    val = pred[0]
    val_py = val.item() if hasattr(val, "item") else val
    return str(val_py)

# Eager load artifacts in a background thread at startup
def _eager_load_background():
    try:
        _load_artifacts_once()
    except Exception as e:
        # Log and continue; app remains healthy and will lazy-load on first request
        logger.warning("Background eager load failed: %s", e, exc_info=True)


# Non-blocking eager load at startup
threading.Thread(target=_eager_load_background, daemon=True).start()

DEMO_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Prediction Demo</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
      textarea { width: 100%; font-size: 1rem; }
      .meta { color: #555; margin-bottom: 1rem; }
      .error { color: #b00020; }
      .result { background:#f6f8fa; padding: 1rem; border-radius: 6px; margin-top: 1rem; }
      pre { white-space: pre-wrap; word-break: break-word; background:#efefef; padding: .5rem; border-radius:4px;}
      button { margin-top: .5rem; }
    </style>
  </head>
  <body>
    <h1>Model Prediction Demo</h1>
    <p class="meta">Model loaded: <strong>{{ 'Yes' if model_loaded else 'No' }}</strong><br/>Model path: <code>{{ model_path }}</code></p>

    <form id="predict-form" method="post" action="/predict-form">
      <label for="message"><strong>Enter text to classify</strong></label><br/>
      <textarea id="message" name="message" rows="4" placeholder="Type a message...">{{ request.form.get('message','') if request else '' }}</textarea><br/>
      <button type="submit">Predict (form)</button>
    </form>

    <div id="server-result" class="result" aria-live="polite">
      {% if prediction %}
        <p><strong>Prediction:</strong> {{ prediction }}</p>
      {% endif %}
      {% if error %}
        <p class="error"><strong>Error:</strong> {{ error }}</p>
      {% endif %}
      {% if not prediction and not error %}
        <p>No prediction yet. Submit the form above or use the JSON API tester below.</p>
      {% endif %}
    </div>

    <hr/>

    <h2>JSON API tester</h2>
    <p>Send a POST to <code>/predict</code> and view JSON response below.</p>
    <input id="api-message" type="text" placeholder="Message for /predict" style="width:70%" />
    <button id="api-send">Call /predict (JSON)</button>
    <pre id="api-response"></pre>

    <script>
      // Submit form via fetch so page updates with server-rendered HTML (keeps same behavior)
      document.getElementById('predict-form').addEventListener('submit', function(e){
        e.preventDefault();
        var form = e.target;
        var data = new FormData(form);
        var btn = form.querySelector('button[type="submit"]');
        btn.disabled = true;
        fetch(form.action, { method: 'POST', body: data })
          .then(function(resp){ return resp.text(); })
          .then(function(html){
            // Replace entire document with server-rendered response so template logic is preserved
            document.open();
            document.write(html);
            document.close();
          })
          .catch(function(err){
            alert('Submission failed: ' + err);
            btn.disabled = false;
          });
      });

      // JSON API tester
      document.getElementById('api-send').addEventListener('click', function(){
        var msg = document.getElementById('api-message').value || '';
        var out = document.getElementById('api-response');
        out.textContent = 'Loading...';
        fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: msg })
        })
        .then(function(resp){ return resp.json().then(function(body){ return {status: resp.status, body: body}; }); })
        .then(function(obj){
          out.textContent = JSON.stringify(obj, null, 2);
        })
        .catch(function(err){
          out.textContent = 'Error: ' + err;
        });
      });
    </script>
  </body>
</html>
"""
# Routes
@application.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_loaded_model is not None and _vectorizer is not None),
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
    }), 200

# Demo page rendering endpoint
@application.get("/demo")
def demo():
    return render_template_string(
        DEMO_HTML,
        model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
        model_path=MODEL_PATH,
        prediction=None,
        error=None,
    )

# Form submission endpoint for demo page
@application.post("/predict-form")
def predict_form():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Field 'message' is required and must be non-empty.",
        ), 400
    try:
        label = _predict_text(message)
        return render_template_string(
            DEMO_HTML,
            model_loaded=True,
            model_path=MODEL_PATH,
            prediction=label,
            error=None,
        )
    except FileNotFoundError:
        return render_template_string(
            DEMO_HTML,
            model_loaded=False,
            model_path=MODEL_PATH,
            prediction=None,
            error="Model artifacts not found on server.",
        ), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Inference failed.",
        ), 500

# JSON API endpoint for predictions
@application.post("/predict")
def predict_json():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    try:
        label = _predict_text(message)
        return jsonify({"label": label}), 200
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found on server."}), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Inference failed."}), 500


if __name__ == "__main__":
    # Local dev run; in EB, Gunicorn (from Procfile) will host the app
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=False)
