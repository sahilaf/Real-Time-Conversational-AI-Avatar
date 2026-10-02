from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from livekit.api import AccessToken, VideoGrants
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# LiveKit configuration
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env file")


# The landing page is a standalone static site; its HTML references its CSS,
# JS, vendored three.js and head.glb by relative path, so it is served from
# the site root and those paths resolve without edits.
LANDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fydp_landing page")


@app.route("/")
def landing():
    """Serve the landing page, which links through to the demo."""
    return send_from_directory(LANDING_DIR, "index.html")


@app.route("/<path:filename>")
def landing_asset(filename):
    """Serve the landing page's assets. Explicit routes (/demo, /token,
    /health, /static/...) match before this catch-all."""
    return send_from_directory(LANDING_DIR, filename)


@app.route("/demo")
def demo():
    """Serve the playground UI.

    The page lives in static/playground.html rather than inline here; it is
    plain HTML/CSS/JS with no server-side templating, so keeping it in its
    own file keeps it editable and lintable.
    """
    return send_from_directory(app.static_folder, "playground.html")


@app.route("/token", methods=["POST"])
def generate_token():
    """Generate a LiveKit access token"""
    try:
        data = request.get_json()
        room_name = data.get("room_name", "my-room")
        participant_name = data.get("participant_name", "user")

        # Create access token
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )

        jwt_token = token.to_jwt()

        return jsonify({
            "token": jwt_token,
            "url": LIVEKIT_URL,
            "room_name": room_name,
            "participant_name": participant_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Plain ASCII: the Windows console defaults to cp1252, which cannot encode
    # emoji, and a UnicodeEncodeError here kills the server before it binds.
    print("=" * 60)
    print("LiveKit token server + avatar playground")
    print("=" * 60)
    print(f"  Landing    : http://localhost:5000")
    print(f"  Demo       : http://localhost:5000/demo")
    print(f"  LiveKit    : {LIVEKIT_URL}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True)