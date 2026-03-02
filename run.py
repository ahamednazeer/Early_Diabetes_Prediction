import os

from app import create_app


app = create_app()

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print("App started on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=debug, use_reloader=debug)
