from pathlib import Path

from flask import Flask

from .config import Config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    Path(app.config["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)

    from .routes import main_bp  # noqa: WPS433

    app.register_blueprint(main_bp)

    with app.app_context():
        if (
            app.config.get("PRELOAD_MODEL_ON_STARTUP", False)
            and app.config.get("PREDICTION_ENGINE", "heuristic") != "heuristic"
        ):
            from .services.model_service import get_prediction_service

            try:
                get_prediction_service().load()
            except FileNotFoundError:
                pass

    return app
