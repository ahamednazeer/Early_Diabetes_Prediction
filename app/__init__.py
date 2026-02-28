from pathlib import Path

from flask import Flask

from .config import Config
from .extensions import db, login_manager


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    Path(app.config["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User  # noqa: WPS433
    from .routes import main_bp  # noqa: WPS433

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.query.get(int(user_id))

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        _ensure_default_user()

    return app


def _ensure_default_user() -> None:
    from .models import User

    default_username = Config.DEFAULT_USERNAME
    default_password = Config.DEFAULT_PASSWORD

    existing_user = User.query.filter_by(username=default_username).first()
    if existing_user:
        return

    user = User(username=default_username)
    user.set_password(default_password)
    db.session.add(user)
    db.session.commit()
