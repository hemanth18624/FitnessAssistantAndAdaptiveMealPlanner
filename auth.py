from flask import Blueprint, request, session, redirect, render_template, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_db
import mysql.connector

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    email = request.form.get("email", "").strip().lower()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not email or not username or not password:
        return jsonify({"error": "All fields are required"}), 400

    password_hash = generate_password_hash(password)

    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute(
            """
            INSERT INTO users (email, username, password_hash)
            VALUES (%s, %s, %s)
            """,
            (email, username, password_hash)
        )
        db.commit()

    except mysql.connector.IntegrityError:
        return jsonify({"error": "Email or Username already exists"}), 409
    finally:
        cursor.close()
        db.close()

    return redirect("/login")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        return jsonify({"error": "Invalid credentials"}), 400

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM users WHERE username = %s",
        (username,)
    )
    user = cursor.fetchone()

    cursor.close()
    db.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]

    return redirect("/")


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/login")
