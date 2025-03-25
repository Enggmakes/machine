import os
import pickle
import sqlite3
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"  # Default value (for local users)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model and vectorizer
with open("file_classifier.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Database setup
def get_db_connection():
    conn = sqlite3.connect("files.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            path TEXT,
            size INTEGER,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

create_table()

@app.route("/")
def home():
    return "AI-Powered File Organizer API is running!"

# Set user's upload folder
@app.route("/set_upload_folder", methods=["POST"])
def set_upload_folder():
    global UPLOAD_FOLDER
    data = request.json
    folder_path = data.get("folder_path", "").strip()

    if not folder_path:
        return jsonify({"error": "Folder path is required"}), 400

    os.makedirs(folder_path, exist_ok=True)
    UPLOAD_FOLDER = folder_path
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    return jsonify({"message": f"Upload folder set to: {UPLOAD_FOLDER}"})

# Classify file
def classify_file(filename):
    features = vectorizer.transform([filename])
    return model.predict(features)[0]

# Organize files into user-defined folder
def organize_file(file_path, category):
    try:
        category_folder = os.path.join(app.config["UPLOAD_FOLDER"], category)
        os.makedirs(category_folder, exist_ok=True)
        
        new_path = os.path.join(category_folder, os.path.basename(file_path))
        shutil.move(file_path, new_path)

        print(f"File moved to: {new_path}")
        return new_path
    except Exception as e:
        print(f"Error organizing file: {e}")
        return file_path

# Upload & classify multiple files
@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    response_data = []
    conn = get_db_connection()

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        file_size = os.path.getsize(file_path)

        # Classify file
        category = classify_file(filename)

        # Organize file
        new_path = organize_file(file_path, category)

        # Store in database
        conn.execute(
            "INSERT INTO files (name, category, path, size) VALUES (?, ?, ?, ?)",
            (filename, category, new_path, file_size),
        )

        response_data.append({"name": filename, "category": category, "path": new_path})

    conn.commit()
    conn.close()

    return jsonify({"message": "Files uploaded successfully", "files": response_data})

# Fetch files from database
@app.route("/files", methods=["GET"])
def get_files():
    conn = get_db_connection()
    files = conn.execute("SELECT * FROM files").fetchall()
    conn.close()
    return jsonify([dict(file) for file in files])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
