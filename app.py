from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# model import
from git_base_model import generate_captions_git_base  # Import from git_large_model.py
from PIL import Image
import requests
from io import BytesIO
import base64
import random  # For selecting game content

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///seesay.db'
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

CORS(app, origins=["http://localhost:3000"], supports_credentials=True)  # Enable CORS for React frontend

@app.route("/",methods=["GET"])
def hello():
    return  jsonify({'Greetings': 'FYP flask app'})

# Image caption function start 
@app.route('/imageCaptioning', methods=['POST'])
def predict_caption():
    generated_caption = None
    image = None
    image2generate = None  # Initialize image2generate
    
    image_file = request.files.get('imageFile')
    image_url = request.form.get('imageURL')
   
    if image_file:
        image2generate = Image.open(image_file)
        image = f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    elif image_url:
        response = requests.get(image_url)
        image2generate = Image.open(BytesIO(response.content))
        image = f"data:image/jpeg;base64,{base64.b64encode(response.content).decode('utf-8')}"
    else:
        return jsonify({'error': 'No image file or URL provided'}), 400  # Return an error if no image is provided

    if image2generate:
        generated_caption = generate_captions_git_base(image2generate)
    else:
        return jsonify({'error': 'Failed to process image'}), 500  # Return an error if image processing fails

    return jsonify({'caption': generated_caption, 'image': image})

# Image caption function end

# User model (for admin + parent accounts)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    profileID = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(150), nullable=False, unique=True)
    name = db.Column(db.String(), nullable=False)
    email = db.Column(db.String(), nullable=True, unique=True)
    password = db.Column(db.String(150), nullable=False)
    parentID = db.Column(db.Integer, nullable=True)
    # icon = db.Column(db.String(), nullable=True)

    # Define relationship
    # children = db.relationship('User', backref=db.backref('parent', remote_side=[id]), lazy=True)

# Game Content model
class GameContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_url = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)

# User Scores model
class UserScores(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    time_taken = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

# Create DB
with app.app_context():
    db.create_all()

# Fetch all users (for user list)
@app.route('/get_users', methods=['GET'])
def get_users():
    # global session_user # temp
    # session_user = session['user']
    user_id = request.args.get('user_id')
    print(user_id)
    # user_id2 = session.get("user_id")
    # print(user_id2)
    if not user_id:
        return jsonify({"message": "Login required!"}), 401
    current_user = User.query.filter_by(id=user_id).first()
    child_users = User.query.filter_by(parentID=user_id).order_by(User.id.desc()).all()
    user_list = [user.name for user in child_users]
    user_list.append(current_user.name)
    return jsonify(user_list), 200

# Add new child user (for user list)
@app.route('/add_child', methods=['POST'])
def add_child():
    # global session_user # temp
    data = request.get_json()
    user_id = data.get('user_id') 
    if not user_id:
        return jsonify({"message": "Login required!"}), 401
    
    # Check if the user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"message": "Username already exists"}), 400

    # Create new user
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(profileID=3, username=data['username'], name=data['name'], password=hashed_password, parentID=user_id)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User added successfully!"}), 201

# Parent Signup Route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    # Check if the user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"message": "Username already exists!"}), 400
    
    # Check if the email already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "Email already exists!"}), 400
    
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(profileID=2, username=data['username'], name=data['name'], email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created successfully!"}), 201

# Login Route
@app.route('/login', methods=['POST'])
def login():
    # global session_user # temp
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        session['user_id'] = user.id
        # session_user = user.id # temp

        # Check if the user role
        if user.profileID == 1: # admin
            return jsonify({"message": "Admin login successful!", "user_id": user.id, "profile": "admin"}), 200
        elif user.profileID == 2: # parent
            return jsonify({"message": "Parent login successful!", "user_id": user.id, "profile": "parent"}), 200
        elif user.profileID == 3: # child
            return jsonify({"message": "Child login successful!", "user_id": user.id, "profile": "child"}), 200
   
    return jsonify({"message": "Invalid username or password!"}), 401

# Logout Route
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    session.clear()
    return jsonify({"message": "Logout successful!"}), 200

# Add new child profile for the logged-in parent
# @app.route('/add_child_profile', methods=['POST'])
# def add_child_profile():
#     if 'user' not in session:
#         return jsonify({"message": "Login required!"}), 401

#     data = request.get_json()
#     user_name = data['name']
#     parent_email = session['user']  # Get the logged-in parent's email

#     parent = Users.query.filter_by(email=parent_email).first()

#     # Check if the child name already exists under this parent
#     if Users.query.filter_by(name=user_name, parent_id=parent.id).first():
#         return jsonify({"message": "Child profile already exists"}), 400

#     # Create the child profile
#     new_child = Users(name=user_name, email=f"{user_name}@example.com", password='default', parent_id=parent.id)
#     db.session.add(new_child)
#     db.session.commit()

#     return jsonify({"message": "Child profile added successfully!"}), 201

# Get game content route
@app.route('/api/game-content', methods=['GET'])
def get_game_content():
    try:
        content = GameContent.query.all()
        if len(content) < 5:
            return jsonify({"error": "Not enough content available to create a game."}), 400

        # Select 5 random pictures from the content
        pictures = random.sample(content, min(len(content), 5))
        matching_words = [picture.name for picture in pictures]
        
        # Select additional words, ensuring we don't select duplicates
        additional_words = random.sample(
            [c.name for c in content if c.name not in matching_words],
            min(len(content) - len(matching_words), 5)
        )
        
        words = matching_words + additional_words
        random.shuffle(words)

        # Prepare the response
        game_data = {
            "pictures": [{"id": p.id, "imageUrl": p.image_url, "name": p.name} for p in pictures],
            "words": words
        }
        return jsonify(game_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Save user score route
@app.route('/api/save-score', methods=['POST'])
def save_score():
    data = request.json
    user_id = data.get('userId')
    score = data.get('score')
    time_taken = data.get('timeTaken')

    if not user_id or not score or not time_taken:
        return jsonify({"error": "Invalid request data"}), 400

    try:
        new_score = UserScores(user_id=user_id, score=score, time_taken=time_taken)
        db.session.add(new_score)
        db.session.commit()
        return jsonify({"message": "Score saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get user scores route
@app.route('/api/user-scores', methods=['GET'])
def get_user_scores():
    user_id = request.args.get('userId')
    
    try:
        if user_id:
            scores = UserScores.query.filter_by(user_id=user_id).all()
        else:
            scores = UserScores.query.all()
        score_list = [
            {
                "userId": score.user_id,
                "score": score.score,
                "timeTaken": score.time_taken,
                "date": score.date.strftime('%Y-%m-%d')
            } for score in scores
        ]
        return jsonify({"scores": score_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
