from app import db, GameContent, User, UserScores
from app import app

def populate_game_content():
    # Sample game content (e.g., images and names)
    content = [
        {"image_url": "http://example.com/shark.jpg", "name": "Shark"},
        {"image_url": "http://example.com/dolphin.jpg", "name": "Dolphin"},
        {"image_url": "http://example.com/whale.jpg", "name": "Whale"},
        {"image_url": "http://example.com/octopus.jpg", "name": "Octopus"},
        {"image_url": "http://example.com/turtle.jpg", "name": "Turtle"}
    ]
    
    for item in content:
        if not GameContent.query.filter_by(name=item["name"]).first():
            new_content = GameContent(image_url=item["image_url"], name=item["name"])
            db.session.add(new_content)
    
    db.session.commit()

def populate_users_and_scores():
    # Sample users
    users = [
        {"name": "Alice", "email": "alice@example.com", "password": "password123"},
        {"name": "Bob", "email": "bob@example.com", "password": "password123"}
    ]
    
    for user in users:
        if not User.query.filter_by(email=user["email"]).first():
            new_user = User(name=user["name"], email=user["email"], password=user["password"])
            db.session.add(new_user)
    
    db.session.commit()

    # Sample scores for the users
    scores = [
        {"user_id": "alice@example.com", "score": 100, "time_taken": "00:02:45"},
        {"user_id": "alice@example.com", "score": 90, "time_taken": "00:03:10"},
        {"user_id": "bob@example.com", "score": 80, "time_taken": "00:03:50"}
    ]
    
    for score in scores:
        new_score = UserScores(user_id=score["user_id"], score=score["score"], time_taken=score["time_taken"])
        db.session.add(new_score)
    
    db.session.commit()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure that all tables are created
        populate_game_content()  # Add game content
        populate_users_and_scores()  # Add users and scores
        print("Sample data populated successfully!")
