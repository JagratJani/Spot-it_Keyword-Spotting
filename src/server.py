from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import librosa
from src.keyword_spotting import KeywordSpotter

app = Flask(__name__)
CORS(app)

detected_words = set()
print('detected:', detected_words)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory('uploads', filename)

BASE_URL = "http://192.168.120.154:3000"

ads_stories = [
    {"id": "11", "user": "Sponsered", "img": "https://img.freepik.com/premium-vector/birds-banner-design-facebook-banner-design-fb-post-design-ads-design-birds-ads-design_585740-47.jpg", "show": "false"},
    {"id": "22", "user": "Sponsered", "img": "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/dog-food-ads-design-template-789d7b7857c76d145a268f44085dd98c_screen.jpg?ts=1650357822", "show": "false"},
    {"id": "33", "user": "Sponsered", "img": "https://i.pinimg.com/originals/e0/be/24/e0be24141974cd0577ea7f443d617069.jpg", "show": "false"},
    {"id": "44", "user": "Sponsered", "img": "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/dream-house-ads-design-template-914cf2a8bc013ed1e37e8cc3e0f09265_screen.jpg?ts=1650964341", "show": "false"},
]


ads_posts = [
    {
        "id": "12",
        "user": "Sponsered",
        "profileImg": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQl_ZYG46UzTwTMwEIdLGB5yKIk_t8U5fBC4A&s",
        "img": "https://img.freepik.com/premium-vector/birds-banner-design-facebook-banner-design-fb-post-design-ads-design-birds-ads-design_585740-47.jpg",
        "likes": 89,
        "caption": "Birds Caring",
        "time": "AD",
        "comments": 17,
        "show": "true"
    },
    {
        "id": "23",
        "user": "Sponsered",
        "profileImg": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQl_ZYG46UzTwTMwEIdLGB5yKIk_t8U5fBC4A&s",
        "img": "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/dog-food-ads-design-template-789d7b7857c76d145a268f44085dd98c_screen.jpg?ts=1650357822",
        "likes": 75,
        "caption": "Dog Stuffs",
        "time": "AD",
        "comments": 12,
        "show": "true"
    },
    {
        "id": "34",
        "user": "Sponsered",
        "profileImg": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQl_ZYG46UzTwTMwEIdLGB5yKIk_t8U5fBC4A&s",
        "img": "https://i.pinimg.com/originals/e0/be/24/e0be24141974cd0577ea7f443d617069.jpg",
        "likes": 169,
        "caption": "Cat Caring",
        "time": "AD",
        "comments": 41,
        "show": "true"
    },
    {
        "id": "45",
        "user": "Sponsered",
        "profileImg": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQl_ZYG46UzTwTMwEIdLGB5yKIk_t8U5fBC4A&s",
        "img": "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/dream-house-ads-design-template-914cf2a8bc013ed1e37e8cc3e0f09265_screen.jpg?ts=1650964341",
        "likes": 204,
        "caption": "Dream House",
        "time": "AD",
        "comments": 37,
        "show": "true"
    }
]

original_stories = [
    {"id": "1", "user": "Your story", "img": f"{BASE_URL}/uploads/images/saaheil_3.jpg", "show": "true"},
    {"id": "2", "user": "jani_jagrat", "img": f"{BASE_URL}/uploads/images/jagrat_1.jpeg", "show": "true"},
    {"id": "3", "user": "aesha_28_", "img": f"{BASE_URL}/uploads/images/aesha_1.jpeg", "show": "true"},
    {"id": "4", "user": "hetvi_parikh", "img": f"{BASE_URL}/uploads/images/hetvi_1.jpg", "show": "true"},
    {"id": "5", "user": "ritu_chauhan711", "img": f"{BASE_URL}/uploads/images/ritu_1.jpeg", "show": "true"},
    {"id": "6", "user": "bhavya_vishnani", "img": f"{BASE_URL}/uploads/images/bhavya_1.jpeg", "show": "true"},
    {"id": "7", "user": "mustansir_petiwala", "img": f"{BASE_URL}/uploads/images/mustan_1.jpeg", "show": "true"}
    
]

original_posts = [
    {
        "id": "5",
        "user": "ritu_chauhan711",
        "profileImg": f"{BASE_URL}/uploads/images/ritu_2.jpg",
        "img": f"{BASE_URL}/uploads/images/ritu_1.jpeg",
        "likes": 9505,
        "caption": "Found the best ones",
        "time": "5h",
        "comments": 333,
        "show": "true"
    },
     {
        "id": "6",
        "user": "bhavya_vishnani",
        "profileImg": f"{BASE_URL}/uploads/images/bhavya_1.jpeg",
        "img": f"{BASE_URL}/uploads/images/bhavya_2.jpg",
        "likes": 98,
        "caption": "Oyee todd dunga phhod dunga!",
        "time": "5h",
        "comments": 30,
        "show": "true"
    },
     {
        "id": "7",
        "user": "mustansir_petiwala",
        "profileImg": f"{BASE_URL}/uploads/images/mustan_1.jpeg",
        "img": f"{BASE_URL}/uploads/images/mustan_2.jpg",
        "likes": 98,
        "caption": "i'm my own admire",
        "time": "5h",
        "comments": 30,
        "show": "true"
    },
    {
        "id": "1",
        "user": "saaheil_2011",
        "profileImg": f"{BASE_URL}/uploads/images/saaheil_3.jpg",
        "img": f"{BASE_URL}/uploads/images/saaheil_2.jpg",
        "likes": 120,
        "caption": "Simplicity with a touch of glass",
        "time": "2h",
        "comments": 5,
        "show": "true"
    },
    {
        "id": "2",
        "user": "jagrat_jani",
        "profileImg": f"{BASE_URL}/uploads/images/jagrat_2.jpg",
        "img": f"{BASE_URL}/uploads/images/jagrat_1.jpeg",
        "likes": 205,
        "caption": "Sandy toes and sun kissed nose!",
        "time": "5h",
        "comments": 30,
        "show": "true"
    },

    {
        "id": "3",
        "user": "aesha_28_",
        "profileImg": f"{BASE_URL}/uploads/images/aesha_1.jpeg",
        "img": f"{BASE_URL}/uploads/images/aesha_2.jpg",
        "likes": 95,
        "caption": "good vibes!",
        "time": "5h",
        "comments": 3,
        "show": "true"
    },
     {
        "id": "4",
        "user": "hetvi_parikh",
        "profileImg": f"{BASE_URL}/uploads/images/hetvi_1.jpg",
        "img": f"{BASE_URL}/uploads/images/hetvi_2.jpg",
        "likes": 95,
        "caption": "Just a princess wandering through her castle halls",
        "time": "5h",
        "comments": 3,
        "show": "true"
    }
    
]



@app.route('/api/stories', methods=['GET'])
def get_stories():
    response_stories = original_stories.copy()
    # if 'dog' in detected_words:
    #     dog_ad = next((ad for ad in ads_stories if ad['id'] == '23'), None)
    #     if dog_ad and not any(story['id'] == '23' for story in response_stories):
    #         response_stories.insert(2, dog_ad)
    return jsonify(response_stories)

@app.route('/api/posts', methods=['GET'])
def get_posts():
    print("posts")
    response_posts = original_posts.copy()
    if 'house' in detected_words:
        house_ad = next((ad for ad in ads_posts if ad['id'] == '45'), None)
        if house_ad and not any(post['id'] == '45' for post in response_posts):
            response_posts.insert(2, house_ad)
    
    if 'cat' in detected_words:
        house_ad = next((ad for ad in ads_posts if ad['id'] == '34'), None)
        if house_ad and not any(post['id'] == '34' for post in response_posts):
            response_posts.insert(6, house_ad)

    if 'dog' in detected_words:
        house_ad = next((ad for ad in ads_posts if ad['id'] == '23'), None)
        if house_ad and not any(post['id'] == '23' for post in response_posts):
            response_posts.insert(4, house_ad)

    
    return jsonify(response_posts)



@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        audio, sr = librosa.load(file_path, sr=16000)
        spotter = KeywordSpotter()
        prediction = spotter.analyze_audio(audio)
        if prediction != '_background_noise_':
            detected_words.add(prediction)
        return jsonify({"message": "Audio processed", "keyword": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)