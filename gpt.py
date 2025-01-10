import os
import base64
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL = "gpt-4o" 
llm = OpenAI(api_key="")

#from cook-book
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


base_dir = "/Users/mhmh/Desktop/liz/data"
emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]

#for results
actual_labels=[]
predicted_labels=[]

# Process each emotion folder
for emotion in emotions:
    emotion_folder = os.path.join(base_dir, emotion)
    
    # Process images in each emotion folder
    for image_filename in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image_filename)
        
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Encode image to base64
            base64_image = encode_image(image_path)
            
            #apicall
            response = llm.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an emotion expert who can detect emotions from human facial expressions in images. Your task is to classify the emotion from the given facial expression based on the CK+ dataset. The possible emotions are: anger, disgust, fear, happy, sadness, surprise."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Classify the emotion from the image.Only output the emotion label."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]}
                ],
                temperature=0.0,
            )
            
            #llm response
            predicted_emotion = response.choices[0].message.content.strip()
            
            #adding to arrays
            actual_labels.append(emotion.lower()) 
            predicted_labels.append(predicted_emotion.lower())
            
            # Print the response
            print(f"Image: {image_filename} | Actual: {emotion} | Predicted: {predicted_emotion}")


 # Convert actual emotion to lowercase
  # Convert predicted emotion to lowercase

accuracy = accuracy_score(actual_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='macro')

# Print metrics
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
