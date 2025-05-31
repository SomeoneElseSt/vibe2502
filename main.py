import requests
import dotenv
import os 
import numpy as np
import time
from dotenv import load_dotenv
from openai import OpenAI
import librosa
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Conv1D = tf.keras.layers.Conv1D
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
concatenate = tf.keras.layers.concatenate
import pickle
import numpy as np
import scipy.stats as stats
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
import eyed3
import pydub
import matplotlib.pyplot 
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

EMOJIS_API_KEY = os.getenv("EMOJIS_API_KEY")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def transcribe_audio(
    file_path: str,
    language_code: str = "es",
    poll_interval: int = 3,
    speakers_expected: int | None = None,
    timeout_seconds: int = 6 * 60,
) -> str:
    headers = {"authorization": os.getenv("ASSEMBLY_API_KEY")}

    with open(file_path, "rb") as f:
        upload_resp = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=f,
            timeout=60,
        )
    upload_resp.raise_for_status()
    upload_url = upload_resp.json()["upload_url"]

    job_payload = {
        "audio_url": upload_url,
        "language_code": language_code,
    }
    if speakers_expected is not None:
        job_payload.update({"speaker_labels": True, "speakers_expected": speakers_expected})

    job_resp = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers | {"content-type": "application/json"},
        json=job_payload,
        timeout=30,
    )
    job_resp.raise_for_status()
    transcript_id = job_resp.json()["id"]

    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    deadline = time.time() + timeout_seconds
    while True:
        poll_resp = requests.get(endpoint, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        status = poll_resp.json()["status"]

        if status == "completed":
            return poll_resp.json().get("text", "")
        if status == "error":
            raise RuntimeError(f"AssemblyAI error: {poll_resp.json().get('error')}")

        if time.time() >= deadline:
            raise RuntimeError("Transcription timed out.")
        time.sleep(poll_interval)

def detect_audio_tone_multiclass(audio_file_path, model_path=None):
    emotion_classes = [
        'happy', 'sad', 'angry', 'neutral', 'fearful', 
        'surprised', 'disgusted', 'sarcastic', 'excited', 'calm'
    ]
    
    y, sr = librosa.load(audio_file_path, sr=22050, duration=10)
    
    features = extract_comprehensive_features(y, sr)
    
    if model_path and os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']
        
        features_scaled = scaler.transform([features['combined_vector']])
        
        probabilities = model.predict_proba(features_scaled)[0]
        confidence_scores = dict(zip(emotion_classes, probabilities))
        
    else:
        confidence_scores = calculate_rule_based_probabilities(features, emotion_classes)
    
    sorted_predictions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    primary_tone = sorted_predictions[0][0]
    top_3_predictions = [
        {'tone': tone, 'confidence': round(conf, 3)} 
        for tone, conf in sorted_predictions[:3]
    ]
    
    transcribed_text = transcribe_audio(audio_file_path)
    
    return {
        'primary_tone': primary_tone,
        'confidence_scores': {k: round(v, 3) for k, v in confidence_scores.items()},
        'top_3_predictions': top_3_predictions,
        'audio_features': features,
        'transcribed_text': transcribed_text,
        'prediction_certainty': sorted_predictions[0][1]
    }

def extract_comprehensive_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_features = np.mean(mfccs.T, axis=0)
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_features = np.mean(mel_spec.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_features = np.mean(chroma.T, axis=0)
    
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_features = np.mean(contrast.T, axis=0)
    
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_features = np.mean(tonnetz.T, axis=0)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    
    if len(pitch_values) > 0:
        pitch_mean = float(np.mean(pitch_values))
        pitch_std = float(np.std(pitch_values))
        pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
    else:
        pitch_mean = pitch_std = pitch_range = 0.0
    
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms_energy = float(np.mean(librosa.feature.rms(y=y)))
    
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)
    
    print(f"Pitch values: mean={pitch_mean}, std={pitch_std}, range={pitch_range}")
    print(f"Spectral: centroid={spectral_centroid}, bandwidth={spectral_bandwidth}, rolloff={spectral_rolloff}")
    print(f"Temporal: zcr={zcr}, energy={rms_energy}, tempo={tempo}")
    
    try:
        mfcc_flat = np.array(mfcc_features).flatten()
        mel_flat = np.array(mel_features[:20]).flatten()
        chroma_flat = np.array(chroma_features).flatten()
        contrast_flat = np.array(contrast_features).flatten()
        tonnetz_flat = np.array(tonnetz_features).flatten()
        
        pitch_array = np.array([pitch_mean, pitch_std, pitch_range], dtype=float)
        spectral_array = np.array([spectral_centroid, spectral_bandwidth, spectral_rolloff], dtype=float)
        temporal_array = np.array([zcr, rms_energy, tempo], dtype=float)
        
        print(f"MFCC shape: {mfcc_flat.shape}")
        print(f"Mel shape: {mel_flat.shape}")
        print(f"Chroma shape: {chroma_flat.shape}")
        print(f"Contrast shape: {contrast_flat.shape}")
        print(f"Tonnetz shape: {tonnetz_flat.shape}")
        print(f"Pitch array shape: {pitch_array.shape}")
        print(f"Spectral array shape: {spectral_array.shape}")
        print(f"Temporal array shape: {temporal_array.shape}")
        
        combined_vector = np.concatenate([
            mfcc_flat,
            mel_flat,
            chroma_flat,
            contrast_flat,
            tonnetz_flat,
            pitch_array,
            spectral_array,
            temporal_array
        ])
        
        print(f"Final combined vector shape: {combined_vector.shape}")
        
    except Exception as e:
        print(f"Error in feature concatenation: {e}")
        combined_vector = np.array([
            float(pitch_mean), float(pitch_std), float(pitch_range),
            float(spectral_centroid), float(spectral_bandwidth), float(spectral_rolloff),
            float(zcr), float(rms_energy), float(tempo)
        ], dtype=float)
        print(f"Using fallback vector with shape: {combined_vector.shape}")
    
    return {
        'mfcc': mfcc_features,
        'mel_spectrogram': mel_features,
        'chroma': chroma_features,
        'contrast': contrast_features,
        'tonnetz': tonnetz_features,
        'pitch_stats': [pitch_mean, pitch_std, pitch_range],
        'spectral_features': [spectral_centroid, spectral_bandwidth, spectral_rolloff],
        'temporal_features': [zcr, rms_energy, tempo],
        'combined_vector': combined_vector
    }

def calculate_rule_based_probabilities(features, emotion_classes):
    pitch_mean, pitch_std, pitch_range = features['pitch_stats']
    zcr, rms_energy, tempo = features['temporal_features']
    
    probs = {emotion: 0.1 for emotion in emotion_classes}
    
    if pitch_mean > 180 and rms_energy > 0.03 and 120 < tempo < 180:
        probs['happy'] += 0.4
        probs['excited'] += 0.3
    
    if pitch_mean < 150 and rms_energy < 0.02 and tempo < 100:
        probs['sad'] += 0.5
        probs['calm'] += 0.2
    
    if pitch_std > 50 and rms_energy > 0.04 and tempo > 140:
        probs['angry'] += 0.6
    
    if pitch_std > 40 and 0.02 < rms_energy < 0.04:
        probs['sarcastic'] += 0.4
    
    if pitch_mean > 200 and pitch_std > 60 and zcr > 0.1:
        probs['fearful'] += 0.5
        probs['surprised'] += 0.3
    
    if pitch_std < 30 and 0.015 < rms_energy < 0.035:
        probs['calm'] += 0.4
        probs['neutral'] += 0.3
    
    total = sum(probs.values())
    return {emotion: prob/total for emotion, prob in probs.items()}

def train_ensemble_emotion_model(audio_files, labels, save_path="emotion_model.pkl"):
    X = []
    for audio_file in audio_files:
        y, sr = librosa.load(audio_file, sr=22050, duration=10)
        features = extract_comprehensive_features(y, sr)
        X.append(features['combined_vector'])
    
    X = np.array(X)
    y = np.array(labels)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'gradient_boost': GradientBoostingClassifier(random_state=42)
    }
    
    ensemble_predictions = []
    for name, model in models.items():
        model.fit(X_scaled, y_encoded)
        ensemble_predictions.append(model.predict_proba(X_scaled))
    
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X_scaled, y_encoded)
    
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'emotion_classes': label_encoder.classes_.tolist()
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return final_model, scaler, label_encoder

def get_embeddings(text: str):
    if not text.strip():
        return None
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def extract_keywords_from_input(user_input: str, k: int = 5) -> str:
    if not user_input.strip():
        return "No input provided"

    words = [w for w in user_input.split() if len(w) > 1]
    if not words:
        return "No valid words found in input"

    word_embeddings = []
    valid_words = []
    for word in words:
        try:
            embedding = get_embeddings(word)
            if embedding is not None:
                embedding_array = np.array(embedding, dtype=np.float64)
                norm = np.linalg.norm(embedding_array) + 1e-8
                normalized_embedding = embedding_array / norm
                word_embeddings.append(normalized_embedding)
                valid_words.append(word)
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            continue

    if not word_embeddings:
        return "Could not generate valid embeddings"

    try:
        input_embedding = get_embeddings(user_input)
        if input_embedding is None:
            return "Failed to generate embedding for input"

        input_embedding = np.array(input_embedding, dtype=np.float64)
        input_norm = np.linalg.norm(input_embedding) + 1e-8
        input_embedding = input_embedding / input_norm

        word_embeddings = np.array(word_embeddings, dtype=np.float64)
        input_embedding = input_embedding.reshape(1, -1)

        similarities = np.zeros(len(word_embeddings))
        for i, word_emb in enumerate(word_embeddings):
            sim = np.clip(np.dot(input_embedding.flatten(), word_emb.flatten()), -1.0, 1.0)
            similarities[i] = sim

        k = min(k, len(valid_words))
        top_k_indices = np.argsort(similarities)[::-1][:k]
        extracted_keywords = [valid_words[i] for i in top_k_indices]

        return ", ".join(extracted_keywords)

    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return "Error extracting keywords"

def get_emoji_image_file_path(prompt, save_dir=".", api_key=None):
    if api_key:
        EMOJIS_API_KEY = api_key
    else:
        EMOJIS_API_KEY = os.getenv('EMOJIS_API_KEY', '<your_api_key_here>')
    
    headers = {
        "Authorization": f"Bearer {EMOJIS_API_KEY}",
        "Content-Type": "application/json"
    }

    def create_emoji(prompt):
        url = "https://api.emojis.com/api/v1/emojis"
        payload = {
            "kind": "text_to_emoji",
            "prompt": prompt,
            "enable_prompt_safety_check": True
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('id'), data.get('status'), data.get('error')
        else:
            return None, None, f"HTTP error {response.status_code}"

    def retrieve_emoji(emoji_id):
        url = f"https://api.emojis.com/api/v1/emojis/{emoji_id}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('status'), data.get('error'), data.get('formats')
        else:
            return None, f"HTTP error {response.status_code}", None

    def download_image(image_url, save_dir):
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                parsed_url = urlparse(image_url)
                filename = os.path.basename(parsed_url.path)
                file_path = os.path.join(save_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path
            else:
                return None
        except Exception:
            return None

    emoji_id, status, error = create_emoji(prompt)
    if not emoji_id:
        return None, f"Failed to create emoji: {error}"

    for _ in range(10):
        status, error, formats = retrieve_emoji(emoji_id)
        if status == "generated":
            if formats and 'png' in formats:
                image_url = formats['png'].get('512') or formats['png'].get('128')
                if image_url:
                    file_path = download_image(image_url, save_dir)
                    if file_path:
                        return file_path, None
                    else:
                        return None, "Failed to download image"
            return None, "No image URL found"
        elif status in ["errored", "inappropriate"]:
            return None, f"Emoji generation failed: {error}"
        time.sleep(2)

    return None, "Emoji generation timed out"

if __name__ == "__main__":
    import sys
    
    audio_file = input("Enter the path to your audio file: ")
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)
    
    print("\n1. Analyzing audio tone...")
    tone_results = detect_audio_tone_multiclass(audio_file)
    print("\nTone Analysis Results:")
    print(f"Primary Tone: {tone_results['primary_tone']}")
    print("\nTop 3 Predicted Emotions:")
    for pred in tone_results['top_3_predictions']:
        print(f"- {pred['tone']}: {pred['confidence']:.3f}")
    
    print("\n2. Getting transcription and analyzing keywords...")
    transcription = transcribe_audio(audio_file)
    print("\nTranscription:", transcription)
    
    keywords = None
    if transcription:
        keywords = extract_keywords_from_input(transcription, k=5)
        print("\nKey semantic words:", keywords)
    else:
        print("\nCould not extract keywords due to transcription failure")
    
    if keywords and tone_results:
        emoji_prompt = f"Generate an emoji based on the following mood and conversation keywords: {tone_results['primary_tone']} mood with themes of {keywords} which off the context of this transcription {transcription}"
        print("\n3. Generating emoji based on analysis...")
        emoji_path, error = get_emoji_image_file_path(emoji_prompt)
        
        if emoji_path:
            print(f"Emoji generated successfully! Saved to: {emoji_path}")
        else:
            print(f"Failed to generate emoji: {error}")
