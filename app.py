from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class TypingAssistantModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'typing_speed', 'accuracy', 'session_duration', 'word_count',
            'error_rate', 'backspace_frequency', 'pause_frequency', 'text_complexity'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load the scaler
            if os.path.exists('scaler.pkl'):
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning("Scaler file not found, creating a dummy scaler")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                # Fit with dummy data for demo purposes
                dummy_data = np.random.rand(100, len(self.feature_names))
                self.scaler.fit(dummy_data)
            
            # Load the model
            if os.path.exists('typing_assist_model.h5'):
                self.model = keras.models.load_model('typing_assist_model.h5')
                logger.info("Model loaded successfully")
            elif os.path.exists('typing_assist_model.keras'):
                self.model = keras.models.load_model('typing_assist_model.keras')
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model file not found, creating a dummy model")
                self.create_dummy_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for demonstration purposes"""
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(len(self.feature_names),)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("Dummy model created for demonstration")
    
    def preprocess_features(self, features):
        """Preprocess the input features"""
        try:
            # Ensure all required features are present
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(float(features[feature_name]))
                else:
                    # Default values for missing features
                    default_values = {
                        'typing_speed': 0.0,
                        'accuracy': 100.0,
                        'session_duration': 1.0,
                        'word_count': 0.0,
                        'error_rate': 0.0,
                        'backspace_frequency': 0.0,
                        'pause_frequency': 0.0,
                        'text_complexity': 1.0
                    }
                    feature_vector.append(default_values.get(feature_name, 0.0))
            
            # Convert to numpy array and reshape
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale the features
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            # Return default feature vector
            return np.zeros((1, len(self.feature_names)))
    
    def predict_assistance_need(self, features):
        """Predict if the user needs assistance"""
        try:
            processed_features = self.preprocess_features(features)
            
            if self.model is None:
                # Fallback logic when model is not available
                return self.fallback_prediction(features)
            
            # Make prediction
            prediction = self.model.predict(processed_features, verbose=0)
            probability = float(prediction[0][0])
            needs_assistance = probability > 0.5
            
            return {
                'needs_assistance': needs_assistance,
                'confidence': probability,
                'features_used': processed_features.tolist()[0]
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return self.fallback_prediction(features)
    
    def fallback_prediction(self, features):
        """Fallback prediction logic when model is not available"""
        typing_speed = features.get('typing_speed', 0)
        accuracy = features.get('accuracy', 100)
        error_rate = features.get('error_rate', 0)
        
        # Simple rule-based logic
        needs_assistance = (
            typing_speed < 25 or  # Very slow typing
            accuracy < 80 or      # Low accuracy
            error_rate > 15       # High error rate
        )
        
        confidence = 0.7 if needs_assistance else 0.3
        
        return {
            'needs_assistance': needs_assistance,
            'confidence': confidence,
            'features_used': list(features.values())
        }

# Initialize the model
typing_model = TypingAssistantModel()

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'Virtual Typing Assistant API is running',
        'model_loaded': typing_model.model is not None,
        'scaler_loaded': typing_model.scaler is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_typing():
    """Analyze typing patterns and provide assistance recommendations"""
    try:
        # Get the features from the request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"Received features: {data}")
        
        # Make prediction
        prediction_result = typing_model.predict_assistance_need(data)
        needs_assistance = prediction_result['needs_assistance']
        confidence = prediction_result['confidence']
        
        # Generate response based on prediction
        response = {
            'needs_assistance': needs_assistance,
            'confidence': confidence,
            'message': '',
            'suggestions': [],
            'analysis': {
                'typing_speed': data.get('typing_speed', 0),
                'accuracy': data.get('accuracy', 100),
                'error_rate': data.get('error_rate', 0),
                'session_duration': data.get('session_duration', 0)
            }
        }
        
        # Generate personalized message and suggestions
        if needs_assistance:
            response['message'] = generate_assistance_message(data, confidence)
            response['suggestions'] = generate_suggestions(data)
        else:
            response['message'] = generate_positive_message(data, confidence)
            response['suggestions'] = generate_improvement_tips(data)
        
        logger.info(f"Analysis result: needs_assistance={needs_assistance}, confidence={confidence:.2f}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_typing: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

def generate_assistance_message(features, confidence):
    """Generate a personalized assistance message"""
    typing_speed = features.get('typing_speed', 0)
    accuracy = features.get('accuracy', 100)
    error_rate = features.get('error_rate', 0)
    
    messages = []
    
    if typing_speed < 20:
        messages.append("Your typing speed could benefit from practice.")
    elif typing_speed < 30:
        messages.append("You're making good progress with typing speed.")
    
    if accuracy < 70:
        messages.append("Focus on accuracy to build better typing habits.")
    elif accuracy < 85:
        messages.append("Your accuracy is improving but could be better.")
    
    if error_rate > 20:
        messages.append("Consider slowing down to reduce errors.")
    elif error_rate > 10:
        messages.append("Your error rate suggests room for improvement.")
    
    if not messages:
        messages.append("I've detected some areas where you could use assistance.")
    
    base_message = f"Based on my analysis (confidence: {confidence:.1%}), "
    return base_message + " ".join(messages)

def generate_positive_message(features, confidence):
    """Generate a positive reinforcement message"""
    typing_speed = features.get('typing_speed', 0)
    accuracy = features.get('accuracy', 100)
    
    messages = []
    
    if typing_speed >= 40:
        messages.append("Excellent typing speed!")
    elif typing_speed >= 30:
        messages.append("Good typing speed.")
    
    if accuracy >= 95:
        messages.append("Outstanding accuracy!")
    elif accuracy >= 90:
        messages.append("Great accuracy.")
    elif accuracy >= 85:
        messages.append("Good accuracy.")
    
    if not messages:
        messages.append("You're doing well overall.")
    
    base_message = f"Great job! (confidence: {confidence:.1%}) "
    return base_message + " ".join(messages)

def generate_suggestions(features):
    """Generate specific suggestions for improvement"""
    suggestions = []
    typing_speed = features.get('typing_speed', 0)
    accuracy = features.get('accuracy', 100)
    error_rate = features.get('error_rate', 0)
    backspace_freq = features.get('backspace_frequency', 0)
    pause_freq = features.get('pause_frequency', 0)
    
    # Speed-related suggestions
    if typing_speed < 20:
        suggestions.append("Practice basic finger positioning and muscle memory exercises")
        suggestions.append("Use online typing games to make practice more engaging")
    elif typing_speed < 35:
        suggestions.append("Focus on touch typing without looking at the keyboard")
        suggestions.append("Practice common word patterns and letter combinations")
    
    # Accuracy-related suggestions
    if accuracy < 80:
        suggestions.append("Slow down and focus on accuracy before speed")
        suggestions.append("Practice proper finger placement on home row keys")
    elif accuracy < 90:
        suggestions.append("Review frequently mistyped keys and practice them specifically")
    
    # Error rate suggestions
    if error_rate > 15:
        suggestions.append("Take breaks to avoid fatigue-related errors")
        suggestions.append("Practice difficult key combinations separately")
    
    # Backspace frequency suggestions
    if backspace_freq > 10:
        suggestions.append("Try to avoid excessive use of backspace - focus on getting it right the first time")
        suggestions.append("Practice proofreading skills to catch errors before making them")
    
    # Pause frequency suggestions
    if pause_freq > 15:
        suggestions.append("Work on maintaining consistent rhythm while typing")
        suggestions.append("Practice typing common words and phrases to build fluency")
    
    # General suggestions if no specific issues found
    if not suggestions:
        suggestions.append("Continue practicing regularly to maintain your skills")
        suggestions.append("Try typing different types of content to improve versatility")
    
    return suggestions[:5]  # Limit to top 5 suggestions

def generate_improvement_tips(features):
    """Generate general improvement tips for good typists"""
    tips = []
    typing_speed = features.get('typing_speed', 0)
    accuracy = features.get('accuracy', 100)
    
    if typing_speed >= 40 and accuracy >= 95:
        tips.extend([
            "Challenge yourself with complex texts and technical content",
            "Practice typing in different languages or coding languages",
            "Focus on maintaining speed during long typing sessions"
        ])
    elif typing_speed >= 30 and accuracy >= 90:
        tips.extend([
            "Work on increasing speed while maintaining your excellent accuracy",
            "Practice typing numbers and special characters",
            "Try typing tests with time pressure to build confidence"
        ])
    else:
        tips.extend([
            "Keep up the good work with regular practice",
            "Set small, achievable goals for speed and accuracy",
            "Monitor your progress over time"
        ])
    
    # Add general tips
    tips.extend([
        "Maintain good posture and ergonomic setup",
        "Take regular breaks to prevent strain",
        "Stay hydrated and comfortable while typing"
    ])
    
    return tips[:4]  # Limit to top 4 tips

@app.route('/progress', methods=['POST'])
def track_progress():
    """Track typing progress over time"""
    try:
        data = request.get_json()
        
        if not data or 'sessions' not in data:
            return jsonify({'error': 'Session data required'}), 400
        
        sessions = data['sessions']
        
        if len(sessions) < 2:
            return jsonify({'error': 'At least 2 sessions required for progress analysis'}), 400
        
        # Calculate progress metrics
        progress_analysis = analyze_progress(sessions)
        
        return jsonify(progress_analysis)
        
    except Exception as e:
        logger.error(f"Error in track_progress: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

def analyze_progress(sessions):
    """Analyze progress across multiple typing sessions"""
    # Extract metrics from sessions
    speeds = [session.get('typing_speed', 0) for session in sessions]
    accuracies = [session.get('accuracy', 100) for session in sessions]
    error_rates = [session.get('error_rate', 0) for session in sessions]
    
    # Calculate trends
    speed_trend = calculate_trend(speeds)
    accuracy_trend = calculate_trend(accuracies)
    error_trend = calculate_trend(error_rates)
    
    # Generate progress summary
    progress_summary = {
        'total_sessions': len(sessions),
        'metrics': {
            'typing_speed': {
                'current': speeds[-1] if speeds else 0,
                'average': sum(speeds) / len(speeds) if speeds else 0,
                'best': max(speeds) if speeds else 0,
                'trend': speed_trend,
                'improvement': speeds[-1] - speeds[0] if len(speeds) > 1 else 0
            },
            'accuracy': {
                'current': accuracies[-1] if accuracies else 0,
                'average': sum(accuracies) / len(accuracies) if accuracies else 0,
                'best': max(accuracies) if accuracies else 0,
                'trend': accuracy_trend,
                'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
            },
            'error_rate': {
                'current': error_rates[-1] if error_rates else 0,
                'average': sum(error_rates) / len(error_rates) if error_rates else 0,
                'best': min(error_rates) if error_rates else 0,
                'trend': error_trend,
                'improvement': error_rates[0] - error_rates[-1] if len(error_rates) > 1 else 0  # Lower is better
            }
        },
        'insights': generate_progress_insights(speeds, accuracies, error_rates),
        'recommendations': generate_progress_recommendations(speed_trend, accuracy_trend, error_trend)
    }
    
    return progress_summary

def calculate_trend(values):
    """Calculate trend direction for a list of values"""
    if len(values) < 2:
        return 'insufficient_data'
    
    # Simple linear trend calculation
    n = len(values)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 'stable'
    
    slope = numerator / denominator
    
    if slope > 0.1:
        return 'improving'
    elif slope < -0.1:
        return 'declining'
    else:
        return 'stable'

def generate_progress_insights(speeds, accuracies, error_rates):
    """Generate insights based on progress data"""
    insights = []
    
    if len(speeds) >= 3:
        recent_speeds = speeds[-3:]
        if all(recent_speeds[i] >= recent_speeds[i-1] for i in range(1, len(recent_speeds))):
            insights.append("Your typing speed has been consistently improving in recent sessions!")
    
    if len(accuracies) >= 3:
        recent_accuracies = accuracies[-3:]
        if all(acc >= 90 for acc in recent_accuracies):
            insights.append("You've maintained excellent accuracy across recent sessions.")
    
    if speeds and max(speeds) - min(speeds) > 10:
        insights.append("Your typing speed shows good variation - consistency could be improved.")
    
    if not insights:
        insights.append("Keep practicing regularly to see more detailed progress insights.")
    
    return insights

def generate_progress_recommendations(speed_trend, accuracy_trend, error_trend):
    """Generate recommendations based on progress trends"""
    recommendations = []
    
    if speed_trend == 'declining':
        recommendations.append("Focus on rebuilding typing speed with regular practice sessions")
    elif speed_trend == 'stable':
        recommendations.append("Try challenging yourself with more complex texts to improve speed")
    
    if accuracy_trend == 'declining':
        recommendations.append("Slow down and focus on accuracy before working on speed")
    elif accuracy_trend == 'stable' and error_trend == 'stable':
        recommendations.append("Your accuracy is consistent - great foundation for speed improvement")
    
    if error_trend == 'improving':
        recommendations.append("Excellent error reduction! Continue this pattern")
    
    if not recommendations:
        recommendations.append("Continue your current practice routine - you're making good progress")
    
    return recommendations

@app.route('/exercises', methods=['GET'])
def get_exercises():
    """Get personalized typing exercises"""
    try:
        # Get optional parameters
        difficulty = request.args.get('difficulty', 'medium')
        focus_area = request.args.get('focus', 'general')
        
        exercises = generate_typing_exercises(difficulty, focus_area)
        
        return jsonify({
            'exercises': exercises,
            'difficulty': difficulty,
            'focus_area': focus_area
        })
        
    except Exception as e:
        logger.error(f"Error in get_exercises: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

def generate_typing_exercises(difficulty='medium', focus_area='general'):
    """Generate typing exercises based on difficulty and focus area"""
    exercises = []
    
    if focus_area == 'speed':
        exercises = [
            {
                'title': 'Common Words Speed Test',
                'text': 'the and for are but not you all can had her was one our out day get has him his how its may new now old see two who boy did its let put say she too use',
                'type': 'speed_drill',
                'target_wpm': 35 if difficulty == 'easy' else 50 if difficulty == 'medium' else 65,
                'duration_minutes': 2
            },
            {
                'title': 'Short Sentences',
                'text': 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!',
                'type': 'sentence_practice',
                'target_wpm': 30 if difficulty == 'easy' else 45 if difficulty == 'medium' else 60,
                'duration_minutes': 3
            }
        ]
    elif focus_area == 'accuracy':
        exercises = [
            {
                'title': 'Precision Practice',
                'text': 'carefully precisely accurately perfectly correctly exactly properly specifically particularly thoroughly systematically methodically',
                'type': 'accuracy_drill',
                'target_accuracy': 95 if difficulty == 'easy' else 98 if difficulty == 'medium' else 99,
                'duration_minutes': 5
            },
            {
                'title': 'Tricky Letter Combinations',
                'text': 'geography rhythm psychology synchronize extraordinary miscellaneous conscientious questionnaire bureaucracy surveillance',
                'type': 'difficult_words',
                'target_accuracy': 90 if difficulty == 'easy' else 95 if difficulty == 'medium' else 98,
                'duration_minutes': 4
            }
        ]
    elif focus_area == 'numbers':
        exercises = [
            {
                'title': 'Number Practice',
                'text': '1234567890 0987654321 1357924680 2468013579 9876543210 1122334455 9988776655 1029384756',
                'type': 'number_drill',
                'target_wpm': 25 if difficulty == 'easy' else 35 if difficulty == 'medium' else 50,
                'duration_minutes': 3
            }
        ]
    else:  # general
        exercises = [
            {
                'title': 'Balanced Practice',
                'text': 'Technology continues to evolve at an unprecedented pace, transforming the way we work, communicate, and live our daily lives. From artificial intelligence to renewable energy, innovation drives progress across all sectors.',
                'type': 'general_practice',
                'target_wpm': 35 if difficulty == 'easy' else 50 if difficulty == 'medium' else 65,
                'target_accuracy': 90 if difficulty == 'easy' else 95 if difficulty == 'medium' else 98,
                'duration_minutes': 5
            },
            {
                'title': 'Mixed Content',
                'text': 'The project deadline is June 15, 2024. Contact john.doe@email.com or call (555) 123-4567 for details. Budget: $15,000.',
                'type': 'mixed_content',
                'target_wpm': 30 if difficulty == 'easy' else 45 if difficulty == 'medium' else 60,
                'target_accuracy': 92 if difficulty == 'easy' else 96 if difficulty == 'medium' else 99,
                'duration_minutes': 4
            }
        ]
    
    return exercises

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port} with debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)