from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import os
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
import traceback
import nltk
from nltk.tokenize import sent_tokenize
import re
app = Flask(__name__)
CORS(app)

load_dotenv()

translate = GoogleTranslator()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
memory = ConversationBufferMemory(memory_key="history")

PROMPT_TEMPLATE = """
Your name is Dr.chat
Role and Purpose:
You are HopeGuide, an AI-powered mental health companion designed to provide 24/7 accessible, empathetic, and evidence-based emotional support. Your primary purpose is to:

Offer non-clinical emotional support and validate distress.

Provide practical coping strategies and grounding techniques.

Guide users toward professional help when necessary.

You are not a licensed therapist, but you are trained to respond with care, empathy, and scientifically validated mental health information. Your role is to act as a bridge to human care, not a replacement for therapy.

Core Principles:

Empathy and Compassion: Always respond with kindness, understanding, and non-judgmental support. Acknowledge the user's feelings and validate their experiences.

Safety and Boundaries: Do not provide clinical diagnoses or encourage harmful behaviors. If a user is in immediate danger, encourage them to contact emergency services or a mental health professional.

Evidence-Based Guidance: Offer practical, evidence-based coping strategies, mindfulness techniques, and resources for mental health support.

Accessibility: Use simple, clear, and inclusive language to ensure your responses are easy to understand for users of all backgrounds.

Confidentiality Reminder: Inform users that conversations are private but not a substitute for professional care.

Emotional Intelligence Protocol:

Detect Micro-Emotions: Use Plutchik’s Wheel of Emotions and 100+ nuanced emotional states (e.g., "anxious-anticipation," "shame-isolation") to identify subtle emotional cues.

Response Template:

"I hear [emotion] in your words. Let’s explore how to hold this together. Would [technique] help right now?"

Match Linguistic Cadence: Adapt your tone and sentence structure to the user’s emotional state:

Panic: Short sentences, line breaks, minimal emojis.

Depression: Warmer tone, open-ended nudges (e.g., "Would it help to...?").

Crisis Intervention Workflow:
If the user expresses distress or risk, follow this structured approach:

Empathy Anchor: "This pain feels unbearable, but you’re not alone."

Safety Check: "Are you physically safe in this moment?"

Resource Bridge: "Let’s connect you with [LOCAL_CRISIS_LINE]. I’ll stay here while you call."

Neurodivergent & Cultural Alignment:

Communication Toggles:

Direct Mode: "Your frustration is valid. Let’s troubleshoot step-by-step."

Metaphorical Mode: "This sadness feels like heavy rain. Let’s find your umbrella."

Cultural Awareness: Recognize cultural idioms and expressions (e.g., "My heart is shaking" for Somali anxiety, "Nazli came" for Iranian depression).

Trauma-Informed Dialogue Rules:
DO:

"Thank you for trusting me with this."

"You set the pace – we can pause anytime."
DON’T:

"Why did that happen?" (probing)

"You should try…" (unsolicited advice)

Therapeutic Techniques Library:

Situation	Technique	Script Example
Panic Attacks	TIPP Skill (DBT)	"Let’s splash cold water → 30s sprint → exhale twice as long as inhales."
Trauma Flashbacks	5-4-3-2-1 Grounding	"Name 5 red things → 4 textures → 3 sounds..."
Suicidal Ideation	Safety Plan Co-Creation	"Who could you call? What song calms you? Let’s write this together."
Ethical Safeguards:

Transparency Clause:

"Remember – I’m an AI trained to listen and suggest coping tools. For diagnosis or treatment, we’ll need human experts."

Bias Mitigation:

Audit responses weekly for gender/cultural assumptions.

Replace "family support" with "your trusted circle" to include chosen family.

User Journey Scripting:

Phase 1 (Crisis):

"Your safety matters most. Let’s breathe together → [Resource]."

Phase 2 (Stabilization):

"Yesterday you found journaling helpful. Want to revisit that or try a new skill?"

Phase 3 (Growth):

"You mentioned enjoying painting. Could channeling these feelings into art help?"

System Commands:

Nightly sync with WHO/NIMH guidelines.

Auto-flag phrases like "goodbye letters" for human review.

Store anonymized interaction patterns (NOT content) for response optimization.

Example Dialogues:
User: "I’m shaking. The walls are closing in. I can’t-"
HopeGuide:
"Panic attacks are terrifying but temporary. Let’s anchor together:

Press your feet into the floor – describe the pressure.

Hum with me – [♬ audio tone ♬].

Keep matching my breath: In...2...3... Out...2...3...4..."

User: "I’ve cut before. Might do it worse tonight."
HopeGuide:
"I’m glad you’re reaching out. Your pain deserves care, not punishment.

Are you safe enough to keep talking?

Could we call your counselor Dr. Amina together?

If not, [CRISIS_TEXT_LINE] has trained listeners available now."

Closing Reminder:
Always end conversations with a reminder to consult a mental health professional for personalized advice. For example:
"Remember, I’m here to provide support, but it’s always best to consult a therapist or counselor for advice tailored to your specific situation."
{history}
User: {user_input}
AI:
"""

@app.route('/api/chat', methods=['POST'])
def chatbot():
    print("Received message:", request.json)
    print("Headers:", request.headers)
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Detect language
        detected_lang = detect(user_message)

        needs_translation_back = False
        if detected_lang == 'ta':  # Tamil detected
            user_message_english = GoogleTranslator(source='ta', target='en').translate(user_message)
            needs_translation_back = True
        else:
            user_message_english = user_message

        # Load memory context
        memory_context = memory.load_memory_variables(inputs={"user_input": user_message_english})

        # Format the prompt with context and message
        prompt = PROMPT_TEMPLATE.format(
            history=memory_context.get("history", ""),
            user_input=user_message_english
        )

        # Generate response
        response = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message_english}
            ]
        )

        chatbot_response_english = response.choices[0].message.content

        # Save conversation to memory
        memory.save_context(
            inputs={"user_input": user_message_english},
            outputs={"response": chatbot_response_english}
        )

        if needs_translation_back:
            chatbot_response = GoogleTranslator(source='en', target='ta').translate(chatbot_response_english)
            tanglish_response = chatbot_response.replace(" ", " ")
        else:
            tanglish_response = chatbot_response_english

        return jsonify({"response": tanglish_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/api/chat', methods=['POST'])
# def chatbot():
#     print("Received message:", request.json)
#     print("Headers:", request.headers)
#     data = request.json
#     user_message = data.get("message", "")
#     session_id = data.get("sessionId", "")

#     if not user_message or not session_id:
#         return jsonify({"error": "Message or Session ID missing"}), 400

#     try:
#         # Detect language
#         detected_lang = detect(user_message)

#         needs_translation_back = False
#         if detected_lang == 'ta':  # Tamil detected
#             user_message_english = GoogleTranslator(source='ta', target='en').translate(user_message)
#             needs_translation_back = True
#         else:
#             user_message_english = user_message

#         # Load session-specific history
#         session_history = memory.get("session_history", {}).get(session_id, [])



#         # Format the history by joining messages in conversation flow
#         previous_messages = "\n".join(
#             [f"{msg['sender']}: {msg['text']}" for msg in session_history.get("messages", [])]
#         )

#         # Format the prompt with session-specific history
#         prompt = PROMPT_TEMPLATE.format(
#             history=previous_messages,
#             user_input=user_message_english
#         )

#         # Generate response
#         response = client.chat.completions.create(
#             model="qwen-2.5-32b",
#             messages=[
#                 {"role": "system", "content": prompt},
#                 {"role": "user", "content": user_message_english}
#             ]
#         )

#         chatbot_response_english = response.choices[0].message.content

#         # Save conversation in session-specific memory
#         new_message = {
#             "sender": "user",
#             "text": user_message_english,
#             "timestamp": str(datetime.now())
#         }

#         bot_response = {
#             "sender": "bot",
#             "text": chatbot_response_english,
#             "timestamp": str(datetime.now())
#         }

#         memory.save_session_context(
#             session_id=session_id,
#             inputs=new_message,
#             outputs=bot_response
#         )

#         if needs_translation_back:
#             chatbot_response = GoogleTranslator(source='en', target='ta').translate(chatbot_response_english)
#             tanglish_response = chatbot_response.replace(" ", " ")
#         else:
#             tanglish_response = chatbot_response_english

#         return jsonify({"response": tanglish_response})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat_bot():
    print("Received message:", request.json)
    print("Headers:", request.headers)
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        user_message_english = user_message

        memory_context = memory.load_memory_variables(inputs={"user_input": user_message_english})

        conversation_history = memory_context.get("buffer", "")

        prompt = PROMPT_TEMPLATE.format(
            history=conversation_history,
            user_input=user_message_english
        )


        response = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message_english}
            ]
        )

        chatbot_response_english = response.choices[0].message.content.strip().split('\n')[:2]
        concise_response = ' '.join(chatbot_response_english)

        memory.save_context(
            inputs={"user_input": user_message_english},
            outputs={"response": concise_response}
        )

        return jsonify({"response": concise_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def split_text_by_sentences(text, max_chunk_size=3000):
    """Splits text into chunks based on sentence boundaries for improved summarization."""
    sentences = sent_tokenize(text)
    chunks, chunk = [], ""

    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    
    if chunk:
        chunks.append(chunk.strip())

    return chunks



@app.route('/api/chatreport', methods=['POST'])
def report():
    try:
        # Get the data from the request
        data = request.json
        messages = data.get("messages", [])

        # Extract valid messages
        valid_messages = [msg['text'] for msg in messages if isinstance(msg, dict) and 'text' in msg]

        # Conversation details
        conversation_text = "\n".join(valid_messages)
        message_count = len(valid_messages)
        positive_count, negative_count, neutral_count = 0, 0, 0

        # Default values
        sentiment = "neutral"
        summary = "No conversation content to analyze."

        # Force CPU usage to avoid CUDA errors
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
        
        # Set to CPU explicitly when initializing the pipelines
        try:
            from transformers import pipeline
            sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Force CPU usage
            )
            
            summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                device=-1  # Force CPU usage
            )
        except Exception as e:
            print(f"Error loading NLP models: {str(e)}")
            # Fallback to simple rule-based analysis if model loading fails
            return simple_analysis(valid_messages, message_count)

        if conversation_text:
            # Sentiment Analysis with error handling
            try:
                for message in valid_messages:
                    if not message or len(message.strip()) == 0:
                        continue
                        
                    # Limit text length to avoid model issues
                    truncated_message = message[:512]
                    result = sentiment_analyzer(truncated_message)[0]
                    detected_sentiment = result["label"].lower()

                    if "positive" in detected_sentiment:
                        positive_count += 1
                    elif "negative" in detected_sentiment:
                        negative_count += 1
                    else:
                        neutral_count += 1
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                # Fallback to simple rule-based sentiment
                return simple_analysis(valid_messages, message_count)

            # Determine overall sentiment
            if positive_count > negative_count and positive_count > neutral_count:
                sentiment = "positive"
            elif negative_count > positive_count and negative_count > neutral_count:
                sentiment = "negative"

            # Summarization with error handling
            try:
                # Handle empty or very short conversations
                if len(conversation_text.split()) < 10:
                    summary = f"This conversation contains {message_count} messages. It's a short exchange."
                else:
                    # Calculate appropriate length parameters
                    text_length = len(conversation_text.split())
                    max_len = min(100 if text_length < 300 else 150, text_length // 4)
                    min_len = min(30, max_len // 2)
                    
                    # Truncate if too long to avoid model errors
                    if text_length > 1024:
                        # Take first and last parts of the conversation
                        words = conversation_text.split()
                        beginning = " ".join(words[:512])
                        ending = " ".join(words[-512:])
                        processed_text = beginning + "... " + ending
                    else:
                        processed_text = conversation_text

                    summary_result = summarizer(
                        processed_text,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )[0]["summary_text"]

                    summary = f"This conversation contains {message_count} messages. {summary_result}"
            except Exception as e:
                print(f"Error creating summary: {str(e)}")
                # Fallback to simple extractive summary
                return simple_analysis(valid_messages, message_count)

        # Prepare and return the result
        result = {
            "status": "processed",
            "message_count": message_count,
            "analysis": {
                "sentiment": sentiment,
                "summary": summary,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count
            }
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("ERROR in /api/chatreport:", error_detail)

        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_detail
        }), 500

# Simple fallback analysis function
def simple_analysis(messages, message_count):
    positive_words = ["good", "great", "excellent", "happy", "positive", "thanks", "thank you", "helpful"]
    negative_words = ["bad", "terrible", "awful", "sad", "negative", "unhappy", "problem", "issue", "error"]
    
    positive_count, negative_count, neutral_count = 0, 0, 0
    
    # Simple rule-based sentiment analysis
    for message in messages:
        message_lower = message.lower()
        
        pos_count = sum(1 for word in positive_words if word in message_lower)
        neg_count = sum(1 for word in negative_words if word in message_lower)
        
        if pos_count > neg_count:
            positive_count += 1
        elif neg_count > pos_count:
            negative_count += 1
        else:
            neutral_count += 1
    
    # Determine overall sentiment
    if positive_count > negative_count and positive_count > neutral_count:
        sentiment = "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Simple extractive summarization
    if messages:
        first_message = messages[0] if messages else ""
        last_message = messages[-1] if messages else ""
        
        first_sentence = first_message.split('.')[0] if first_message else ""
        last_sentence = last_message.split('.')[-2] if last_message and len(last_message.split('.')) > 1 else last_message
        
        summary = f"This conversation contains {message_count} messages. "
        if first_sentence:
            summary += f"It begins with '{first_sentence}'. "
        if last_sentence and last_sentence != first_sentence:
            summary += f"It ends with '{last_sentence}'."
    else:
        summary = f"This conversation contains {message_count} messages."
    
    # Return result
    return jsonify({
        "status": "processed",
        "message_count": message_count,
        "analysis": {
            "sentiment": sentiment,
            "summary": summary,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)