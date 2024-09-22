import openai
import json
import time

# Load the API key safely
with open('API_KEY.txt', 'r') as key_file:
    API_KEY = key_file.read().strip()

openai.api_key = API_KEY

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# Sample JSON structure (you would load your actual record.json in practice)
record = {
    "exercise": "shoulder raises",
    "correct_reps": 10,
    "incorrect_reps": 2,
    "weight": 10
}

# If the record is empty, generate a beginner's workout plan
if not record:
    record = {
        "exercise": "Beginner Workout",
        "correct_reps": 0,
        "incorrect_reps": 0,
        "weight": 0
    }

# Create the prompt
prompt = f"""
Your task is to create a specialized workout plan based on a user's performance data, provided in a JSON format.
Here is the data:
{json.dumps(record, indent=4)}
"""

def get_openai_response():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant helping a user create a workout plan."},
                {"role": "user", "content": prompt}
            ]
        )
        return response
    except openai.error.RateLimitError:
        print("Rate limit reached. Retrying after 30 seconds...")
        time.sleep(30)  # wait 30 seconds and retry
        return get_openai_response()
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request: {e}")
        return None
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Get and print the response from the API
response = get_openai_response()
if response:
    print(response.choices[0].message['content'])
else:
    print("Failed to get a response.")
