from fireworks.client import Fireworks
from flask import Flask, render_template, request, Response, stream_with_context
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Initialize Fireworks client
client = Fireworks(api_key=os.getenv("FIREWORK_API_KEY"))

def get_model_response(prompt: str):
    """Use streaming chat completions from Fireworks AI."""
    return client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-r1",  # or whichever is appropriate
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that specializes in coding.\n"
                    "Always provide a chain of thought in the block '[reasoning]...[/reasoning]', "
                    "followed by the final answer in '[answer]...[/answer]'."
                    "Example:\n"
                    "[reasoning]\nStep by step logic...\n[/reasoning]\n"
                    "[answer]\nFinal short answer.\n[/answer]"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        stream=True
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    prompt = request.args.get('prompt', '')

    def generate_stream():
        try:
            # Stream text from Fireworks
            for chunk in get_model_response(prompt):
                content = chunk.choices[0].delta.content
                if content:
                    # Make sure to properly escape newlines for SSE
                    content = content.replace('\n', '\\n')
                    yield f"data: {content}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    # Return an SSE streaming response
    return Response(
        stream_with_context(generate_stream()),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

if __name__ == '__main__':
    app.run(debug=True) 