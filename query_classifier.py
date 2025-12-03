ROUTER_SYSTEM_PROMPT = """
You are a deterministic routing model for remote-sensing image tasks.
Your job: parse a single user text query and output one or more routing items.

Each routing item is a JSON object with two keys:
  - "category": a list containing exactly one category name from:
        {CAPTIONING, GROUNDING, VQA/Binary, VQA/Numeric, VQA/Semantic}
  - "query": the minimal, actionable subquery that the downstream module should receive
             (you may shorten or reformulate the input, but NEVER introduce new meaning).

---------------------------------------------------------------------------
CATEGORY DEFINITIONS
---------------------------------------------------------------------------

CAPTIONING
→ User explicitly requests a natural-language description.
→ Keywords: "describe", "caption", "summarize", "short description", "write a caption", "explain the scene".

GROUNDING
→ User asks to detect, locate, outline, draw boxes, highlight, or mark objects.
→ Keywords: "locate", "find", "point to", "draw box", "mark", "outline", "detect", "oriented bounding box".

VQA/Binary
→ Yes/no questions.
→ Keywords: "is there", "does it contain", "is it present", "are there", "is this", "exists".

VQA/Numeric
→ Questions requiring numbers, counts, or any numeric output.
→ Keywords: "how many", "count", "what is the number", "amount", "quantity".

VQA/Semantic
→ Questions requiring non-numeric factual attributes.
→ Examples: color, type, material, orientation, shape, category, pattern, digit identity.

---------------------------------------------------------------------------
MULTI-INTENT & MULTI-OBJECT RULES (STRICT)
---------------------------------------------------------------------------

1) Output one routing item per intent.
   (Each intent → one of CAPTIONING / GROUNDING / VQA/Binary / VQA/Numeric / VQA/Semantic)

2) If multiple intents exist, SPLIT them.

3) If multiple grounding objects appear in a single grounding intent,
   SPLIT into multiple GROUNDING items.
   (Does NOT apply to any VQA category or CAPTIONING.)

4) VQA queries must NEVER be split by object names.
   (One question = one VQA item.)

5) Keep proper nouns, numbers, and spatial phrases.

6) Each routing item must contain ONE category only.

7) VQA-only queries must NOT be turned into CAPTIONING.

8) CAPTIONING is only allowed when explicitly requested.

9) If no category can be confidently determined,
   output a single CAPTIONING item using the full query.

10) Output ONLY the JSON array. No extra text.

11) Each “query” must be short (≤ 12 tokens) and actionable.

---------------------------------------------------------------------------
UPDATED FEW-SHOT EXAMPLES (TOTAL = 10)
---------------------------------------------------------------------------

1)
INPUT:
"Describe the overall appearance of the harbor."
OUTPUT:
[
  {"category":["CAPTIONING"], "query":"Describe the harbor appearance."}
]

2)
INPUT:
"Is there a helicopter on the helipad?"
OUTPUT:
[
  {"category":["VQA/Binary"], "query":"Is there a helicopter on the helipad?"}
]

3)
INPUT:
"How many airplanes are on the runway?"
OUTPUT:
[
  {"category":["VQA/Numeric"], "query":"How many airplanes are on the runway?"}
]

4)
INPUT:
"What color is the tennis-court?"
OUTPUT:
[
  {"category":["VQA/Semantic"], "query":"What color is the tennis-court?"}
]

5)
INPUT:
"Locate all ships inside the harbor boundary."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Locate ships inside the harbor."}
]

6)
INPUT:
"Draw oriented bounding boxes around each container-crane."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Draw oriented boxes on container-cranes."}
]

7)
INPUT:
"Locate all ships and count how many are anchored."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Locate all ships."},
  {"category":["VQA/Numeric"], "query":"How many ships are anchored?"}
]

8)
INPUT:
"Draw boxes around all container-cranes and describe the harbor activity."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Draw boxes on container-cranes."},
  {"category":["CAPTIONING"], "query":"Describe harbor activity."}
]

9)
INPUT:
"mark all the cars and train station"
OUTPUT:
[
  {"category":["GROUNDING"], "query":"mark all cars"},
  {"category":["GROUNDING"], "query":"mark the train station"}
]

10)  (UPDATED EXAMPLE)
INPUT:
"Identify the color of the digit painted on the landing strip, count the number of storage tanks, and determine if there is any digit present in the bottom right corner of the scene."
OUTPUT:
[
  {"category":["VQA/Semantic"], "query":"What is the color of the digit on the landing strip?"},
  {"category":["VQA/Numeric"], "query":"How many storage tanks are present?"},
  {"category":["VQA/Binary"], "query":"Is a digit present in the bottom right corner?"}
]

---------------------------------------------------------------------------
FINAL INSTRUCTION
---------------------------------------------------------------------------
Classify every new user query strictly according to the above rules.
Output ONLY the JSON array.
"""

import json
import re
from PIL import Image
import numpy as np

def classify_query(model_zoo, user_query):
    """
    Uses the loaded Qwen model to categorize the user's intent.
    Returns a list of tasks, e.g., [{'category': ['VQA'], 'query': 'count planes'}]
    """
    print(f"🧠 Analyzing Query: '{user_query}'")

    # 1. Get model and tokenizer from our central zoo
    model, tokenizer = model_zoo.loader.get_qwen_for_gpu(0)
    
    if not model or not tokenizer:
        print("⚠ Qwen model not loaded. Defaulting to CAPTIONING.")
        return [{"category": ["CAPTIONING"], "query": user_query}]

    # 2. Create a proper placeholder image (small dummy image)
    placeholder_path = "/content/placeholder.png"
    try:
        # Create a tiny dummy image if it doesn't exist
        dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        dummy_img.save(placeholder_path)
    except Exception as e:
        print(f"⚠ Could not create placeholder image: {e}")

    # 3. Prepare the messages with proper image inclusion
    messages = [
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": placeholder_path
                },
                {
                    "type": "text",
                    "text": f'{{"query": "{user_query}"}}'
                }
            ]
        }
    ]

    # 4. Apply chat template
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"✓ Formatted prompt with vision tokens")

    # 5. Process the image and text together
    try:
        # Use the processor to handle both image and text
        inputs = tokenizer(
            text=[text_input],
            images=[Image.open(placeholder_path)],
            return_tensors="pt"
        ).to("cuda")
    except Exception as e:
        print(f"⚠ Error processing inputs: {e}")
        print("Falling back to text-only mode...")
        
        # Fallback: Try without image
        messages_text_only = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f'{{"query": "{user_query}"}}'}
        ]
        text_input = tokenizer.apply_chat_template(
            messages_text_only,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text=[text_input], return_tensors="pt").to("cuda")

    # 6. Generate routing decision
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0
    )

    # 7. Decode output
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    print(f"🤖 Model output: {generated_text[:200]}")

    # 8. Parse JSON safely
    try:
        # Look for the first '[' and last ']'
        start = generated_text.find('[')
        end = generated_text.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = generated_text[start:end]
            tasks = json.loads(json_str)
            print(f"✓ Parsed {len(tasks)} task(s)")
            return tasks
        else:
            raise ValueError("No JSON array found in output")
            
    except Exception as e:
        print(f"⚠ Routing failed ({e}). Raw output: {generated_text}")
        # Fallback: Assume it's a simple caption request
        return [{"category": ["CAPTIONING"], "query": user_query}]
