# ARIAI
Auto regressive instruccion auto improver (ARIAI)



```
!pip install -U -q "google-generativeai>=0.8.2"
import os
from google import genai
from google.genai import types
#Here put your google API
os.environ["API_KEY"] = ""

def generate():
    # Set up the client with the API key
    client = genai.Client(
        api_key=os.environ.get("API_KEY"),
    )

    # Define the model to be used
    model = "gemini-2.0-flash-thinking-exp-01-21"
    
    # Initial user question
    user_input = "What is the sense of life?"
    
    # Initial content with the user's question
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        ),
    ]
    
    def generate_initial_instruction(question):
        # Generate a base instruction for the first cycle
        return f"To answer the question '{question}', analyze the context and the key elements involved, and then generate a detailed response based on those elements. Make sure to reason step by step before providing the final answer."

    def analyze_and_generate_instruction(question, instruction):
        # Reflect and improve the previous instruction
        return f"Evaluate the previous instruction: '{instruction}'. Identify if it lacks depth, clarity, or focus on essential aspects of the question '{question}'. Reformulate the instruction to make it more precise and detailed, incorporating the need to reason step by step."

    # Number of iterations to refine the instruction
    num_iterations = 20
    final_instruction = ""
    previous_instruction = ""  # Initialize the variable here to avoid errors

    # Loop to refine the instruction
    for i in range(num_iterations):
        if i == 0:
            # First iteration: initial instruction
            system_instruction_text = generate_initial_instruction(user_input)
        else:
            # Subsequent iterations: improve the instruction
            system_instruction_text = analyze_and_generate_instruction(user_input, previous_instruction)

        # Configure the model with the current instruction
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            max_output_tokens=65536,
            response_mime_type="text/plain",
            system_instruction=[types.Part.from_text(text=system_instruction_text)],
        )

        print(f"\n--- Generated instruction (iteration {i+1}): {system_instruction_text} ---")

        # Generate provisional content
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")

        # Save the current instruction as previous for the next iteration
        previous_instruction = system_instruction_text

        # Update content for the next reflection
        print("\n--- AI Reflection ---")
        contents[0] = types.Content(
            role="user",
            parts=[types.Part.from_text(text="Reflect on the previous response and adjust the instruction if necessary.")],
        )

        # Save the last instruction
        if i == num_iterations - 1:
            final_instruction = system_instruction_text

    # Generate the final answer with the last refined instruction
    print("\n--- Final answer with the last refined instruction ---")
    
    # Ensure that the final instruction includes the step-by-step reasoning directive
    final_instruction += " Remember to reason step by step before providing the final answer and remember not to mention the instruction, just take it into account to make your answer."

    generate_content_config_final = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=65536,
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=final_instruction)],  # Last instruction as system instruction
    )

    # Restore the original question
    contents[0] = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_input)],
    )

    # Generate and display the final response
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config_final,
    ):
        print(chunk.text, end="")

# Run the process
generate()

```
