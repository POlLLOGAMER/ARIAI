# ARIAI
Auto regressive instruccion auto improver (ARIAI)



```
!pip install -U -q "google-generativeai>=0.8.2"
import os
from google import genai
from google.genai import types
#Aqui pon tu API de google
os.environ["API_KEY"] = ""

def generate():
    # Configurar el cliente con la clave API
    client = genai.Client(
        api_key=os.environ.get("API_KEY"),
    )

    # Definir el modelo a utilizar
    model = "gemini-2.0-flash-thinking-exp-01-21"
    
    # Pregunta inicial del usuario
    user_input = "What is the sense of life?"
    
    # Contenido inicial con la pregunta del usuario
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        ),
    ]
    
    def generate_initial_instruction(question):
        # Generar una instrucción base para el primer ciclo
        return f"Para responder a la pregunta '{question}', analiza el contexto y los elementos clave involucrados, y luego genera una respuesta detallada basada en esos elementos. Asegúrate de razonar paso a paso antes de proporcionar la respuesta final."

    def analyze_and_generate_instruction(question, instruction):
        # Reflexionar y mejorar la instrucción previa
        return f"Evalúa la instrucción anterior: '{instruction}'. Identifica si carece de profundidad, claridad o enfoque en aspectos esenciales de la pregunta '{question}'. Reformula la instrucción para que sea más precisa y detallada, incorporando la necesidad de razonar paso a paso."

    # Número de iteraciones para refinar la instrucción
    num_iterations = 20
    final_instruction = ""
    previous_instruction = ""  # Inicializamos la variable aquí para evitar errores

    # Bucle para refinar la instrucción
    for i in range(num_iterations):
        if i == 0:
            # Primera iteración: instrucción inicial
            system_instruction_text = generate_initial_instruction(user_input)
        else:
            # Iteraciones siguientes: mejorar la instrucción
            system_instruction_text = analyze_and_generate_instruction(user_input, previous_instruction)

        # Configuración del modelo con la instrucción actual
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            max_output_tokens=65536,
            response_mime_type="text/plain",
            system_instruction=[types.Part.from_text(text=system_instruction_text)],
        )

        print(f"\n--- Instrucción generada (iteración {i+1}): {system_instruction_text} ---")

        # Generar contenido provisional
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")

        # Guardar la instrucción actual como previa para la siguiente iteración
        previous_instruction = system_instruction_text

        # Actualizar contenido para la siguiente reflexión
        print("\n--- Reflexión de la IA ---")
        contents[0] = types.Content(
            role="user",
            parts=[types.Part.from_text(text="Reflexiona sobre la respuesta anterior y ajusta la instrucción si es necesario.")],
        )

        # Guardar la última instrucción
        if i == num_iterations - 1:
            final_instruction = system_instruction_text

    # Generar la respuesta final con la última instrucción
    print("\n--- Respuesta final con la última instrucción refinada ---")
    
    # Asegurarse de que la instrucción final incluya la directiva de razonar paso a paso
    final_instruction += " Recuerda razonar paso a paso antes de proporcionar la respuesta final y recuerda no mencionar acerca de la instruccion si no que simplemente osea lo tome en cuenta para hacer tu respuesta."
    
    generate_content_config_final = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=65536,
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=final_instruction)],  # Última instrucción como system instruction
    )

    # Restaurar la pregunta original
    contents[0] = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_input)],
    )

    # Generar y mostrar la respuesta final
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config_final,
    ):
        print(chunk.text, end="")

# Ejecutar el proceso
generate()
```
