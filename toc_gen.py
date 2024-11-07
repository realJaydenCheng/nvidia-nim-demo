
import gradio as gr
from openai import OpenAI

from doc_rag import IngestedDoc
from __env import NVIDIA_API_KEY
from io import BytesIO

CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

BASE_TEMPLATE = """Please generate a {layer}-level outline based on the topic and reference: '{idea}' (suitable for speech inspiration, article writing, PowerPoint presentations, etc.):\n"""

FIRST_LEVEL_EXAMPLES = """For the topic:
Technology is a double-edged sword; it can drive society forward while also causing harm through misuse, such as the digital divide.
Generate the first level of the outline, one item per line, less than 5 items, enclosed in <>, no additional explanation or content required:
<The Age of Information Overload>
<Technological Revolution and Industrial Transformation>
<Social Impact Becoming More Prominent>
<Precautionary Measures for Human Beings to Maximize Benefits and Minimize Risks>
<Trends and Prospects, Conclusion>

For the topic:
Why should we study hard? In a rapidly changing world, continuous learning is key to realizing personal value and making social contributions.
Generate the first level of the outline, one item per line, less than 5 items, enclosed in <>, no additional explanation or content required:
<Introduction>
<Significance of Learning for Personal Growth>
<Responsibility of Learning to Society>
<How to Learn Effectively>

For the topic:
In the golden age of being single, how do we face love: be brave enough to pursue love.
Generate the first level of the outline, one item per line, less than 5 items, enclosed in <>, no additional explanation or content required:
<What is the Golden Age of Being Single?>
<The Meaning of Love to People>
<More Complex Survival and Situations Today>
<Warmest Things Are Feelings Between People>
<Be Brave Enough to Pursue Love>
"""

FIRST_LEVEL_TEMPLATE = """{few_shot}
The above are examples of generating the first level of an outline. Please generate the first level of the outline based on the following topic and reference:
{idea}
Generate one item per line immediately, enclosed in <>, less than 5 items, no prefix and head, no additional explanation or content required:"""

NEXT_LEVEL_EXAMPLES = """For the topic:
Technology is a double-edged sword; it can drive society forward while also causing harm through misuse, such as the digital divide.
Do not repeat from the previous layer: The Age of Information Overload, Technological Revolution and Industrial Transformation, Precautionary Measures for Human Beings to Maximize Benefits and Minimize Risks, Trends and Prospects, Conclusion
Generate the second level of the outline under "Social Impact Becoming More Prominent", one item per line, less than 5 items, enclosed in <>, no additional explanation or content required:
<Example: Digital Divide Phenomenon>
<Various Issues Are Widespread>
<Deep Impact on Various Aspects of Human Life>

For the topic:
In the golden age of being single, how do we face love: be brave enough to pursue love.
Do not repeat from the previous layer: What is the Golden Age of Being Single?, The Meaning of Love to People, More Complex Survival and Situations Today, Love Muscle Weakness, Warmest Things Are Feelings Between People
Generate the second level of the outline under "Be Brave Enough to Pursue Love", one item per line, less than 5 items, enclosed in <>, no additional explanation or content required:
<Set Clear Goals>
<Enhance Mental Strength>
<Pursue Legally and Ethically>

For the topic:
Why should we study hard? In a rapidly changing world, continuous learning is key to realizing personal value and making social contributions.
Do not repeat from the previous layer: Understanding Rights and Obligations, Correct Values and Social Responsibility
Generate the third level of the outline under "Knowledge Is the Key Force Driving Social Development", one item per line, less than 5 items, enclosed in <>, no additional explanation or content required:
<Innovation in Technology and Economic Growth>
<Governance and Public Policy>
<Cultural Heritage and Development>
"""

NEXT_LEVEL_TEMPLATE = """{few_shot}
The above are examples of generating specific levels of an outline. Please generate the level {current_level} content under "{previous_outline}" based on the following topic and reference:
{idea}
Do not repeat from the previous layer: {formatted_previous_list}
Generate one item per line immediately, enclosed in <>, less than 5 items, no prefix and head, no additional explanation or content required:
"""

AVAILABLE_MODELS = (
    "mistralai/mistral-7b-instruct-v0.2",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "qwen/qwen2-7b-instruct",
)


def request_model_for_one_response(
    model_name: str,
    prompt: str,
):
    response = CLIENT.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def generate_outline(
    input_text: str,
    layer: int,
    model_name: str,
):
    prompt = BASE_TEMPLATE.format(idea=input_text, layer=layer)
    return request_model_for_one_response(model_name, prompt)


def extract_outlines(model_output: str):
    return [
        line.strip().strip('<').strip('>')
        for line in model_output.split('\n')
        if line.strip() and "<" in line and ">" in line
    ]


def generate_first_level_outlines(idea: str, model_name: str):
    first_level_prompt = FIRST_LEVEL_TEMPLATE.format(
        few_shot=FIRST_LEVEL_EXAMPLES,
        idea=idea,
    )
    return extract_outlines(
        request_model_for_one_response(model_name, first_level_prompt)
    )


def generate_next_level_outline(
    idea: str,
    current_level: int,
    previous_outline: str,
    previous_list: list[str],
    target_layer: int,
    model_name: str,
) -> list[str] | dict:
    formatted_previous_list = ', '.join(previous_list)
    next_level_prompt = NEXT_LEVEL_TEMPLATE.format(
        few_shot=NEXT_LEVEL_EXAMPLES,
        idea=idea,
        formatted_previous_list=formatted_previous_list,
        previous_outline=previous_outline,
        current_level=current_level,
    )
    outlines = extract_outlines(
        request_model_for_one_response(model_name, next_level_prompt)
    )

    if target_layer == current_level:
        return outlines
    else:
        return {
            text: generate_next_level_outline(
                idea, current_level + 1, text,
                outlines, target_layer, model_name
            ) for text in outlines
        }


def recursive_outline(idea: str, target_layer: int, model_name: str,):
    first_level = generate_first_level_outlines(idea, model_name)
    return {
        text: generate_next_level_outline(
            idea, 2, text, first_level, target_layer, model_name
        ) for text in first_level
    }


def dict_to_markdown(data: dict, indent=0):
    markdown_lines = []
    for key, value in data.items():
        markdown_lines.append('  ' * indent + f'- {key}')
        if isinstance(value, dict):
            markdown_lines.append(dict_to_markdown(value, indent + 1))
        elif isinstance(value, list):
            markdown_lines.extend([
                '  ' * (indent + 1) + f'- {item}'
                for item in value
            ])
    return '\n'.join(markdown_lines)


def compare_outlines(
    topic_text: str,
    ref_doc: bytes,
    layers: float,
    model_name: str,
):
    doc = IngestedDoc(BytesIO(ref_doc))
    retrieved = doc.retrieve(topic_text)
    input_text = "\n".join([topic_text] + retrieved) + "\n"

    direct = generate_outline(input_text, int(layers), model_name)
    recursive = recursive_outline(input_text, int(layers), model_name)
    return "\n".join(retrieved), direct, dict_to_markdown(recursive)


if __name__ == "__main__":

    iface = gr.Interface(
        fn=compare_outlines,
        inputs=[
            gr.Textbox(label="Topic"),
            gr.UploadButton(
                "Upload a PDF",
                type="bytes",
                file_types=["pdf"],
                file_count="single",
                show_progress=True,
            ),
            gr.Number(label="Num of Layers"),
            gr.Dropdown(AVAILABLE_MODELS, label="LLMs"),
        ],
        outputs=[
            gr.Textbox(label="Retrieved Content"),
            gr.Textbox(label="Directly"),
            gr.Markdown(
                label="Recursively", 
                container=True, 
                show_copy_button=True
            ),
        ],
        title="Outline Generation",
        description="Input topic description and reference document, then you can compare with 2 methods' output performances."
    )

    iface.launch()
