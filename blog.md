
# 【NVIDIA NIM 黑客松训练营】优化LLM生成大纲：基于 NVIDIA NIM 平台的递归结构化生成

## 项目简介

生成大纲（如演讲灵感、文章写作、PPT幻灯片制作等）是大语言模型在办公场景下常见应用需求。

然而由于

1. 大模型为了保证输出不局限于一种形式，在生成过程中存在随机采样，生成的内容和格式不可控；
2. 大模型是基于自回归的，在生成有复杂层次的大纲时上层大纲并没有生成完就生成下层内容，和人类规划大纲自顶向下的方法不同。

【目前的问题】

会导致直接让大模型生成大纲存在不限于以下的问题：

- 输出的内容是非结构化的，输出格式多样，难以统一和规范，生成的大纲后续的分析与利用变得复杂；
- 生成过程中的层级深度和每层的颗粒度无法灵活控制，导致生成内容有时过于简单，有时又过于复杂，缺乏可控性；
- 大纲的层级之间往往存在模糊和重叠，无法有效传达不同层次的重点内容；

为了解决上述问题，本文提出了一种递归结构化生成的方法，通过分层递归控制大模型生成大纲的过程，自顶向下，不仅能使大纲结构清晰，使输出结果便于下游任务使用，而且可以灵活控制各层的深度和颗粒度。这样生成的大纲不仅层次分明，内容也更加贴合实际应用需求，避免了层级模糊和内容重叠的现象。

本项目基于 NVIDIA NIM 平台，使用 LLaMA/Qwen/mistral 等作为基座模型，采用 gradio 构建示例用户图形界面。NVIDIA NIM 平台提供了强大的自然语言处理等人工智能能力，能够方便、高效地调用各种预训练模型，甚至进行定制化的推理。

【nim平台截图】

## 核心方案

该项目的核心是递归生成指定主题的大纲，先总后分，自顶向下，逐层递进，确保大纲的层次清晰、内容具体，避免生成结果的模糊和重叠问题。核心流程如下：

1. **第一层生成**：首先，通过模型生成第一层的PPT大纲内容，确保初始大纲框架具有明确的层次。
2. **递归生成**：在生成初始层之后，递归生成每个子级的内容，逐层递进。每一级的生成都是基于上一层的大纲内容进行细化和补充，保证了层次结构的清晰和逻辑性。
3. **递归终止条件**：当生成到达指定的层级时，递归终止，输出完整的结构化大纲。

Prompts 是生成式任务方案的重要组成部分，本项目中 Prompts 是用了基本的指令和 few-shot 方案，具体的 prompt 将在下一节“实现细节”中介绍。

【方案示意图】

### 第一层大纲生成 (`generate_first_level_outlines`)

这个函数负责生成大纲的第一层内容，基于用户输入的主题调用大模型生成初始的大纲框架。第一层大纲的生成比较特殊，与后续层次大纲生成有所不同，第一层生成时不存在前一层的信息，故有着单独的实现。

```python
def generate_first_level_outlines(idea: str, model_name: str):
    first_level_prompt = FIRST_LEVEL_TEMPLATE.format(
        few_shot=FIRST_LEVEL_EXAMPLES,
        idea=idea,
    )
    return extract_outlines(
        request_model_for_one_response(model_name, first_level_prompt)
    )
```

为了简化对模型调用的方式，本项目对大模型的请求简化成一条文本输入对应一条文本输出的函数，抽象成`request_model_for_one_response`方法。

得到模型的输出文本后，通过`extract_outlines`函数处理生成结果，去掉不必要的空白行，得到大纲的第一层结构化内容（即一个字符串列表）。

上面两个工具函数的详情见下一节“实现细节”。

### 递归生成下一级大纲 (`generate_next_level_outline`)

当生成了第一层的大纲后，系统将递归生成各个子级内容。`generate_next_level_outline`函数根据全局和上一层的大纲的信息，生成当前层的细化内容。

模型生成后，如果当前层级达到了目标层（target_layer），则返回当前层的大纲内容。否则，函数将继续递归调用自己，为每个子项生成更深层的大纲内容，直到达到目标层。

```python
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
```

注意，prompt 中会将上一级的内容和当前的层次信息整合，让模型有更充分的信息生成该层的大纲内容。

### 递归结构化大纲生成(`recursive_outline`)与大纲数据结构

这个函数是整个递归生成过程的整合，通过逐层调用`generate_next_level_outline`函数，生成每一层的大纲，直到生成目标层的所有内容。

```python
def recursive_outline(idea: str, target_layer: int, model_name: str,):
    first_level = generate_first_level_outlines(idea, model_name)
    return {
        text: generate_next_level_outline(
            idea, 2, text, first_level, target_layer, model_name
        ) for text in first_level
    }
```

最终生成的大纲是一个具有树状结构的嵌套 dict 数据结构。所有的大纲条目文本均是这个树状结构的一个节点。同一个双亲节点下的叶子节点的大纲条目将存储在一个 list 中，非叶子节点的大纲条目都是 dict 中的键，其子节点将存储在后续的 dict 或 list 中。

【数据结构示意图】

## 实现细节

### prompt

本项目的 Prompt 明确告知模型生成PPT大纲的任务，提供主题信息（idea）和所需层级（layer），确保模型能够聚焦于特定主题，并生成相应层次的内容。此外，通过提供多个生成示例（few-shot），向模型展示了预期的输出格式和内容结构，并提供了明确的上下文，帮助模型更好地生成和避免生成重叠和模糊地内容。

```python
BASE_TEMPLATE = """请你根据主题：“{idea}”生成{layer}层的幻灯片（PPT）大纲：\n"""

FIRST_LEVEL_EXAMPLES = """请你根据
技术是一把双刃剑，它既能推动社会向前发展，也可能因不当使用而造成伤害。比如数字鸿沟现象。
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容：
<信息爆炸的时代>
<科技革命与产业变革>
<社会影响日益凸显>
<人类应未雨绸缪，趋利避害>
<发展趋势及前景,结论>

请你根据
为什么我们要好好学习？在快速变化的世界中，持续学习是实现自我价值和社会贡献的关键。
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容：
<引言>
<学习对个人成长的意义>
<学习对社会的责任>
<如何有效学习>

请你根据
在单身的黄金年代我们如何面对爱情：要勇敢地去追求爱情
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容：
<何为单身的黄金时代>
<爱情对人们的意义>
<目前人的生存和处境更加复杂>
<爱情存在肌无力>
<最温暖的东西是人和人之间的感情>
<要勇敢地追求爱情>
"""

FIRST_LEVEL_TEMPLATE = """{few_shot}
上面是几个生成大纲第一层的例子。请你根据
{idea}
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容："""

NEXT_LEVEL_EXAMPLES = """请你根据
技术是一把双刃剑，它既能推动社会向前发展，也可能因不当使用而造成伤害。比如数字鸿沟现象。
不要和上一层的 信息爆炸的时代, 科技革命与产业变革, 人类应未雨绸缪，趋利避害, 发展趋势及前景,结论 重复，
生成“社会影响日益凸显”下的大纲第2层内容，一行一个条目，生成在<>中，无需其他解释和内容：
<举例：数字鸿沟现象>
<各种问题普遍存在>
<对人类各方面产生深远影响>

请你根据
在单身的黄金年代我们如何面对爱情：要勇敢地去追求爱情
不要和上一层的 何为单身的黄金时代, 爱情对人们的意义, 目前人的生存和处境更加复杂, 爱情存在肌无力, 最温暖的东西是人和人之间的感情 重复，
生成“要勇敢地追求爱情”下的大纲第2层内容，一行一个条目，生成在<>中，无需其他解释和内容：
<必须树立目标>
<提升心灵的力量>
<走合法的途径去获取>

请你根据
为什么我们要好好学习？在快速变化的世界中，持续学习是实现自我价值和社会贡献的关键。
不要和上一层的 理解自己的权利与义务, 正确价值观和社会责任感 重复，
生成“知识是推动社会发展的关键力量”下的大纲第3层内容，一行一个条目，生成在<>中，无需其他解释和内容：
<科技创新与经济发展>
<社会治理与公共政策>
<文化传承与发展>
"""

NEXT_LEVEL_TEMPLATE = """{few_shot}
上面是几个生成大纲具体层的例子。请你根据
{idea}
不要和上一层的 {formatted_previous_list} 重复，
生成“{previous_outline}”下的大纲第{current_level}层内容，一行一个条目，生成在<>中，无需其他解释和内容：
"""
```

【prompt示意图】

### 获取模型响应

本项目中对模型的调用需求比较简单，仅期望模型根据输入返回一条文本输出，故将所有的相关重复操作（配置、请求、提取）打包到一个函数中，简化调用流程：

```python
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
```

### 大纲文本提取

这里实现的是一个简单的提取方法，将模型输出按行分割并去除空白行和格式字符。

```python
def extract_outlines(model_output: str):
    return [
        line.strip().strip('<').strip('>')
        for line in model_output.split('\n')
        if line.strip()
    ]
```

### baseline 对比、全局变量和简易用户界面

大纲生成的基本过程通过调用 NVIDIA NIM 平台的接口，将主题与层级信息传入模型生成器，控制输出大纲的层级深度和层次内容。复现时需要更换`NVIDIA_API_KEY`变量。

简易的 web 用户界面是基于 gradio 构建的。用户只需输入主题和层数，根据喜好选择基座模型，项目就可以生成两个版本的PPT大纲，一个是直接生成的，一个是递归生成的，并将两者进行对比展示。

递归生成的大纲是结构化的，因此可以更好地在各种场景下和终端中展示。以本项目的此 demo 为例，最终展示是 gradio 构建的。因此，本 demo 将结构化的数据转换为 markdown 格式的字符串，借助 gradio 的 markdown 的渲染能力更清晰地展示生成效果。

```python
import gradio as gr
from openai import OpenAI

CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

AVAILABLE_MODELS = (
    "mistralai/mistral-7b-instruct-v0.2",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
    "qwen/qwen2-7b-instruct",
)

def generate_outline(
    input_text: str,
    layer: int,
    model_name: str,
):
    prompt = BASE_TEMPLATE.format(idea=input_text, layer=layer)
    return request_model_for_one_response(model_name, prompt)


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


def compare_outlines(input_text: str, layers: float, model_name: str):
    direct = generate_outline(input_text, int(layers), model_name)
    recursive = recursive_outline(input_text, int(layers), model_name)
    return direct, dict_to_markdown(recursive)


if __name__ == "__main__":

    iface = gr.Interface(
        fn=compare_outlines,
        inputs=[
            gr.Textbox(label="生成的主题"),
            gr.Number(label="生成的层数"),
            gr.Dropdown(AVAILABLE_MODELS, label="选择语言模型"),
        ],
        outputs=[
            gr.Textbox(label="直接生成"),
            gr.Markdown(label="递归结构化生成"),
        ],
        title="PPT大纲生成对比",
        description="输入一个句子，比较直接生成和递归生成的PPT大纲效果。"
    )

    iface.launch()
```

## 项目效果与分析总结

运行后，在浏览器中打开项目前端，输入要生成的主题和期望的层数，可以得到普通方法生成和本方法生成的结果。

【生成效果】

如图所示，递归结构化生成的效果不仅结构清晰、方便利用，而且相对于普通方法，生成了正确的层次深度，大纲对于主题的分析更加深入和全面，有更多的信息量和参考价值。

通过采用递归结构化生成的方法，本项目能够有效地解决传统PPT大纲生成中遇到的各种问题，为用户提供更清晰、有条理的内容。希望本项目的实践经验能够为更多相关应用的探索与优化提供启示和借鉴。

**完整代码：**

```python

import gradio as gr
from openai import OpenAI

from __env import NVIDIA_API_KEY

CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

BASE_TEMPLATE = """请你根据主题：“{idea}”生成{layer}层的幻灯片（PPT）大纲：\n"""

FIRST_LEVEL_EXAMPLES = """请你根据
技术是一把双刃剑，它既能推动社会向前发展，也可能因不当使用而造成伤害。比如数字鸿沟现象。
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容：
<信息爆炸的时代>
<科技革命与产业变革>
<社会影响日益凸显>
<人类应未雨绸缪，趋利避害>
<发展趋势及前景,结论>

请你根据
为什么我们要好好学习？在快速变化的世界中，持续学习是实现自我价值和社会贡献的关键。
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容：
<引言>
<学习对个人成长的意义>
<学习对社会的责任>
<如何有效学习>

请你根据
在单身的黄金年代我们如何面对爱情：要勇敢地去追求爱情
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容：
<何为单身的黄金时代>
<爱情对人们的意义>
<目前人的生存和处境更加复杂>
<爱情存在肌无力>
<最温暖的东西是人和人之间的感情>
<要勇敢地追求爱情>
"""

FIRST_LEVEL_TEMPLATE = """{few_shot}
上面是几个生成大纲第一层的例子。请你根据
{idea}
生成大纲第1层，一行一个条目，生成在<>中，无需其他解释和内容："""

NEXT_LEVEL_EXAMPLES = """请你根据
技术是一把双刃剑，它既能推动社会向前发展，也可能因不当使用而造成伤害。比如数字鸿沟现象。
不要和上一层的 信息爆炸的时代, 科技革命与产业变革, 人类应未雨绸缪，趋利避害, 发展趋势及前景,结论 重复，
生成“社会影响日益凸显”下的大纲第2层内容，一行一个条目，生成在<>中，无需其他解释和内容：
<举例：数字鸿沟现象>
<各种问题普遍存在>
<对人类各方面产生深远影响>

请你根据
在单身的黄金年代我们如何面对爱情：要勇敢地去追求爱情
不要和上一层的 何为单身的黄金时代, 爱情对人们的意义, 目前人的生存和处境更加复杂, 爱情存在肌无力, 最温暖的东西是人和人之间的感情 重复，
生成“要勇敢地追求爱情”下的大纲第2层内容，一行一个条目，生成在<>中，无需其他解释和内容：
<必须树立目标>
<提升心灵的力量>
<走合法的途径去获取>

请你根据
为什么我们要好好学习？在快速变化的世界中，持续学习是实现自我价值和社会贡献的关键。
不要和上一层的 理解自己的权利与义务, 正确价值观和社会责任感 重复，
生成“知识是推动社会发展的关键力量”下的大纲第3层内容，一行一个条目，生成在<>中，无需其他解释和内容：
<科技创新与经济发展>
<社会治理与公共政策>
<文化传承与发展>
"""

NEXT_LEVEL_TEMPLATE = """{few_shot}
上面是几个生成大纲具体层的例子。请你根据
{idea}
不要和上一层的 {formatted_previous_list} 重复，
生成“{previous_outline}”下的大纲第{current_level}层内容，一行一个条目，生成在<>中，无需其他解释和内容：
"""

AVAILABLE_MODELS = (
    "mistralai/mistral-7b-instruct-v0.2",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
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
        if line.strip()
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


def compare_outlines(input_text: str, layers: float, model_name: str):
    direct = generate_outline(input_text, int(layers), model_name)
    recursive = recursive_outline(input_text, int(layers), model_name)
    return direct, dict_to_markdown(recursive)


if __name__ == "__main__":

    iface = gr.Interface(
        fn=compare_outlines,
        inputs=[
            gr.Textbox(label="生成的主题"),
            gr.Number(label="生成的层数"),
            gr.Dropdown(AVAILABLE_MODELS, label="选择语言模型"),
        ],
        outputs=[
            gr.Textbox(label="直接生成"),
            gr.Markdown(label="递归结构化生成"),
        ],
        title="PPT大纲生成对比",
        description="输入一个句子，比较直接生成和递归生成的PPT大纲效果。"
    )

    iface.launch()

```

[https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag]
[https://ashevat.medium.com/growth-is-a-promise-retention-is-a-promise-kept-839ce317310c]
