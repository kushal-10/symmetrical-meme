import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from jinja2 import Template

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is the capital of France?"}
        ],
    },
]

# Define the Jinja2 template for formatting the conversation
chat_template = Template("{% for message in messages %}{{ message['role'].capitalize() + ': ' + (message['content'][0]['text'] if message['content'] else '') + '\\n' }}{% endfor %}Assistant: ")

# Use the template to format the conversation
formatted_prompt = chat_template.render(messages=conversation)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
# inputs = processor(images=None, text=prompt, return_tensors='pt').to(0, torch.float16)
inputs = processor(text=formatted_prompt, images = [] return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))


"""
Sample response
USER: <image>
What are these? ASSISTANT:
Expanding inputs for image tokens in LLaVa should be done in processing. Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. Using processors without these attributes in the config is deprecated and will throw an error in v4.47.
Expanding inputs for image tokens in LLaVa should be done in processing. Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. Using processors without these attributes in the config is deprecated and will throw an error in v4.47.
ER:
What are these? ASSISTANT: These are two cats lying on a pink couch.

processor = AutoProcessor.from_pretrained(model_id)

"""