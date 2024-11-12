from jinja2 import Template

# Define the Jinja template for messages
# message_template =  Template("{% for message in messages %}{% if message['role'] == 'user' %}{% if message['image'] %}{% if loop.index > 1 %}Image-{{ loop.index }}: {{ message['image'] }}\n{% else %}USER:<image>\n{{ message['content'] }}{% endif %}{% else %}USER:\n{{ message['content'] }}{% endif %}{% elif message['role'] == 'assistant' %}ASSISTANT:{{ message['content'] }}{% endif %}{% endfor %}{% if messages|selectattr('image')|list|length > 1 %}Describe the two images in detail.{% endif %}ASSISTANT:")
olmo_template =  "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content']}}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

message_template = olmo_template

message_template = Template(olmo_template)  # Create a Template object

def render_messages(messages, eos_token=""):
    return message_template.render(messages=messages, eos_token=eos_token, add_generation_prompt=True)


text_only_messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "What is the capital of Germany?"}  
]

single_image_messages = [
    {"role": "user", "content": "What is this image?", "image": "examples/image1.jpg"},
    {"role": "assistant", "content": "Shoes a dining room table with 6 chairs"},
    {"role": "user", "content": "Tell me more about this image"}  
]

multi_image_messages = [
    {"role": "user", "content": "What is this image?", "image": "examples/image1.jpg"},
    {"role": "assistant", "content": "Shoes a dining room table with 6 chairs"},
    {"role": "user", "content": "What is this image?", "image": "examples/image2.jpg"},
    {"role": "assistant", "content": "A childrens playroom with a TV and images on the walls"},
    {"role": "user", "content": "Tell me the difference between the two images"}  
]

image_url_messages = [
    {"role": "user", "content": "What is this image?", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"},
]


text_only_rendered = render_messages(text_only_messages)
print(text_only_rendered)
print("-"*100)

single_image_rendered = render_messages(single_image_messages)
print(single_image_rendered)
print("-"*100)

multi_image_rendered = render_messages(multi_image_messages)
print(multi_image_rendered)
print("-"*100)

image_url_rendered = render_messages(image_url_messages)
print(image_url_rendered)
