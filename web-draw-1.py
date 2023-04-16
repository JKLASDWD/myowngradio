import gradio as gr
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
#import torch
from diffusers import DDIMScheduler
model_path = "/content/Fo1"
from PIL import Image
#pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
g_cuda = None
import datetime
#pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
#pipe2 = pipe2.to("cuda")
prompt = ""
negative_prompt=""

def greet(name):
    return "Hello " + name + "!"

def change_p1(prompt):
    prompt += "sky-pion,"
    return prompt

def change_p2(prompt):
    prompt += "peo-dw,"
    return prompt

def change_p3(prompt):
    prompt += "plant-dw,"
    return prompt
def change_p4():
    global prompt
    prompt += ""
    return prompt
def change_p5():
    global prompt
    prompt += ""
    return prompt
def change_p6():
    global prompt
    prompt += ""
    return prompt
def upload_file(files):
    image_=Image.open(files.name)
    return image_
def save_file(g):
    image_list = g
    try:
        for image in image_list:
            try:
                name = datetime.datetime.now()
                name = name.strftime("%Y-%m-%d-%H-%M-%S-%f")
                image.save(name+".jpg")
            except Exception as reason:
                print(reason)
                try:
                    ima=Image.open(image)
                    name = datetime.datetime.now()
                    name = name.strftime("%Y-%m-%d-%H-%M-%S-%f")
                    ima.save(name+".jpg")
                except Exception as reason1:
                    print(reason1)
    except Exception as reason2:
        name = datetime.datetime.now()
        name = name.strftime("%Y-%m-%d-%H-%M-%S-%f")
        g.save(name+".jpg")

def inference(prompt,num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

def img2imginference(prompt,num_samples,image,height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
      with torch.autocast("cuda"), torch.inference_mode():
        return pipe2(
                prompt,init_image=image,height=int(height), width=int(width),
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

with gr.Blocks(css="style-draw.css") as demo:
    with gr.Row():
        with gr.Column(elem_id="title-bg"):
            gr.Markdown("AI ARITIST",elem_id="title")
            gr.Markdown("基于扩散模型的多模态图像生成",elem_id="title2")
    with gr.Row(elem_id="test-1"):
        with gr.Tab(label="文生图"):
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        title = gr.Textbox(value="可选择如下提示词，或者自己输入英文提示词", label="提示词")
                        prompt1 = gr.Button(value="扁平化风景")
                        prompt2 = gr.Button(value="卡通插画人物")
                        prompt3 = gr.Button(value="植被")
                    with gr.Column():
                        in_prompt = gr.Textbox(label="文本提示词", interactive=True, elem_id="input-text-1")
                    with gr.Row():
                        guidance_scale = gr.Slider(label="文本拟合度", value=7.5, interactive=True, maximum=20,
                                                   minimum=1,
                                                   step=1)
                        Steps_ = gr.Slider(label="步长", value=20, interactive=True, maximum=50, minimum=10, step=1)
                    with gr.Row():
                        height = gr.Slider(label="图片长度", value=512, interactive=True, maximum=1024, step=4)
                        width = gr.Slider(label="图片宽度", value=512, interactive=True, maximum=1024, step=4)
                    num_samples = gr.Textbox(label="图片数量", value=1, interactive=True)
                    run = gr.Button(value="生成", elem_id="btn-primary-2")
                with gr.Column():
                    gallery = gr.Gallery(elem_id="gallery-1")
            prompt1.click(fn=change_p1, inputs=in_prompt, outputs=in_prompt)
            prompt2.click(fn=change_p2, inputs=in_prompt, outputs=in_prompt)
            prompt3.click(fn=change_p3, inputs=in_prompt, outputs=in_prompt)
            run.click(inference, inputs=[in_prompt,num_samples, height, width, Steps_, guidance_scale], outputs=gallery)
        with gr.Tab(label="图生图"):
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        title = gr.Textbox(value="可选择如下提示词，或者自己输入英文提示词", label="提示词")
                        prompt1 = gr.Button(value="扁平化风景")
                        prompt2 = gr.Button(value="卡通插画人物")
                        prompt3 = gr.Button(value="植被")
                    with gr.Column():
                        in_prompt = gr.Textbox(label="文本提示词", interactive=True, elem_id="input-text-1")
                    with gr.Row():
                        guidance_scale = gr.Slider(label="文本拟合度", value=7.5, interactive=True, maximum=20,
                                                   minimum=1, step=1)
                        Steps_ = gr.Slider(label="步长", value=20, interactive=True, maximum=50, minimum=10, step=1)
                    with gr.Row():
                        height = gr.Slider(label="图片长度", value=512, interactive=True, maximum=1024, step=4)
                        width = gr.Slider(label="图片宽度", value=512, interactive=True, maximum=1024, step=4)
                    num_samples = gr.Textbox(label="图片数量", value=1, interactive=True)
                    img2img = gr.Image(type="pil",interactive=True,elem_id="gallery-3")
                with gr.Column():
                    img2img_resu = gr.Gallery(elem_id="gallery-2")
                    run2 = gr.Button(value="生成", elem_id="btn-primary-3")
            prompt1.click(fn=change_p1, inputs=in_prompt, outputs=in_prompt)
            prompt2.click(fn=change_p2, inputs=in_prompt, outputs=in_prompt)
            prompt3.click(fn=change_p3, inputs=in_prompt, outputs=in_prompt)
            run2.click(fn=img2imginference,inputs=[in_prompt,img2img,num_samples, height, width, Steps_, guidance_scale],outputs=img2img_resu)
##prompt的修改稍微有点问题，但是模型是没有问题的
##修改prompt的函数和上传文件到图生图的流程还是有点问题

demo.launch(debug=True)

'''    with gr.Row(elem_id="body"):
        T2IMG = gr.Button(value="文生图",elem_id="btn-primary")
        I2IMG = gr.Button(value="图生图",elem_id="btn-primary-1")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                title= gr.Textbox(value="可选择如下提示词，或者自己输入英文提示词",label="提示词")
                prompt1 = gr.Button(value="扁平化风景")
                prompt2 = gr.Button(value="卡通插画人物")
                prompt3 = gr.Button(value="植被")
            with gr.Column():
                in_prompt = gr.Textbox(label="文本提示词",interactive=True,elem_id="input-text-1")
            with gr.Row():
                guidance_scale = gr.Slider(label="文本拟合度",value=7.5,interactive=True,maximum=20,minimum=1,step=1)
                Steps_ = gr.Slider(label="步长",value=20,interactive=True,maximum=50,minimum=10,step=1)
            with gr.Row():
                height = gr.Slider(label="图片长度", value=512,interactive=True,maximum=1024,step=4)
                width = gr.Slider(label="图片宽度", value=512,interactive=True,maximum=1024,step=4)
            num_samples = gr.Textbox(label="图片数量", value=1, interactive=True)
            run = gr.Button(value="生成", elem_id="btn-primary-2")
        with gr.Column():
            gallery = gr.Gallery(elem_id="gallery-1")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    img2img = gr.Image(type="pil",interactive=True,elem_id="gallery-3")
                    run2 = gr.Button(value="生成", elem_id="btn-primary-3")
                with gr.Column():
                    img2img_resu = gr.Gallery(elem_id="gallery-2")'''