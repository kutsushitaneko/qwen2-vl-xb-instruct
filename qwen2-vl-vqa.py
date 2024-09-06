import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_options = ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]
model = None
processor = None
current_model_id = None

def load_model(model_id):
    global model, processor, current_model_id
    if current_model_id == model_id:
        return "既にロードされています。"
    
    print(f"{model_id}をロード中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 既存のモデルがある場合、メモリを解放
    if model is not None:
        del model
        torch.cuda.empty_cache()
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    current_model_id = model_id
    return "モデルがロードされました。"

def generate_response(model_choice, image1, image2, question):
    global model, processor, current_model_id
    
    if current_model_id != model_choice:
        load_model(model_choice)
    
    messages = [{"role": "user", "content": []}]

    # 画像の追加
    if image1 is not None:
        messages[0]["content"].append({"type": "image", "image": image1})
    if image2 is not None:
        messages[0]["content"].append({"type": "image", "image": image2})

    # テキストプロンプトの追加
    messages[0]["content"].append({"type": "text", "text": question})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def clear_inputs():
    return None, None, ""

with gr.Blocks() as demo:
    gr.Markdown("# Qwen2-VL Visual Question Answering")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=model_options, label="モデル選択", value=model_options[0])
    
    with gr.Row():
        image_input1 = gr.Image(type="pil", label="画像1")
        image_input2 = gr.Image(type="pil", label="画像2")
    
    question_input = gr.Textbox(label="質問", lines=2, show_copy_button=True)
    
    with gr.Row():
        submit_button = gr.Button("送信")
        clear_button = gr.Button("クリア")
    
    output = gr.Textbox(label="回答", lines=10, show_copy_button=True)
    
    submit_button.click(generate_response, inputs=[model_dropdown, image_input1, image_input2, question_input], outputs=[output])
    clear_button.click(clear_inputs, outputs=[image_input1, image_input2, question_input])

demo.launch(share=True, inbrowser=True)

