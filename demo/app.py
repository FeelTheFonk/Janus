import os
import numpy as np
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

MODEL_PATH = "deepseek-ai/Janus-1.3B"

vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer
vl_gpt = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def multimodal_understanding(
    image_array, question_text, max_new_tokens=512, do_sample=False, temperature=1.0, top_k=50, top_p=1.0
):
    conversation = [
        {"role": "User", "content": f"<image_placeholder>\n{question_text}", "images": [image_array]},
        {"role": "Assistant", "content": ""},
    ]
    pil_image = Image.fromarray(image_array)
    inputs = vl_chat_processor(conversations=conversation, images=[pil_image], force_batchify=True).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        use_cache=True,
    )
    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return response

@torch.inference_mode()
def text_to_image(prompt_text, temperature=1.0, cfg_weight=5.0, parallel_size=4, img_size=384):
    conversation = [{"role": "User", "content": prompt_text}, {"role": "Assistant", "content": ""}]
    sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
    )
    full_prompt = sft_prompt + vl_chat_processor.image_start_tag
    input_ids = vl_chat_processor.tokenizer.encode(full_prompt)
    input_ids_tensor = torch.LongTensor(input_ids)
    total_size = int(parallel_size) * 2
    tokens = torch.zeros((total_size, len(input_ids)), dtype=torch.int).cuda()
    for idx in range(total_size):
        tokens[idx] = input_ids_tensor
        if idx % 2 != 0:
            tokens[idx, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    image_token_num = 576
    patch_size = 16
    generated_tokens = torch.zeros((int(parallel_size), image_token_num), dtype=torch.int).cuda()
    past_key_values = None
    for i in range(image_token_num):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values
        )
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logits_cond = logits[0::2]
        logits_uncond = logits[1::2]
        logits = logits_uncond + cfg_weight * (logits_cond - logits_uncond)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_tokens = torch.multinomial(probabilities, num_samples=1)
        generated_tokens[:, i] = next_tokens.squeeze(-1)
        next_tokens_double = torch.cat([next_tokens, next_tokens], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_tokens_double)
        inputs_embeds = img_embeds.unsqueeze(1)
        past_key_values = outputs.past_key_values
    decoded_images = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.int(),
        shape=[int(parallel_size), 8, int(img_size) // patch_size, int(img_size) // patch_size],
    )
    decoded_images = decoded_images.float().cpu().numpy().transpose(0, 2, 3, 1)
    decoded_images = np.clip((decoded_images + 1) / 2 * 255, 0, 255).astype(np.uint8)
    images = [Image.fromarray(img) for img in decoded_images]
    return images

with gr.Blocks() as demo:
    gr.Markdown("# Janus Model Demo")
    with gr.Tab("Multimodal Understanding"):
        with gr.Row():
            image_input = gr.Image()
            question_input = gr.Textbox(label="Question")
        with gr.Accordion("Advanced Options", open=False):
            max_new_tokens_input = gr.Slider(minimum=1, maximum=1024, value=512, label="Max New Tokens")
            do_sample_input = gr.Checkbox(label="Use Sampling", value=False)
            temperature_input_mmu = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature")
            top_k_input = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Top-k")
            top_p_input = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Top-p")
        analyze_button = gr.Button("Analyze")
        answer_output = gr.Textbox(label="Answer")
    with gr.Tab("Text-to-Image Generation"):
        prompt_input = gr.Textbox(label="Prompt")
        with gr.Accordion("Advanced Options", open=False):
            temperature_input_t2i = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
            cfg_weight_input = gr.Slider(1, 10, value=5.0, step=0.5, label="CFG Weight")
            parallel_size_input = gr.Slider(1, 16, step=1, value=4, label="Parallel Size")
            img_size_input = gr.Dropdown(choices=[256, 384, 512], value=384, label="Image Size")
        generate_button = gr.Button("Generate Images")
        images_output = gr.Gallery(label="Generated Images", columns=2, rows=2)
    analyze_button.click(
        fn=multimodal_understanding,
        inputs=[
            image_input,
            question_input,
            max_new_tokens_input,
            do_sample_input,
            temperature_input_mmu,
            top_k_input,
            top_p_input,
        ],
        outputs=answer_output,
    )
    generate_button.click(
        fn=text_to_image,
        inputs=[
            prompt_input,
            temperature_input_t2i,
            cfg_weight_input,
            parallel_size_input,
            img_size_input,
        ],
        outputs=images_output,
    )
demo.launch(share=False)