from flask import Flask, request, jsonify
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import transformers
app = Flask(__name__)

# Load the model and tokenizer
import torch
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = "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"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token=access_token,
    device_map="auto",
)
# translate_tokenizer = AutoTokenizer.from_pretrained("thilina/mt5-sinhalese-english")
# translator = AutoModelForSeq2SeqLM.from_pretrained("thilina/mt5-sinhalese-english")
# translate_tokenizer = AutoTokenizer.from_pretrained("machinelearningzuu/t5-small-sinhala-english-nmt")
# translator = AutoModelForSeq2SeqLM.from_pretrained("machinelearningzuu/t5-small-sinhala-english-nmt") 
engTosin = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
engTosin_t = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
sinToeng = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
sinToeng_t = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# FastLanguageModel.for_inference(model)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('input', "")
    payload = {
    'input': text
    }
    print("---------------------------------------------------------------------------------------------------------------")
    response = requests.post('http://localhost:5000/transliterate', json=payload)
    next_input=response.json()['transliterated']
    print("input :" + text)
    print("transliteration "+next_input)
    # next_input is in sinhala language
    # inputs = translate_tokenizer(next_input, return_tensors="pt")
    # outputs = translator.generate(**inputs)
    # translated_text = translate_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # translated_text is in english now.
    sinToeng_t.src_lang = "si_LK"
    encoded_hi = sinToeng_t(next_input, return_tensors="pt")
    generated_tokens = sinToeng.generate(**encoded_hi)
    t=sinToeng_t.batch_decode(generated_tokens, skip_special_tokens=True)
    translated_text=t[0]
    print("translated_text "+translated_text)

    # inputs = tokenizer(
    #     [alpaca_prompt.format(translated_text, "", "")], 
    #     return_tensors="pt"
    # ).to("cuda")

    # text_streamer = TextStreamer(tokenizer)
    # output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
    messages = [
        {"role": "system", "content": "Write a response that appropriately completes the request.!"},
        {"role": "user", "content": translated_text},
    ]
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.9,
        top_p=0.9,
    )    
    # Convert output to string
    t =outputs[0]["generated_text"][-1]
    from_llama=t["content"]
    model_inputs = engTosin_t(from_llama, return_tensors="pt")
    generated_tokens = engTosin.generate(
        **model_inputs,
        forced_bos_token_id=engTosin_t.lang_code_to_id["si_LK"]
    )
    t=engTosin_t.batch_decode(generated_tokens, skip_special_tokens=True)
    translated_text=t[0]
    print("translated_text "+translated_text)

    # from_llama is the response we get from llama in english language
    # inputs = translate_tokenizer(from_llama, return_tensors="pt")

    # outputs = translator.generate(**inputs)
    # translated_text = translate_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(translated_text)
    # this translated_text is in sinhala now.
    payload = {
    'input': translated_text
    }
    response = requests.post('http://localhost:5000/romanize', json=payload)
    output=response.json()['romanized']
    print("output " + output)
    return jsonify({"msg": output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
