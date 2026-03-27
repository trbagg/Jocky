import gc
import torch
from finetunejockypt import model_name
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.randn(1).cuda())
torch.cuda.empty_cache()
import transformers
from peft import PeftModel

local_model_path = "./jockypt-ft/checkpoint-210" # ./jockypt-ft/checkpoint-510

print("Loading models...")

cache_dir="./LLaMA_Quant"
sys_instruct_path = "./content/system_instruction.txt"

with open(sys_instruct_path, 'r') as f:
	sys_instruct = f.read()

bnb_config = transformers.BitsAndBytesConfig(
												load_in_4bit=True,
												bnb_4bit_quant_type="nf4",
												bnb_4bit_use_double_quant=True,
												bnb_4bit_compute_dtype=torch.bfloat16
											)

base_model = transformers.AutoModelForImageTextToText.from_pretrained(
	model_name,
	cache_dir=cache_dir, 
	quantization_config=bnb_config,
	dtype=torch.bfloat16,
	device_map="auto",
	trust_remote_code=True,
	attn_implementation="sdpa",
)

processor = transformers.AutoProcessor.from_pretrained(
                                                                "./jockypt-ft/processor",
																local_files_only=True,
                                                                add_bos_token=True,
                                                                add_eos_token=False, 
                                                            )
    
eval_tokenizer = processor.tokenizer

test_text = "[gif: ;) :) \\_/ :3"
tokens = eval_tokenizer.encode(test_text)
decoded = eval_tokenizer.decode(tokens)
print(f"Original: {test_text}")
print(f"Decoded: {decoded}")
test_string = "[gif: malphite, meme, gif]"
print(f"Tokenizer special tokens pre-sanity check: {test_string[:5]} == {eval_tokenizer.encode(test_string[:5], add_special_tokens=False)} && {test_string} == {eval_tokenizer.encode(test_string, add_special_tokens=False)}")
print()

base_model.resize_token_embeddings(len(eval_tokenizer))

peft_model = PeftModel.from_pretrained(
										base_model, 
									 	local_model_path,
										device_map="auto",
										dtype=torch.bfloat16
									)



def inference(message, history, url = None, image = None, no_bot = False, member = 'Yogipanda', temperature = None):
	print(f"Infering {member}'s message: {message}")
	try:
		if no_bot:
			message = input("User: ")
		
		content = []
		if url:
			content.append({"type": "image", "url": url})
		if image:
			content.append({"type": "image", "image": image})
		
		content.append({"type": "text", "text": message})
		
		prompt_inputs = processor.apply_chat_template(
															#[{"role": "system", "content": f"{sys_instruct}"}] + 
															[{"role": entry['role'], "content": [{"type": "text", "text": entry['content']}]} for entry in history] + #
															[{"role": "user", "content": content
															}],
															tokenize=True,
															add_generation_prompt=True,
															add_special_tokens=False,
															return_dict=True,
															return_tensors="pt",
														).to('cuda')
		
		peft_model.eval()

		
		with torch.no_grad():
			output = base_model.generate(
											**prompt_inputs, 
											max_new_tokens=128, # 512
											temperature= 0.7 if temperature is None else temperature, # Note: initial temp is set in jocky_bot.py when calling inference()
											repetition_penalty=1.2,
											top_p=0.8,
											top_k=20,
											do_sample=True,
										)
			
			trimmed = [
				out[len(in_ids):]
				for in_ids, out in zip(prompt_inputs["input_ids"], output)
			]

			generated_text = processor.batch_decode(
														trimmed,
														skip_special_tokens=True,
														clean_up_tokenization_spaces=False
													)

		return generated_text[0]

	except KeyboardInterrupt:
		return





def message_history(history, user, msg, context_length = 8192, max_messages = 16):
	history['messages'].append({"role": user, "content": msg})
	tokenized_history = tokenize_history(history=history['messages'])
	history['length'] = len(eval_tokenizer.tokenize(tokenized_history))
	while history['length'] >= context_length:
		history['messages'].pop(0)
		tokenized_history = tokenize_history(history=history['messages'])
		history['length'] = len(eval_tokenizer.tokenize(tokenize_history(history=history['messages'])))
	
	while max_messages > -1 and len(history['messages']) > max_messages:
		history['messages'].pop(0)
		if history['messages'][0]['role'] != 'user':
			history['messages'].pop(0)
		history['length'] = len(eval_tokenizer.tokenize(tokenize_history(history=history['messages'])))

def tokenize_history(history):
	return processor.apply_chat_template(
												history,
												tokenize=False,
												add_generation_prompt=True,
												add_special_tokens=False,
											)

def cleanup():
	gc.collect()
	torch.cuda.empty_cache()
	print("Cleanup done at program exit.")

if __name__ == '__main__':

	cleanup()
	print("This file is not meant to be run. Perhaps you meant to run jocky_bot.py?")
	exit(1)