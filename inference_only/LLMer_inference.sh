set -e

# cd code

base_dir=""
local_dir=""

# data_name="zero_test"
# data_name="few_test"
data_name="cot_test"


for model_name in google/flan-t5-large \
google/flan-t5-xl \
google/flan-t5-xxl \
meta-llama/Llama-2-7b-chat-hf \
meta-llama/Llama-2-13b-chat-hf \
lmsys/vicuna-7b-v1.5 \
lmsys/vicuna-13b-v1.5 \
WizardLM/WizardLM-13B-V1.2 \
baichuan-inc/Baichuan2-7B-Chat \
baichuan-inc/Baichuan2-13B-Chat \
THUDM/chatglm2-6b \
Qwen/Qwen-14B-Chat \
internlm/internlm-chat-7b-v1_1 \
tiiuae/falcon-7b-instruct 
do
model_name_short=$(echo $model_name | sed 's/\//_/g') # replacing \ with _
echo $model_name_short
CUDA_VISIBLE_DEVICES=1 python LLMer_inference.py --call_type llm \
--model_name ${model_name} --local_dir ${local_dir} \
--data_path "${base_dir}/data/${data_name}.json" \
--save_path "${base_dir}/exp/${data_name}/${model_name_short}.jsonl"
done
