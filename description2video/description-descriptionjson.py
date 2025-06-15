import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import os
def qprint(var,str):
    print("\033[92m"+"{}:{}".format(str,var)+"\033[0m")
def parse_tagged_text_to_json(id,tagged_text):
    """
    将模型输出的标签化文本解析为JSON格式。
    假设标签化文本结构类似：
    <starttime>0</starttime><endtime>1</endtime><speed>high</speed><direction>down</direction><sep>...
    """

    # 先以<sep>分割不同的动作片段
    segments = tagged_text.split("<sep>")

    results = []
    for seg in segments:
        # 对于每个seg，解析其中的字段
        starttime_match = re.search(r"<starttime>(.*?)</starttime>", seg)
        endtime_match = re.search(r"<endtime>(.*?)</endtime>", seg)
        speed_match = re.search(r"<speed>(.*?)</speed>", seg)
        direction_match = re.search(r"<direction>(.*?)</direction>", seg)
        rotate_match = re.search(r"<rotate>(.*?)</rotate>", seg)

        if starttime_match and endtime_match and speed_match and direction_match and rotate_match:
            starttime = starttime_match.group(1)
            endtime = endtime_match.group(1)
            speed = speed_match.group(1)
            direction = direction_match.group(1)
            rotate=rotate_match.group(1)

            # 尝试转换starttime和endtime为数值类型
            try:
                starttime = float(starttime)
            except ValueError:
                pass
            try:
                endtime = float(endtime)
            except ValueError:
                pass

            results.append({
                "starttime": starttime,
                "endtime": endtime,
                "speed": speed,
                "direction": direction,
                "rotate":rotate
            })

    # 假设ID为1，这里可根据需要修改,
    final_json = {f"{id+1}": results}
    return final_json

if __name__ == "__main__":
    # 测试输入文件
    test_input_path = "/data/home/yangxiaoda/FlexCaM/description-video/inferenceinput.txt"
    with open(test_input_path, "r", encoding="utf-8") as f:
        input_texts = f.read().strip()
        # qprint(input_text,'input_text')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    checkpoint_path = "/data/home/yangxiaoda/FlexCaM/description-video/llamacheckpoint/best_model.pth"
    # checkpoint_path = "/data/home/yangxiaoda/FlexCaM/PPO/ppo_epoch_2"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型和 LoRA 权重
    base_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map='auto')
    model = PeftModel.from_pretrained(base_model, checkpoint_path,local_files_only=True)
    model.eval()
    model.to(device)

    descriptions = re.split(r'\d+:', input_texts)
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]
    merged_results={}
    for id,desc in enumerate(descriptions):
        # 对输入文本进行分词
        inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 使用模型生成输出
        # 根据任务可能需要添加适当的prompt或调整参数，比如max_new_tokens, temperature等
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,  # 根据任务需要调参
                do_sample=True,     # 是否随机采样
                num_beams=1,         # beam search宽度
                eos_token_id=tokenizer.eos_token_id
            )

        # 解码生成的输出序列
        generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        # qprint(generated_text,'generated_text')
        # 对生成的标签化文本进行解析为JSON
        result_json = parse_tagged_text_to_json(id,generated_text)
        # qprint(result_json,'result_json')
        merged_results.update(result_json)
        # qprint(merged_results,'merged_results')

    # 打印或保存结果
    # print(json.dumps(merged_results, ensure_ascii=False, indent=4))
        # 如果需要保存为文件：
        # with open("output.json", "w", encoding="utf-8") as out_f:
        #     json.dump(result_json, out_f, ensure_ascii=False, indent=4)

    json_data = json.dumps(merged_results, ensure_ascii=False, indent=4)

    # 指定输出文件路径
    output_path = "/data/home/yangxiaoda/FlexCaM/description-video/mytrajectory.json"

    # 确保目标目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON数据到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_data)

    print(f"合并后的结果已保存到 {output_path}")