#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import subprocess
import numpy as np
from tqdm import tqdm

# ========== CUDA/cuDNN 配置 ==========
# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 配置cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # 设为False以避免图优化问题
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32以提高稳定性
torch.backends.cudnn.allow_tf32 = False

# ========== Step1: description -> descriptionjson ==========

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_tagged_text_to_json(idx, tagged_text):
    """
    将模型输出的标签化文本解析为JSON格式，并返回 {idx: [ ... ]}
    标签示例：
       <starttime>0</starttime><endtime>1</endtime><speed>high</speed><direction>down</direction><rotate>0</rotate><sep>...
    """
    segments = tagged_text.split("<sep>")
    results = []

    for seg in segments:
        starttime_match = re.search(r"<starttime>(.*?)</starttime>", seg)
        endtime_match   = re.search(r"<endtime>(.*?)</endtime>", seg)
        speed_match     = re.search(r"<speed>(.*?)</speed>", seg)
        direction_match = re.search(r"<direction>(.*?)</direction>", seg)
        rotate_match    = re.search(r"<rotate>(.*?)</rotate>", seg)

        if all([starttime_match, endtime_match, speed_match, direction_match, rotate_match]):
            starttime = starttime_match.group(1)
            endtime   = endtime_match.group(1)
            speed     = speed_match.group(1)
            direction = direction_match.group(1)
            rotate    = rotate_match.group(1)

            # 转换为 float
            try:
                starttime = float(starttime)
            except:
                pass
            try:
                endtime = float(endtime)
            except:
                pass

            results.append({
                "starttime": starttime,
                "endtime": endtime,
                "speed": speed,
                "direction": direction,
                "rotate": rotate
            })

    return {f"{idx+1}": results}


def generate_description_json(
    input_txt_path,
    output_json_path,
    llama_model_name="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint_path="/data/sustech/home/yangxiaoda/FlexCaM/description-video/llamacheckpoint/best_model.pth"
):
    """
    读取 input_txt_path 中的描述文本，通过 LLaMA(or 其它) 模型生成标签化描述，
    并输出 JSON 到 output_json_path。
    """

    # 读取输入文本
    with open(input_txt_path, "r", encoding="utf-8") as f:
        input_texts = f.read().strip()

    # 准备设备 - 添加错误处理
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA不可用，使用CPU")

    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        if tokenizer.bos_token is None:
            tokenizer.bos_token = ""
        if tokenizer.eos_token is None:
            tokenizer.eos_token = ""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载基础模型 + LoRA 权重
        print("正在加载模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name, 
            torch_dtype=torch.float16, 
            device_map='auto',
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        print("模型加载完成")

        # 分割多段输入（若你有 1:xxx\n2:xxx 的结构）
        descriptions = re.split(r'\d+:', input_texts)
        descriptions = [desc.strip() for desc in descriptions if desc.strip()]

        merged_results = {}

        for idx, desc in enumerate(descriptions):
            print(f"处理第 {idx+1} 段描述...")
            
            inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                try:
                    generation_output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        do_sample=True,
                        num_beams=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True
                    )
                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        print(f"cuDNN错误，尝试使用CPU: {e}")
                        # 将模型移到CPU
                        model.cpu()
                        input_ids = input_ids.cpu()
                        attention_mask = attention_mask.cpu()
                        
                        generation_output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=128,
                            do_sample=True,
                            num_beams=1,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True
                        )
                    else:
                        raise e

            generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            # 解析输出文本为 JSON
            result_json = parse_tagged_text_to_json(idx, generated_text)
            merged_results.update(result_json)

        # 将合并结果写入 JSON 文件
        json_data = json.dumps(merged_results, ensure_ascii=False, indent=4)
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            f.write(json_data)

        print(f"Step1 完成：已将描述信息输出到 {output_json_path}")
        
    except Exception as e:
        print(f"模型加载或推理过程中出现错误: {e}")
        print("尝试使用CPU模式...")
        # 如果GPU出现问题，回退到CPU模式
        generate_description_json_cpu(input_txt_path, output_json_path, llama_model_name, checkpoint_path)


def generate_description_json_cpu(input_txt_path, output_json_path, llama_model_name, checkpoint_path):
    """CPU模式的备用函数"""
    print("使用CPU模式运行...")
    # 这里可以实现CPU版本的逻辑，或者简单地设置device="cpu"
    # 为了简化，这里只是一个占位符
    pass


# ========== Step2: descriptionjson -> trajectoryjson ==========

class Text2VideoSet:
    def __init__(self, input_path, output_dir, fps=8):
        self.input_path = input_path
        self.output_dir = output_dir
        self.fps = fps

        with open(self.input_path, 'r') as f:
            self.JsonData = json.load(f)

    def process_direction(self, direction, speed):
        """
        将方向和速度转为 (dphi, dtheta, dr)
        """
        # 针对平移(前后)的简化处理
        dr = 0.2
        if direction in ['backward', 'forward']:
            if direction == 'forward':
                dr = -0.2
            if speed == 'low':
                dr *= 0.5
            return 0, 0, dr

        # 针对旋转(上下左右)的处理
        dphi, dtheta = 20, 15

        if direction == 'left':
            dphi, dtheta = -dphi, 0
        elif direction == 'right':
            dphi, dtheta = dphi, 0
        elif direction == 'up':
            dphi, dtheta = 0, -dtheta
        elif direction == 'down':
            dphi, dtheta = 0, dtheta
        elif direction == 'leftup45':
            dphi, dtheta = -dphi * 0.707, -dtheta * 0.707
        elif direction == 'rightup45':
            dphi, dtheta = dphi * 0.707, -dtheta * 0.707
        elif direction == 'leftdown45':
            dphi, dtheta = -dphi * 0.707, dtheta * 0.707
        elif direction == 'rightdown45':
            dphi, dtheta = dphi * 0.707, dtheta * 0.707
        elif direction == 'leftup30':
            dphi, dtheta = -dphi * 0.5, -dtheta * 0.866
        elif direction == 'rightup30':
            dphi, dtheta = dphi * 0.5, -dtheta * 0.866
        elif direction == 'leftdown30':
            dphi, dtheta = -dphi * 0.5, dtheta * 0.866
        elif direction == 'rightdown30':
            dphi, dtheta = dphi * 0.5, dtheta * 0.866
        elif direction == 'leftup60':
            dphi, dtheta = -dphi * 0.866, -dtheta * 0.5
        elif direction == 'rightup60':
            dphi, dtheta = dphi * 0.866, -dtheta * 0.5
        elif direction == 'leftdown60':
            dphi, dtheta = -dphi * 0.866, dtheta * 0.5
        elif direction == 'rightdown60':
            dphi, dtheta = dphi * 0.866, dtheta * 0.5
        else:
            dphi, dtheta = 0, 0

        if speed == 'low':
            dphi *= 0.5
            dtheta *= 0.5

        return dphi, dtheta, 0

    def tune_pose(self, key, records):
        """
        将若干段 (starttime, endtime, direction, speed) 合并到一个以 fps 为步长的姿态序列
        """
        phi, theta, r = 0, 0, 0  # 初始姿态
        out = np.array([[phi], [theta], [r]])
        last_time = 0

        for record in records:
            start_t = record['starttime']
            end_t   = record['endtime']
            # 先补全上一个时间段 -> 本段开始时间 的静止状态
            if last_time < start_t:
                still_frame_num = int(self.fps * (start_t - last_time))
                still_out = np.array([
                    [out[0, -1]] * still_frame_num,
                    [out[1, -1]] * still_frame_num,
                    [out[2, -1]] * still_frame_num
                ])
                out = np.concatenate((out, still_out), axis=1)

            frame_num = int(self.fps * (end_t - start_t))
            dphi, dtheta, dr = self.process_direction(record['direction'], record['speed'])

            t_out = np.zeros((3, frame_num))
            if frame_num > 0:
                # 第一帧
                t_out[:, 0] = out[:, -1] + (1.0 / self.fps) * np.array([dphi, dtheta, dr])

                # 后续帧
                for i in range(1, frame_num):
                    t_out[:, i] = t_out[:, i - 1] + (1.0 / self.fps) * np.array([dphi, dtheta, dr])

                out = np.concatenate((out, t_out), axis=1)

            last_time = end_t

        return out.T  # shape: (N, 3)

    def process(self):
        """
        读取 self.input_path 中的 JSON（包含标签信息），
        转为每帧 (phi, theta, r)，并覆盖写到 mytrajectory.json
        """
        for key, records in tqdm(self.JsonData.items()):
            out = self.tune_pose(key, records)
            # 转 list 并保留三位小数
            out_list = out.tolist()
            for i in range(len(out_list)):
                out_list[i] = [round(x, 3) for x in out_list[i]]
            # 将 key 对应的记录替换为 数值列表
            self.JsonData[key] = out_list

        # 覆盖写回
        output_json_path = os.path.join(self.output_dir, 'mytrajectory.json')
        with open(output_json_path, 'w') as f:
            json.dump(self.JsonData, f, indent=4)

        print(f"Step2 完成：已将动作信息转换为帧级别姿态，并写回 {output_json_path}")


def convert_description_to_trajectory_json(input_json_path, output_dir, fps=8):
    """
    读取 Step1 生成的动作描述 JSON，转换为每帧姿态信息并覆盖写到同名文件
    """
    t2vset = Text2VideoSet(input_path=input_json_path, output_dir=output_dir, fps=fps)
    t2vset.process()


# ========== Step3: trajectoryjson -> viewcrafterinput(txt) ==========

def read_trajectory(json_path):
    """
    读取 Step2 产生的 Nx3 JSON，并返回 (theta, phi, r)
    JSON 格式假设:
       {
         "1": [
           [phi0, theta0, r0],
           [phi1, theta1, r1],
           ...
         ]
       }
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    # 假设只有一个 key，比如 "1"
    # 你也可以修改以适配多 key 的情况
    key = list(data.keys())[0]
    trajectory = data.get(key, [])

    phi_vals = []
    theta_vals = []
    r_vals = []

    for point in trajectory:
        # point 应该是 [phi, theta, r]
        if len(point) == 3:
            phi_vals.append(point[0])
            theta_vals.append(point[1])
            r_vals.append(point[2])
        else:
            print(f"警告: 数据点格式异常: {point}")

    return phi_vals, theta_vals, r_vals

def save_to_txt(phi_list, theta_list, r_list, txt_path):
    """
    将 phi, theta, r 序列保存到文本文件，每个序列一行，空格分隔
    """
    with open(txt_path, 'w') as file:
        phi_line    = ' '.join(map(str, phi_list))
        theta_line  = ' '.join(map(str, theta_list))
        r_line      = ' '.join(map(str, r_list))

        file.write(phi_line + '\n')
        file.write(theta_line + '\n')
        file.write(r_line + '\n')

    print(f"Step3 完成：已将 (phi, theta, r) 写入 {txt_path}")


def convert_trajectory_to_txt(input_json_path, output_txt_path):
    phi, theta, r = read_trajectory(input_json_path)
    save_to_txt(phi, theta, r, output_txt_path)


# ========== Step4: viewcrafterinput(txt) -> final video/images ==========

def run_viewcrafter_inference(
    image_dir="/data/sustech/home/yangxiaoda/FlexCaM/vase.png",
    out_dir="/data/sustech/home/yangxiaoda/FlexCaM/description-video",
    traj_txt="/data/sustech/home/yangxiaoda/FlexCaM/description-video/myviewcrafterinput.txt"
):
    """
    通过调用 ViewCrafter 的 inference.py，生成结果
    请确保 ViewCrafter 项目及其依赖已安装，并修改好相应的 ckpt/config 路径
    """
    command = [
        "python", "/data/sustech/home/yangxiaoda/ViewCrafter/inference.py",
        "--image_dir", image_dir,
        "--out_dir", out_dir,
        "--traj_txt", traj_txt,
        "--mode", "single_view_txt",
        "--center_scale", "1.",
        "--elevation", "5",
        "--seed", "123",
        "--d_theta", "-30",
        "--d_phi", "45",
        "--d_r", "-.2",
        "--d_x", "50",
        "--d_y", "25",
        "--ckpt_path", "./checkpoints/model.ckpt",
        "--config", "configs/inference_pvd_1024.yaml",
        "--ddim_steps", "50",
        "--video_length", "25",
        "--device", "cuda:0",
        "--height", "576",
        "--width", "1024",
        "--model_path", "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ]

    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Step4 完成：ViewCrafter 推理执行成功!")
        print("ViewCrafter Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running ViewCrafter inference!")
        print("Error:\n", e.stderr)


# ========== 主流程 ==========

def main():
    """
    将四个步骤整合在一起的主函数。
    1) 读入 /data/sustech/home/yangxiaoda/FlexCaM/description-video/inferenceinput.txt
       -> 生成带标签描述的 JSON (mytrajectory.json, 动作描述)
    2) 读入该 JSON, 转为帧级别姿态 JSON (mytrajectory.json, Nx3)
    3) 读入 Nx3 JSON, 导出 myviewcrafterinput.txt
    4) 调用 ViewCrafter 生成最终视频/图像
    """

    # 可根据需求自定义路径
    input_txt_path  = "/data/sustech/home/yangxiaoda/FlexCaM/description-video/inferenceinput1.txt"
    output_dir      = "/data/sustech/home/yangxiaoda/luankaixuan/dataset/text2video_camera/test"
    trajectory_json = os.path.join(output_dir, "mytrajectory.json")
    viewcrafter_txt = os.path.join(output_dir, "myviewcrafterinput.txt")
    image_path      = "/data/sustech/home/yangxiaoda/FlexCaM/test.jpg"

    # Step1: 文本 -> 标签描述 JSON
    generate_description_json(
        input_txt_path   = input_txt_path,
        output_json_path = trajectory_json,
        llama_model_name = "meta-llama/Llama-3.1-8B-Instruct",
        checkpoint_path  = "/data/sustech/home/yangxiaoda/FlexCaM/description-video/llamacheckpoint/best_model.pth"
    )

    # Step2: 动作描述 JSON -> 帧级别姿态 JSON
    convert_description_to_trajectory_json(
        input_json_path = trajectory_json,
        output_dir      = output_dir,
        fps=8
    )

    # Step3: 帧级别姿态 JSON -> txt
    convert_trajectory_to_txt(
        input_json_path = trajectory_json,
        output_txt_path = viewcrafter_txt
    )

    # Step4: 调用 ViewCrafter 生成视频/图像
    run_viewcrafter_inference(
        image_dir = image_path,
        out_dir   = output_dir,
        traj_txt  = viewcrafter_txt
    )

    print("所有步骤执行完毕！")


if __name__ == "__main__":
    main()