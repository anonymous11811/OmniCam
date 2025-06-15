import json

# 定义输入和输出文件路径
input_json_path = '/data/home/yangxiaoda/FlexCaM/description-video/mytrajectory.json'
output_txt_path = '/data/home/yangxiaoda/FlexCaM/description-video/myviewcrafterinput.txt'

def read_trajectory(json_path):
    """读取JSON文件并提取(theta, phi, r)序列"""
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # 假设键是字符串"1"
    trajectory = data.get("1", [])
    
    # 分别提取theta, phi, r
    theta = []
    phi = []
    r = []
    
    for point in trajectory:
        if len(point) == 3:
            theta.append(point[0])
            phi.append(point[1])
            r.append(point[2])
        else:
            print(f"警告: 数据点格式不正确: {point}")
    
    return theta, phi, r

def save_to_txt(theta, phi, r, txt_path):
    """将theta, phi, r序列保存到文本文件中，每个序列一行，使用空格分隔"""
    with open(txt_path, 'w') as file:
        # 将数值转换为字符串，并使用空格连接
        theta_line = ' '.join(map(str, theta))
        phi_line = ' '.join(map(str, phi))
        r_line = ' '.join(map(str, r))
        
        # 写入文件
        file.write(theta_line + '\n')
        file.write(phi_line + '\n')
        file.write(r_line + '\n')

def main():
    theta, phi, r = read_trajectory(input_json_path)
    save_to_txt(theta, phi, r, output_txt_path)
    print(f"轨迹数据已成功保存到 {output_txt_path}")

if __name__ == "__main__":
    main()