#读取/data/home/yangxiaoda/FlexCaM/trajectory.json，
#读取/data/home/yangxiaoda/FlexCaM/vid2json/data/vase.png，跑viewcrafter
#viewcrafter怎么跑 sh run.sh
import subprocess
import sys
sys.path.append('/data/home/yangxiaoda/ViewCrafter')
def run_inference():
    command = [
        "python", "/data/home/yangxiaoda/ViewCrafter/inference.py",
        "--image_dir", "/data/home/yangxiaoda/FlexCaM/vase.png",
        "--out_dir", "/data/home/yangxiaoda/FlexCaM/description-video",#请修改代码，我希望for i in range(1,300)
        "--traj_txt", "/data/home/yangxiaoda/FlexCaM/description-video/myviewcrafterinput.txt",
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
        # 使用 subprocess.run 执行命令
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command executed successfully!")
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the command!")
        print("Error:\n", e.stderr)

if __name__ == "__main__":
    run_inference()