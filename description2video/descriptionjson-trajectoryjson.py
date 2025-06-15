import os
import json
import pandas as pd
import numpy as np
import pdb

from tqdm import tqdm

class Text2VideoSet:
    def __init__(self, input_path, output_dir, fps=8):
        self.input_path = input_path
        self.output_dir = output_dir
        self.fps = fps


        with open(self.input_path, 'r') as f:
            self.JsonData = json.load(f)


    # 处理方向，返回变化率
    def process_direction(self, direction,speed):
        dr = 0.2
        if direction == 'backward' or direction == 'forward':
            if direction == 'forward':
                dr = -0.2
            if speed == 'low':
                dr *= 0.5
            return 0, 0, dr

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
            dphi, dtheta = -dphi*0.707, -dtheta*0.707
        elif direction == 'rightup45':
            dphi, dtheta = dphi*0.707, -dtheta*0.707
        elif direction == 'leftdown45':
            dphi, dtheta = -dphi*0.707, dtheta*0.707
        elif direction == 'rightdown45':
            dphi, dtheta = dphi*0.707, dtheta*0.707
        elif direction == 'leftup30':
            dphi, dtheta = -dphi*0.5, -dtheta*0.866
        elif direction == 'rightup30':
            dphi, dtheta = dphi*0.5, -dtheta*0.866
        elif direction == 'leftdown30':
            dphi, dtheta = -dphi*0.5, dtheta*0.866
        elif direction == 'rightdown30':
            dphi, dtheta = dphi*0.5, dtheta*0.866
        elif direction == 'leftup60':
            dphi, dtheta = -dphi*0.866, -dtheta*0.5
        elif direction == 'rightup60':
            dphi, dtheta = dphi*0.866, -dtheta*0.5
        elif direction == 'leftdown60':
            dphi, dtheta = -dphi*0.866, dtheta*0.5
        elif direction == 'rightdown60':
            dphi, dtheta = dphi*0.866, dtheta*0.5
        else:
            dphi, dtheta = 0, 0
        
        if speed == 'low':
            dphi *= 0.5
            dtheta *= 0.5

        return dphi, dtheta, 0

    # 调整姿态
    def tune_pose(self, key, records):
        phi, theta, r = 0, 0, 0
        out = np.array([[phi],[theta],[r]])
        # print(out.shape)
        last_time = 0
        for record in records:
            
            if last_time != record['starttime']: # 中间这段时间是静止的，补全
                still_frame_num = int(self.fps*(record['starttime']-last_time))
                still_out = np.array([[out[0,-1]]*still_frame_num, [out[1,-1]]*still_frame_num, [out[2,-1]]*still_frame_num])
                # print(type(still_out))
                # print(still_out.shape)
                # print(out.shape)
                out = np.concatenate((out,still_out), axis=1)
            frame_num = int(self.fps*(record['endtime'] - record['starttime']))
            
            dphi, dtheta, dr = self.process_direction(record['direction'], record['speed'])

            t_out = np.zeros((3, frame_num))
            t_out[:, 0] = out[:, -1] + 1.0/self.fps*np.array([dphi, dtheta, dr])

            for i in range(1, frame_num):
                t_out[:, i] = t_out[:, i-1] + 1.0/self.fps*np.array([dphi, dtheta, dr])

            out = np.concatenate((out, t_out), axis=1)

            last_time = record['endtime']
        # 分隔符为分号
        # np.savetxt(os.path.join(self.output_dir, key+'.txt'), out.T, fmt='%.3f')
        # np.savetxt(os.path.join(self.output_dir, key+'.txt'), out.T, fmt='%.3f', delimiter=' ')
        return out.T


    # 处理json
    def process(self):
        
        # 输出结果也存到json文件中,Json要有正常的格式，所有键包含在一个字典中
        with open(os.path.join(self.output_dir, 'mytrajectory.json'), 'w') as f:
            for key, records in tqdm(self.JsonData.items()):
                # print(key)
                out = self.tune_pose(key, records)
                # out中的数据是numpy类型，需要转换为list，小数保留三位。
                out = out.tolist()
                # import pdb; pdb.set_trace()
                for i in range(len(out)):
                    out[i] = [round(x, 3) for x in out[i]]
                self.JsonData[key] = out
                # break
            json.dump(self.JsonData, f, indent=4)


                




if __name__ == "__main__":
    t2vset = Text2VideoSet(input_path='/data/home/yangxiaoda/FlexCaM/description-video/mydecription.json',output_dir='/data/home/yangxiaoda/FlexCaM/description-video')
    t2vset.process()