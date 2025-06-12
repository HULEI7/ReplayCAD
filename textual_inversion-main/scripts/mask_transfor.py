import PIL
from PIL import Image, ImageFilter, ImageOps
import random
import numpy as np
import cv2
from skimage import transform, io, util
from scipy.ndimage import label, rotate, generate_binary_structure

def random_rorate(image):
    
    angle=random.randint(0,360)
    rotated_img = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    return rotated_img

def random_3direcions_rotate(image):
    
    id_x = random.randint(0,5)
    
    if id_x==0:
        image = image.rotate(90)

    elif id_x==1:
        image = image.rotate(180)

    elif id_x==2:
        image = image.rotate(270)

    elif id_x==3:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    elif id_x==4:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    else:
        pass
    
    return image



def merge_images(images, n_per_row):
    # 确定每张图片的尺寸
    img_width, img_height = images[0].size

    # 计算合并后的大图尺寸
    total_width = img_width * n_per_row
    total_height = img_height * (len(images) // n_per_row + (1 if len(images) % n_per_row > 0 else 0))

    # 创建一个新的空白图像，用于拼接
    new_image = Image.new('RGB', (total_width, total_height))

    # 将每张图片粘贴到新图像的相应位置
    for i, img in enumerate(images):
        x = (i % n_per_row) * img_width
        y = (i // n_per_row) * img_height
        new_image.paste(img, (x, y))

    return new_image


def little_rorate_and_move(image,angles=2,distance=0.05,tranpose=False):
    
    ##小角度+小平移
    #mvtec中的晶体管
    
    #旋转
    #transistor
    # angle=random.randint(0,2)
    
    if tranpose and (random.randint(0,1)==1):
        image = image.rotate(180)
        
    #cashew
    angle=random.randint(0,angles)
    
    angle=random.choice([angle, 360-angle])
    rotated_img = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    #平移
    
    mask = np.array(rotated_img)
    original_height, original_width = mask.shape[:2]

    # 设定要移动的像素数
    shift_pixels = min(random.randint(0,int(rotated_img.size[0]*distance)),original_height // 2, original_width // 2)

    # 随机选择扩展和收缩的方向 (0: 上, 1: 下, 2: 左, 3: 右)
    expand_direction = np.random.choice([0, 1, 2, 3])

    # 定义相应的扩展和收缩操作
    if expand_direction == 0:  # 上扩展，下收缩
        # 上扩展
        mask = np.pad(mask, ((shift_pixels, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        # 下收缩
        mask = mask[:original_height, :, :]
    elif expand_direction == 1:  # 下扩展，上收缩
        # 下扩展
        mask = np.pad(mask, ((0, shift_pixels), (0, 0), (0, 0)), mode='constant', constant_values=0)
        # 上收缩
        mask = mask[-original_height:, :, :]
    elif expand_direction == 2:  # 左扩展，右收缩
        # 左扩展
        mask = np.pad(mask, ((0, 0), (shift_pixels, 0), (0, 0)), mode='constant', constant_values=0)
        # 右收缩
        mask = mask[:, :original_width, :]
    else:  # 右扩展，左收缩
        # 右扩展
        mask = np.pad(mask, ((0, 0), (0, shift_pixels), (0, 0)), mode='constant', constant_values=0)
        # 左收缩
        mask = mask[:, -original_width:, :]

    # 将结果保存为新的图像
    shifted_mask_image = Image.fromarray(mask.astype(np.uint8))  # 确保数据类型是uint8
    
    return shifted_mask_image



def visa_candle(image,distance=20, rotate=False, angle_range=(-10, 10)):
    
    mask_np_rgb = np.array(image)

    # 转换为灰度图以便进行轮廓检测
    mask_np_gray = cv2.cvtColor(mask_np_rgb, cv2.COLOR_RGB2GRAY)

    # 使用OpenCV来查找轮廓
    _, thresh = cv2.threshold(mask_np_gray, 128, 255, cv2.THRESH_BINARY)

    # 查找轮廓（cv2.RETR_EXTERNAL表示只检测外部轮廓，cv2.CHAIN_APPROX_SIMPLE表示压缩轮廓）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 随机选择一个轮廓
    selected_contour = random.choice(contours)

    # 创建一个与mask大小相同的空白mask（灰度版）
    temp_mask = np.zeros_like(mask_np_gray)

    # 绘制轮廓到temp_mask中，填充颜色为255（白色）
    cv2.drawContours(temp_mask, [selected_contour], -1, 255, thickness=cv2.FILLED)

    # 将轮廓区域扩展到3个通道（RGB），以便删除原RGB图像中的对应部分
    temp_mask_rgb = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2RGB)

    # 将选定的轮廓从原始RGB图像中删除（设为黑色）
    mask_np_rgb = cv2.bitwise_and(mask_np_rgb, cv2.bitwise_not(temp_mask_rgb))

    # 设定平移的随机方向和距离范围 (dx, dy)
    dx = random.randint(-distance, distance)  # 水平移动范围：-20 到 20 像素
    dy = random.randint(-distance, distance)  # 垂直移动范围：-20 到 20 像素

    # 平移选中的轮廓
    # 创建平移矩阵
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # 对当前物体的mask进行平移
    shifted_mask = cv2.warpAffine(temp_mask, M, (mask_np_gray.shape[1], mask_np_gray.shape[0]))
    
    
    if rotate:
        # 生成随机的旋转角度
        angle = random.uniform(angle_range[0], angle_range[1])

        # 获取图像中心点
        center = (shifted_mask.shape[1] // 2, shifted_mask.shape[0] // 2)

        # 创建旋转矩阵
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 旋转轮廓
        shifted_mask = cv2.warpAffine(shifted_mask, M_rotate, (mask_np_gray.shape[1], mask_np_gray.shape[0]))

    # 将平移后的轮廓转换回RGB格式
    shifted_mask_rgb = cv2.cvtColor(shifted_mask, cv2.COLOR_GRAY2RGB)

    # 将移动后的物体添加到原始RGB图像中
    final_mask_rgb_np = cv2.bitwise_or(mask_np_rgb, shifted_mask_rgb)

    # 转换为PIL图像并保存
    final_mask_image = Image.fromarray(final_mask_rgb_np)
    
    
    return final_mask_image


def radomreset(img):
    
    # 提取mask层（白色物体为目标，黑色背景）
    gray_img = img.convert('L')
    binary_img = np.array(gray_img) > 128  # 生成二值图像，白色区域为True，黑色区域为False

    # 使用连通域算法识别白色物体
    from scipy.ndimage import label
    labeled_array, num_features = label(binary_img)

    # 创建新的全黑背景（RGB格式）
    new_img = Image.new('RGB', img.size, (0, 0, 0))

    # 获取每个物体的边界
    objects = []
    for i in range(1, num_features + 1):
        # 获取物体的mask
        mask = labeled_array == i
        coords = np.argwhere(mask)
        
        # 获取物体的最小矩形边界
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 保存物体的图像区域
        object_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        objects.append((object_img, (x_max - x_min + 1, y_max - y_min + 1)))  # (图像, 尺寸)

    # 随机排列这些物体
    positions = []
    for obj_img, (width, height) in objects:
        # 尝试找到随机的位置放置该物体，且不重叠
        for _ in range(100):  # 最多尝试100次
            x = random.randint(0, img.size[0] - width)
            y = random.randint(0, img.size[1] - height)
            new_box = (x, y, x + width, y + height)
            
            # 检查是否与已经放置的物体重叠
            overlap = False
            for pos in positions:
                if (new_box[0] < pos[2] and new_box[2] > pos[0] and 
                    new_box[1] < pos[3] and new_box[3] > pos[1]):
                    overlap = True
                    break
            if not overlap:
                positions.append(new_box)
                break

    # 将物体粘贴到新的位置
    for (obj_img, _), pos in zip(objects, positions):
        new_img.paste(obj_img, pos[:2])
    
    return new_img


if __name__ == "__main__":
    
    mask_image = Image.open("SAM/data/mvtec/transistor/001.png").convert("RGB")
    mask_image = little_rorate_and_move(mask_image,angles=5,distance=0.1)
    mask_image.save("test.png")





    

    
    
    
    
