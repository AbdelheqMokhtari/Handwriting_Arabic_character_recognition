import os


img_train_path = r'Train'
img_data_array = []
class_name = []
for dir1 in os.listdir(img_train_path):
    for file in os.listdir(os.path.join(img_train_path, dir1)):


