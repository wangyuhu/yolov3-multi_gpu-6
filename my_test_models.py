import test

net_config_path = 'cfg/yolov3-original.cfg'
data_config_path = 'cfg/coco.data'
latest_weights_file = "/data3/wangyuhu/weights-yolo/weights-original/best.pt"
#latest_weights_file = 'tfc-best.pt'
batch_size = 8
img_size = 416

mAP, R, P = test.test(
            net_config_path,
            data_config_path,
            latest_weights_file,
            batch_size=batch_size,
            img_size=img_size,
        )