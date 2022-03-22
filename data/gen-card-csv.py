import glob
import cv2

class_list = (
    'As', 'Ks', 'Qs', 'Js', '10s', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s', 'Ah', 'Kh', 'Qh', 'Jh', '10h', '9h',
    '8h', '7h', '6h', '5h', '4h', '3h', '2h', 'Ad', 'Kd', 'Qd', 'Jd', '10d', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d',
    'Ac', 'Kc', 'Qc', 'Jc', '10c', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c'
)
image_width = 720
image_height = 720
cardlist = glob.glob("saved/*.txt")
for card in cardlist:
    with open(card, "r") as f:
        for line in f:
            # img = cv2.imread(card.replace(".txt",".jpg"))
            # image_width = img.shape[1]
            # image_height = img.shape[0]
            data = line.strip().split(" ")
            width = float(data[3]) * image_width
            height = float(data[4]) * image_height
            left = float(data[1]) * image_width - width / 2
            top = float(data[2]) * image_height - height / 2
            print("%s,%d,%d,%d,%d,%s" % (card.replace(".txt", ".jpg"),
                                         int(left),int(top),int(left+width),int(top+height),class_list[int(data[0])]))
