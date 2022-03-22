import glob
import json
import cv2
import numpy as np

classname_to_id = open("cards.names", "r").read().split("\n")

files = glob.glob("saved/*.json")
for fname in files:
    with open(fname, "r") as f:
        lm = json.load(f)

    img = cv2.imread(fname.replace(".json", ".jpg"))
    if img.shape[1] != img.shape[0]:
        new_img = np.zeros((img.shape[1], img.shape[1], 3), dtype=np.uint8)
        new_img[0:img.shape[0], :, :] = img
        cv2.imwrite(fname.replace(".json", ".jpg"), new_img)
        print("Write new image %s" % fname.replace(".json", ".jpg"))
        img = new_img

    darknet_width = 640
    darknet_height = 640

    for x in range(0, img.shape[1], darknet_width):
        for y in range(0, img.shape[0], darknet_height):
            x2 = x + darknet_width if x + darknet_width < img.shape[1] else img.shape[1]
            y2 = y + darknet_height if y + darknet_height < img.shape[0] else img.shape[0]

            if y2 - y != x2 - x:
                diff = (y2 - y) - (x2 - x)
                if diff > 0:
                    if x2 + diff <= img.shape[1]:
                        x2 += diff
                    elif x - diff >= 0:
                        x -= diff
                elif diff < 0:
                    if y2 - diff <= img.shape[0]:
                        y2 -= diff
                    elif y + diff >= 0:
                        y += diff
                # print("Fixed: Scale %3s matched %4d, %4d, %4d, %4d   S: %dx%d  I: %dx%d Diff %d  D: %dx%d" % ("is" if (y2-y)==(x2-x) else "not", x, y, x2, y2, x2 - x, y2 - y, frame.shape[1], frame.shape[0], diff,  self.darknet_width, self.darknet_height))

            pick_frame = img[y:y2, x:x2, :]

            f = None
            for shape in lm['shapes']:
                points = np.array(shape['points'])

                width = np.max(points[:, 0]) - np.min(points[:, 0])
                height = np.max(points[:, 1]) - np.min(points[:, 1])
                left = (np.min(points[:, 0]) + width / 2)
                top = (np.min(points[:, 1]) + height / 2)

                if left >= x and left + width <= x2 and top >= y and top + height <= y2:
                    left -= x
                    top -= y
                    if f is None:
                        cv2.imwrite(fname.replace(".json", "_%d_%d.jpg" % (x, y)), pick_frame)
                        f = open(fname.replace(".json", "_%d_%d.txt" % (x, y)), "w")
                    print("%d %f %f %f %f" % (
                        classname_to_id.index(shape['label']),
                        left / pick_frame.shape[1],
                        top / pick_frame.shape[0],
                        width / pick_frame.shape[1],
                        height / pick_frame.shape[0]
                    ), file=f)
            if f is not None:
                f.close()

    # img[0:int(image.shape[1]*9/16),:,3]

    f = open(fname.replace(".json", ".txt"), "w")
    for shape in lm['shapes']:
        points = np.array(shape['points'])

        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        print("%d %f %f %f %f" % (
            classname_to_id.index(shape['label']),
            (np.min(points[:, 0]) + width / 2) / img.shape[1],
            (np.min(points[:, 1]) + height / 2) / img.shape[0],
            width / img.shape[1],
            height / img.shape[0]
        ), file=f)
    f.close()

ftrain = open("train.txt", "w")
print("data/"+"\ndata/".join(glob.glob("cards/*.jpg")), file=ftrain)
print("data/" + "\ndata/".join(glob.glob("saved/*.jpg")), file=ftrain)
ftrain.close()

ftrain = open("valid.txt", "w")
print("data/" + "\ndata/".join(glob.glob("saved/*.jpg")), file=ftrain)
ftrain.close()
