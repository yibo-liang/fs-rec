from config import *
import glob
import random
import re
import face_recognition
import cv2

from sklearn import metrics
from sklearn.externals import joblib


def slash_sub(s):
    return re.sub(r'\\+', "/", s)


def format_dir(s="./"):
    s = slash_sub(s)
    if s.endswith("/"):
        return s
    else:
        return s + "/"


def list_files(dir):
    dir = format_dir(dir) + "*.*"
    return list(map(slash_sub, glob.glob(dir)))


#
def sample_files(filelist, n=10):
    d = {}
    l = len(filelist)

    if n <= l / 2:
        pass
    else:
        n = int(l / 2 + 1)

    while len(d.keys()) < n:
        i = random.randrange(0, l)

        f = filelist[i]
        d[f] = i
    return list(d.keys())


from  face_recognition.api import *


def _face_encodings(face_image, known_face_locations=None, num_jitters=1, model="large"):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = face_recognition.api._raw_face_landmarks(face_image, known_face_locations, model=model)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# Get list of randomly sampled face encodings
# Only for single face image
def get_encodings(dir, sampling=False, sample_count=10):
    from PIL import Image
    files = list_files(dir)

    if sampling:
        filenames = sample_files(filelist=files, n=sample_count)
    else:
        filenames = files
    encodings = []
    print(filenames)
    i = 0
    for file in filenames:
        img = face_recognition.load_image_file(file)
        rects = face_recognition.face_locations(img=img, number_of_times_to_upsample=1, model="cnn")

        if len(rects) > 0:
            (x0, y1, x1, y0) = rects[0]

            h = x1 - x0
            w = y1 - y0
            face_img = img[x0:x1, y0:y1, :]
            face_resized = cv2.resize(face_img, (256, 256))
            j = Image.fromarray(face_resized, mode='RGB')
            i += 1
            j.save("G:\\FakeAppData\\tmp\\" + str(i) + ".png")
            # rect_resized = 0, 0, 256, 256
            encs = _face_encodings(face_resized, num_jitters=30, known_face_locations=[(0, 0, 256, 256)])

            enc = encs[0]
            # print(enc)
            encodings.append(enc)
            print(1, end="", flush=True)
        else:
            print(0, end="")
    print()

    print("Got img encoding in :" + dir, " count = %d" % len(encodings))
    return encodings


#
# def test_face(enc, encs, compare_tolerance=0.35):
#     sum_true = 0
#     sum_dist = 0
#     res = face_recognition.compare_faces(known_face_encodings=encs, face_encoding_to_check=enc,
#                                          tolerance=compare_tolerance)
#     dists = face_recognition.face_distance(face_encodings=encs, face_to_compare=enc)
#     # print(res)
#     # print(dists)
#     i = 0
#     least_dist = 99999
#     for r in res:
#         sum_dist += dists[i]
#         print(i, dists[i])
#         if dists[i] < least_dist:
#             least_dist = dists[i]
#
#         i += 1
#         if r:
#             sum_true += 1
#     count = len(encs)
#     print("<", "%.02f" % sum_dist, "%d/%d" % (sum_true, count), end=">+", flush=True)
#     return sum_dist / count, sum_true / count


def score(res, dists):
    sum_dist = 0
    sum_match = 0
    l = len(res)
    for i in range(l):
        if res[i]:
            sum_match += 1
        sum_dist += dists[i]
    return sum_dist / l, sum_match / l


# def test_face(enc_to_test, encs_known, encs_other, tolerance=0.35):
#     merg_encs = encs_known + encs_other
#     res = face_recognition.compare_faces(known_face_encodings=merg_encs, face_encoding_to_check=enc_to_test,
#                                          tolerance=tolerance)
#     dists = face_recognition.face_distance(face_encodings=merg_encs, face_to_compare=enc_to_test)
#     l = len(encs_known)
#     res_known = res[:l]
#     res_other = res[l:]
#
#     dists_known = dists[:l]
#     dists_other = dists[l:]
#
#     avr_dist_known, avr_match_known = score(res_known, dists_known)
#     avr_dist_other, avr_match_other = score(res_other, dists_other)
#     print(" (%.2f,%.2f; %.2f,%.2f)" % (avr_dist_known, avr_match_known, avr_dist_other, avr_match_other))

def test_face(enc, encs_known, encs_other):
    pass


def create_text_encoding(input_dir, output_file, write="w"):
    wild_encs = get_encodings(input_dir, sampling=False)
    f = open(output_file, "w")
    for enc in wild_encs:
        line = ""
        for d in enc:
            line += str(d) + " "
        f.write(line + "\n")
    f.close()


def load_text_encodings(file):
    lines = open(file, "r").readlines()
    result = []
    for line in lines:
        s = line.split()
        if len(s) < 3:
            continue

        result.append([float(i) for i in s])
    return result


# create_text_encoding("G:\\FakeAppData\\taohong_new_raw\\aligned\\", "model/taohong.txt")
#

def Xy_from_data(list_of_datagroups):
    X = []
    y = []
    count = 0
    for encs in list_of_datagroups:
        X += encs
        y += [count for i in range(len(encs))]
        count += 1
    return X, y


def knn_classifier(list_encs):
    X, y = Xy_from_data(list_encs)
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=9)
    # merge_encs = knowning_encs + other_encs
    # merge_class = [0 for i in range(len(knowning_encs))] + [1 for i in range(len(other_encs))]
    model.fit(X=X, y=y)

    return model


def svm_classifier(list_encs):
    X, y = Xy_from_data(list_encs)
    from sklearn.svm import SVC

    clf = SVC(C=5, kernel="sigmoid", coef0=0.1, tol=0.00001)
    clf.fit(X, y)
    return clf


#
# global encs_A
# encs_A = get_encodings("G:\\FakeAppData\\g\\a", sampling=False, sample_count=10)
# global encs_zhang
# encs_zhang = get_encodings("G:\\FakeAppData\\g\\b", sampling=False, sample_count=10)




def is_face_in_img(enc, face_encs_known, face_encs_other):
    test_face(enc, face_encs_known, face_encs_other, 0.4)
    # print("(%.3f %.2f)" % (avr_dist, avr_truth), end=", ", flush=True)  # debug use


def adjust_face(x0, x1, y0, y1, shape):
    width = x1 - x0
    height = y1 - y0
    extra = 0.20
    _x0 = x0 - int(extra * width)
    _x1 = x1 + int(extra * width)
    _y0 = y0 - int(extra * height)
    _y1 = y1 + int(extra * height)

    width = _x1 - _x0
    height = _y1 - _y0
    if width > height:
        d = width - height
        _y0 -= d // 2
        _y1 += d // 2
    if height > width:
        d = height - width
        _x0 -= d // 2
        _x1 += d // 2

    if _x1 > shape[0] - 1:
        _x0 -= (_x1 - (shape[0] - 1))
        _x1 = shape[0] - 1
    if _y1 > shape[1] - 1:
        _y0 -= (_y1 - (shape[1] - 1))
        _y1 = shape[1] - 1
    if _x0 < 0:
        _x1 += abs(_x0)
        _x0 = 0
    if _y0 < 0:
        _y1 += abs(_y0)
        _y0 = 0

    width = _x1 - _x0
    height = _y1 - _y0
    if height > shape[1]:
        _y0 = 0
        _y1 = shape[1] - 1
        width = int(width * shape[1] / height)
        _x1 = _x0 + width
    if width > shape[0]:
        _x0 = 0
        _x1 = shape[0] - 1
        height = _y1 - _y0
        height = int(height * shape[0] / width)
        _y1 = _y0 + height
    return _x0, _x1, _y0, _y1


def test_all(testdir):
    # testdir = "G:\\FakeAppData\\face_test"

    taohong_encs = load_text_encodings("model/taohong.txt")
    taohong_all_encs = load_text_encodings("model/taohong_all.txt")

    wild_encs = load_text_encodings("./model/wild_encs.txt")
    red_others_encs = load_text_encodings("./model/red_others.txt")
    tielin_encs = load_text_encodings("./model/tielin.txt")
    yingzuo_encs = load_text_encodings("./model/yingzuo.txt")
    zhangluyi_encs = load_text_encodings("./model/zhangluyi_all.txt")

    classifiers = knn_classifier
    # classifiers = svm_classifier

    model = classifiers(
        [taohong_encs + taohong_all_encs, wild_encs, red_others_encs, tielin_encs, yingzuo_encs, zhangluyi_encs])

    random_encs = get_encodings(testdir)
    predict = model.predict(random_encs)
    print('{}'.format(predict))


# test_all("G:\\FakeAppData\\face_test")
#
def main():
    create_text_encoding("G:\\FakeAppData\\taohong_subset", "model/large_taohong_all.txt")

    create_text_encoding("G:\\FakeAppData\\taohong_subset", "model/large_taohong_all.txt")
    create_text_encoding("G:\\FakeAppData\\red\\zhangluyi", "model/large_zhangluyi_all.txt")

    create_text_encoding("G:\\FakeAppData\\red\\other", "model/large_red_others.txt")
    create_text_encoding("G:\\FakeAppData\\red\\yingzuo", "model/large_yingzuo.txt")
    create_text_encoding("G:\\FakeAppData\\red\\tielin", "model/large_tielin.txt")

    create_text_encoding("G:\\FakeAppData\\face_wild_100", "model/large_wild_faces.txt")


from PIL import Image
import traceback
import sys

import queue
import threading

global count
count = 0
global lock
lock = threading.Lock()
global file_queue
file_queue = queue.Queue()


def img_crop(image):
    faces = face_recognition.face_locations(image)
    result = []
    for face in faces:
        (x0, y1, x1, y0) = face
        (x0, x1, y0, y1) = adjust_face(x0, x1, y0, y1, image.shape)
        result.append(image[x0:x1, y0:y1, :])
    return result


import os


def deal_file(arg):
    file, output_dir = arg
    id = os.getpid()
    output_dir = format_dir(output_dir)
    print("%d : %s." % (id, file))

    try:

        image_data = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        data = img_crop(image_data)
        count = 0

        file_shortname = file.split("/")[-1].split(".")[0]
        for d in data:
            count += 1

            filename = file_shortname + "_" + str(count) + ".png"
            Image.fromarray(d).save(output_dir + filename)
            print(str(id) + " : " + output_dir + filename + " saved.")
    except Exception as ex:

        print(traceback.format_exc())
        # or
        print(sys.exc_info()[0])
        print("cannot process file : " + file)


def overfit_crop(input_dir, output_dir):
    input_dir = format_dir(input_dir)
    output_dir = format_dir(output_dir)

    for file in list_files(input_dir):
        # print("put file " + file)
        file_queue.put(file)


if __name__ == "__main__":

    sources = ["G:/Pictures/love/nichijo/Icecream",
               "G:/Pictures/love/nichijo/某次吃饭",
               "G:/Pictures/love/nichijo/雪藏",
               "G:/Pictures/love/phone/20150106",
               "G:/Pictures/love/phone/Camera-20140918",
               "G:/Pictures/love/pretty",
               "G:/Pictures/love/Sure",
               "G:/Pictures/love/Wedding/姚辉",
               "G:/Pictures/love/Wedding/好看的"]
    all_args = []

    output_dir, str_rule = "G:\\FakeAppData\\xuecong\\new", "xuecong%07d.png"
    i = 0
    for d in sources:
        print("Working in Dir = " + d)
        # overfit_crop(d, "G:\\FakeAppData\\xuecong\\new", "xuecong%07d.png")
        i += 1
        for f in list_files(d):
            arg = (f, output_dir)
            all_args.append(arg)

    threads = []
    from multiprocessing import Process, Pool

    Pool(processes=5).map(deal_file, all_args)
    # #
    # #
    # test_all("G:\\FakeAppData\\face_test", encs_A)
    # test_all("G:\\FakeAppData\\face_test", encs_zhang)
    # known_image = face_recognition.load_image_file(slash_sub("G:\\FakeAppData\\g\\b.png"))
    # unknown_image = face_recognition.load_image_file(slash_sub("G:\\FakeAppData\\g\\g.png"))
    # test_image = face_recognition.load_image_file(slash_sub("G:\\FakeAppData\\face_test\\out0000453.png"))
    #
    # biden_encoding = face_recognition.face_encodings(known_image)[0]
    # unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    # test_enc = face_recognition.face_encodings(test_image)[0]
    #
    # results = face_recognition.compare_faces([biden_encoding, unknown_encoding], test_enc)
    # print(results)
