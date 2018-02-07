from config import *
import glob
import random
import re
import face_recognition
import cv2


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
    for file in filenames:
        img = face_recognition.load_image_file(file)
        rects = face_recognition.face_locations(img=img, number_of_times_to_upsample=1, model="cnn")

        if len(rects) > 0:
            (x0, y1, x1, y0) = rects[0]

            h = x1 - x0
            w = y1 - y0
            face_img = img[x0:x1, y0:y1, :]
            face_resized = cv2.resize(face_img, (256, 256))
            # j = Image.fromarray(face_resized, mode='RGB')
            # j.save(file + ".2.png")
            # rect_resized = 0, 0, 256, 256
            encs = face_recognition.face_encodings(face_img, num_jitters=3, known_face_locations=rects)

            enc = encs[0]
            # print(enc)
            encodings.append(enc)
            print(1, end="", flush=True)
        else:
            print(0, end="")
    print()

    print("Got img encoding in :" + dir, " count = %d" % len(encodings))
    return encodings


def test_face(enc, encs, compare_tolerance=0.35):
    sum_true = 0
    sum_dist = 0
    res = face_recognition.compare_faces(known_face_encodings=encs, face_encoding_to_check=enc,
                                         tolerance=compare_tolerance)
    dists = face_recognition.face_distance(face_encodings=encs, face_to_compare=enc)
    # print(res)
    # print(dists)
    i = 0
    least_dist = 99999
    for r in res:
        sum_dist += dists[i]
        print(i, dists[i])
        if dists[i] < least_dist:
            least_dist = dists[i]

        i += 1
        if r:
            sum_true += 1
    count = len(encs)
    print("<", "%.02f" % sum_dist, "%d/%d" % (sum_true, count), end=">+", flush=True)
    return sum_dist / count, sum_true / count


global encs_A
encs_A = get_encodings("G:\\FakeAppData\\g\\a", sampling=False, sample_count=10)
global encs_zhang
encs_zhang = get_encodings("G:\\FakeAppData\\g\\b", sampling=False, sample_count=10)


def is_face_in_img(enc, face_encs):
    avr_dist, avr_truth = test_face(enc, face_encs, 0.15)
    # print("(%.3f %.2f)" % (avr_dist, avr_truth), end=", ", flush=True)  # debug use


def test_all(testdir, encs):
    # testdir = "G:\\FakeAppData\\face_test"
    random_encs = get_encodings(testdir)

    for img in random_encs:
        is_face_in_img(img, encs)
    print()


#
# #
test_all("G:\\FakeAppData\\face_test", encs_A)
test_all("G:\\FakeAppData\\face_test", encs_zhang)
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
