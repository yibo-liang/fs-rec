from GAN import *
from face_helper import *
import face_helper

encoder = Encoder()
decoder_A = Decoder_ps()
decoder_B = Decoder_ps()

netGA = Model(x, decoder_A(encoder(x)))
netGB = Model(x, decoder_B(encoder(x)))

netDA = Discriminator(nc_D_inp)
netDB = Discriminator(nc_D_inp)

x = Input(shape=IMAGE_SHAPE)

# model_dir = format_dir("G:\\FakeAppData\\model_face_rec")
# try:
#     encoder.load_weights(model_dir + "encoder.h5")
#     decoder_A.load_weights(model_dir + "decoder_A.h5")
#     decoder_B.load_weights(model_dir + "decoder_B.h5")
#     netDA.load_weights(model_dir + "netDA.h5")
#     netDB.load_weights(model_dir + "netDB.h5")
#     print("model loaded.")
# except:
#     print("Weights file not found.")
#     pass

import face_recognition
from moviepy.editor import VideoFileClip

global prev_x0, prev_x1, prev_y0, prev_y1
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0

use_smoothed_mask = True
use_smoothed_bbox = True


def get_smoothed_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    x0 = int(0.65 * prev_x0 + 0.35 * x0)
    x1 = int(0.65 * prev_x1 + 0.35 * x1)
    y1 = int(0.65 * prev_y1 + 0.35 * y1)
    y0 = int(0.65 * prev_y0 + 0.35 * y0)
    return x0, x1, y0, y1


def set_global_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    prev_x0 = x0
    prev_x1 = x1
    prev_y1 = y1
    prev_y0 = y0


whom2whom = "AtoB"  # default trainsforming faceB to faceA

if whom2whom is "AtoB":
    path_func = path_abgr_B
elif whom2whom is "BtoA":
    path_func = path_abgr_A
else:
    print("whom2whom should be either AtoB or BtoA")

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


def process_frame(input_img):
    # modify this line to reduce input size
    # input_img = input_img[:, input_img.shape[1]//3:2*input_img.shape[1]//3,:]
    image = input_img
    faces = face_recognition.face_locations(image, model="cnn")  # model="cnn"

    if len(faces) == 0:
        result_img = image
        return result_img

    mask_map = np.zeros_like(image)

    global prev_x0, prev_x1, prev_y0, prev_y1
    global frames

    face = faces[0]
    if len(faces) > 1:
        encs = [face_helper._face_encodings(image, known_face_locations=face, model="small") for face in faces]
        predicts = model.predict_proba(encs)
        select = -1
        for i in range(len(predicts)):
            if predicts[i] == 0:
                select = i
                break
        if select == -1:
            return image
        face = faces[select]

    # coordinates
    (x0, y1, x1, y0) = face
    (x0, x1, y0, y1) = adjust_face(x0, x1, y0, y1, image.shape)
    h = x1 - x0
    w = y1 - y0

    # smoothing bounding box
    if use_smoothed_bbox:
        #     if frames != 0:
        x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1)
        set_global_coord(x0, x1, y0, y1)
    else:
        set_global_coord(x0, x1, y0, y1)
        #     frames += 1

    cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    roi_image = cv2_img[x0 + h // 15:x1 - h // 15, y0 + w // 15:y1 - w // 15, :]
    roi_size = roi_image.shape

    # smoothing mask
    if use_smoothed_mask:
        mask = np.zeros_like(roi_image)
        mask[h // 15:-h // 15, w // 15:-w // 15, :] = 255
        mask = cv2.GaussianBlur(mask, (15, 15), 10)
        orig_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

    ae_input = cv2.resize(roi_image, (128, 128)) / 255. * 2 - 1
    result = np.squeeze(np.array([path_func([[ae_input]])]))  # Change path_A/path_B here
    result_a = result[:, :, 0] * 255
    # result_a = np.clip(result_a * 1.5, 0, 255).astype('uint8')
    result_bgr = np.clip((result[:, :, 1:] + 1) * 255 / 2, 0, 255)
    result_a = cv2.GaussianBlur(result_a, (7, 7), 6)
    result_a = np.expand_dims(result_a, axis=2)
    result = (result_a / 255 * result_bgr + (1 - result_a / 255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    mask_map[x0 + h // 15:x1 - h // 15, y0 + w // 15:y1 - w // 15, :] = np.expand_dims(
        cv2.resize(result_a, (roi_size[1], roi_size[0])), axis=2)
    mask_map = np.clip(mask_map + .15 * input_img, 0, 255)

    result = cv2.resize(result, (roi_size[1], roi_size[0]))
    comb_img = np.zeros([input_img.shape[0], input_img.shape[1] * 2, input_img.shape[2]])
    comb_img[:, :input_img.shape[1], :] = input_img
    comb_img[:, input_img.shape[1]:, :] = input_img

    if use_smoothed_mask:
        comb_img[x0 + h // 15:x1 - h // 15, input_img.shape[1] + y0 + w // 15:input_img.shape[1] + y1 - w // 15,
        :] = mask / 255 * result + (1 - mask / 255) * orig_img
    else:
        comb_img[x0 + h // 15:x1 - h // 15, input_img.shape[1] + y0 + w // 15:input_img.shape[1] + y1 - w // 15,
        :] = result

    triple_img = np.zeros([input_img.shape[0], input_img.shape[1] * 3, input_img.shape[2]])
    triple_img[:, :input_img.shape[1] * 2, :] = comb_img
    triple_img[:, input_img.shape[1] * 2:, :] = mask_map
    # ========== Change rthe following line==========
    return triple_img
    # return comb_img[:, input_img.shape[1]:, :]  # return only result image
    # return comb_img  # return input and result image combined as one
    # return triple_img #return input,result and mask heatmap image combined as one


def load_frame_chunk_data(file):
    f = open(file, "r")
    lines = f.readlines()
    chunks = []
    for line in lines:
        start, end = line.split()
        chunks.append((int(start), int(end)))
    return chunks


chunk_data = load_frame_chunk_data("./model/frame_chunks.txt")

import re


def process_files(input_dir, output_dir, str_rule):
    input_dir = format_dir(input_dir)
    output_dir = format_dir(output_dir)
    for (start, end) in chunk_data:
        for i in range(start, end + 1):
            filename = str_rule % i
            try:
                image_data = cv2.imread(input_dir + filename)
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                data = process_frame(image_data)

                rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                Image.fromarray(rescaled).save(output_dir + filename)
                print(filename + " done.")
            except Exception as ex:

                print(traceback.format_exc())
                # or
                print(sys.exc_info()[0])
                print("cannot open file : " + input_dir + filename)


process_files("G:\\FakeAppData\\frames_128", "G:\\FakeAppData\\video\\result", "out%07d.png")
