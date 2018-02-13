from GAN import *

from random import shuffle

# Get filenames
train_A = load_data(img_dirA)
train_B = load_data(img_dirB)
shuffle(train_A)
shuffle(train_B)

assert len(train_A), "No image found in " + str(img_dirA)
assert len(train_B), "No image found in " + str(img_dirB)

# In[ ]:

expect_loss_A = 0.02
expect_loss_B = 0.02

t0 = time.time()
niter = 150
gen_iterations = 0
epoch = 0
errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

display_iters = 10
train_batchA = minibatchAB(train_A, batchSize)
train_batchB = minibatchAB(train_B, batchSize)

min_loss_A = 999
last_epoch = -1
min_gloss_A = 999


def save(suf=""):
    print("Saved Model to " + model_dir)
    encoder.save_weights(model_dir + "encoder%s.h5" % suf)
    decoder_A.save_weights(model_dir + "decoder_A%s.h5" % suf)
    decoder_B.save_weights(model_dir + "decoder_B%s.h5" % suf)
    netDA.save_weights(model_dir + "netDA%s.h5" % suf)
    netDB.save_weights(model_dir + "netDB%s.h5" % suf)


while True:
    epoch, warped_A, target_A = next(train_batchA)
    epoch, warped_B, target_B = next(train_batchB)

    # Train dicriminators for one batch
    if gen_iterations % 1 == 0:
        errDA = netDA_train([warped_A, target_A])
        errDB = netDB_train([warped_B, target_B])
    errDA_sum += errDA[0]
    errDB_sum += errDB[0]

    if gen_iterations == 5:
        print("working.")

    # Train generators for one batch
    errGA = netGA_train([warped_A, target_A])
    errGB = netGB_train([warped_B, target_B])
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    gen_iterations += 1

    if epoch > last_epoch:
        last_epoch = epoch
        # if gen_iterations % display_iters == 0:

        loss_DA_display = errDA_sum / display_iters
        loss_DB_display = errDB_sum / display_iters
        loss_GA_display = errGA_sum / display_iters
        loss_GB_display = errGB_sum / display_iters
        print('[epoch=%d][gen=%d] Per epoch:: Loss_DA: %05f Loss_DB: %05f Loss_GA: %05f Loss_GB: %05f time: %01f'
              % (epoch, gen_iterations, loss_DA_display, loss_DB_display,
                 loss_GA_display, loss_GB_display, time.time() - t0))
        # if gen_iterations % display_iters == 0:  # clear_output every display_iters iters
        #     clear_output()

        # get new batch of images and generate results for visualization
        _, wA, tA = train_batchA.send(14)
        _, wB, tB = train_batchB.send(14)
        showG(tA, tB, path_A, path_B, temp_show_dir + "combined.png")
        showG(wA, wB, path_bgr_A, path_bgr_B, temp_show_dir + "generated.png")
        showG_mask(tA, tB, path_mask_A, path_mask_B, temp_show_dir + "mask.png")

        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

        # Save models
        if loss_DA_display < min_loss_A and epoch > 5:
            min_loss_A = loss_DA_display
            # if gen_iterations % (display_iters * 3) == 0:
            save()

        if loss_GA_display < min_gloss_A:
            min_gloss_A = loss_GA_display
            save("_lowg")

        save("_epoch")
print("Training Complete!")
