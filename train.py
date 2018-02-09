from GAN import *

# Path of training image
from config import *

# Get filenames
train_A = load_data(img_dirA)
train_B = load_data(img_dirB)

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

    loss_DA_display = errDA_sum
    loss_DB_display = errDB_sum
    loss_GA_display = errGA_sum
    loss_GB_display = errGB_sum

    if gen_iterations % display_iters == 0:
        print('[epoch=%d][gen=%d] Per epoch:: Loss_DA: %05f Loss_DB: %05f Loss_GA: %05f Loss_GB: %05f time: %01f'
              % (epoch, gen_iterations, loss_DA_display, loss_DB_display,
                 errGA_sum / display_iters, errGB_sum / display_iters, time.time() - t0))
        # if gen_iterations % display_iters == 0:  # clear_output every display_iters iters
        #     clear_output()

        # get new batch of images and generate results for visualization
        _, wA, tA = train_batchA.send(14)
        _, wB, tB = train_batchB.send(14)
        # showG(tA, tB, path_A, path_B)
        # showG(wA, wB, path_bgr_A, path_bgr_B)
        # showG_mask(tA, tB, path_mask_A, path_mask_B)

        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

        # Save models
        if gen_iterations % (display_iters * 3) == 0:
            print("Saved Model to " + model_dir)
            encoder.save_weights(model_dir + "encoder.h5")
            decoder_A.save_weights(model_dir + "decoder_A.h5")
            decoder_B.save_weights(model_dir + "decoder_B.h5")
            netDA.save_weights(model_dir + "netDA.h5")
            netDB.save_weights(model_dir + "netDB.h5")

print("Training Complete!")
