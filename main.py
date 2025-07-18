
import numpy as np
import tensorflow as tf
from Model_ADHD_biotyping import VAE_ADHD_subtype,prepare_data,loss_function_2,loss_function_diff


if __name__ == '__main__':

    K_num = 300
    for K in range(K_num):

        ALFF_data,label_data = prepare_data()
        source_data = ALFF_data
        source_label = label_data

        number_ = 0
        count = 0
        num_epochs = 1000

        vae_ADHD = VAE_ADHD_subtype()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        mean_HC = np.mean(source_data[source_label.ravel() == 0], axis=0, keepdims=True)

        for epoch in range(num_epochs):
            Flag = 0
            if epoch == 0:
                Flag = 1
                mu_ave = tf.zeros([3, 20], dtype=tf.float32)

            with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:

                reconstructed_x, mu, logvar, class_output, closest_idx_all, mu_ave,z \
                    = vae_ADHD.forward_1(data=source_data, flag=Flag, mu_ave=mu_ave, label=source_label)

                MSE, KLD, CE =loss_function_2(reconstructed_data=reconstructed_x,
                                    data=source_data,
                                    label=closest_idx_all,
                                    class_output=class_output,
                                    mu=mu,
                                    logvar=logvar)
                MU, _ = loss_function_diff(mu_ave)

                reconstructed_mu_ave = vae_ADHD.decoder.call(mu_ave, mean_HC)
                Recon_MU,Recon_MU_mse = loss_function_diff(reconstructed_mu_ave)

                loss_total = 100 * MSE + 1 * KLD +  10000 * CE + 10000 * MU + 1 * Recon_MU - 1 * Recon_MU_mse

            grads1 = tape_1.gradient(loss_total, vae_ADHD.trainable_variables)
            optimizer.apply_gradients(zip(grads1, vae_ADHD.trainable_variables))
            print(
                f'K:{K + 1}--{epoch + 1}/{num_epochs},MSE:{MSE},CE:{CE},MU:{MU},Recon_MU:{Recon_MU},Recon_MU_mse:{Recon_MU_mse},{np.count_nonzero(closest_idx_all == 2)}')










