import matplotlib.pyplot as plt


def plot_all_curves(
    train_hist,
    val_hist,
    elbo_val_hist,
    train_rec_hist,
    val_rec_hist,
    train_kl_hist,
    val_kl_hist,
    train_z_hist,
    val_z_hist,
    C_values,
    train_capgap_hist,
    val_capgap_hist,
):
    """学習曲線を1枚の図にまとめて描画する"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. Loss
    axes[0].plot(train_hist, label="Train")
    axes[0].plot(val_hist, label="Val")
    axes[0].plot(elbo_val_hist, label="Val(ELBO)")
    axes[0].set_title("Learning Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # 2. Reconstruction Loss
    axes[1].plot(train_rec_hist, label="Train rec")
    axes[1].plot(val_rec_hist, label="Val rec")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()

    # 3. KL Divergence
    axes[2].plot(train_kl_hist, label="Train KL")
    axes[2].plot(val_kl_hist, label="Val KL")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].legend()

    # 4. Prediction Loss
    axes[3].plot(train_z_hist, label="Train zloss")
    axes[3].plot(val_z_hist, label="Val zloss")
    axes[3].set_title("Prediction Loss (ŷ vs y)")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("MSE")
    axes[3].legend()

    # 5. KL vs Capacity
    axes[4].plot(C_values, label="C (capacity)")
    axes[4].plot(val_kl_hist, label="Val KL")
    axes[4].set_title("KL vs Capacity C")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("Value")
    axes[4].legend()

    # 6. Capacity Gap
    axes[5].plot(train_capgap_hist, label="Train |KL-C|")
    axes[5].plot(val_capgap_hist, label="Val |KL-C|")
    axes[5].set_title("Capacity Gap")
    axes[5].set_xlabel("Epoch")
    axes[5].set_ylabel("|KL - C|")
    axes[5].legend()

    plt.tight_layout()
    plt.show()
