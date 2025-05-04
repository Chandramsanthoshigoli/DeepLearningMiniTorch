def default_log_fn(epoch, total_loss, correct, losses) -> None:
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def default_plot_fn(fig, losses) -> None:
    pass
