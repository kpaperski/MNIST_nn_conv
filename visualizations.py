import matplotlib.pyplot as plt

def my_plot(training_losses, training_acc, test_losses, test_acc, plot_title="Loss, train acc, test acc", step=100):
    training_iters = len(training_losses)
    iter_steps = [step * k for k in range(training_iters)]
    imh = plt.figure(1, figsize=(15, 14), dpi=160)
    final_acc = test_acc[-1]
    img_title = '{}, test acc = {:.4f}'.format(plot_title, final_acc)
    imh.suptitle(img_title)
    plt.subplot(221)
    plt.semilogy(iter_steps, training_losses, '-g', label='Training Loss')
    plt.title('Training Loss')
    plt.subplot(222)
    plt.plot(iter_steps, training_acc, '-r', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.subplot(223)
    plt.plot(iter_steps, test_losses, '-g', label='Test Loss')
    plt.title('Test Loss')
    plt.subplot(224)
    plt.plot(iter_steps, test_acc, '-r', label='Test Loss')
    plt.title('Test Accuracy')
    plt.subplots_adjust(top=0.88)
    plt.show()
    plot_file = "./plots/{}.png".format(plot_title.replace(" ", "_"))
    plt.savefig(plot_file)
