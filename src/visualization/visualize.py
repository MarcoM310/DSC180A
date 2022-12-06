def PlotTrainValLoss(train_loss, val_loss, save_path, save_name):
    # Plot the training loss and validation loss
    epochs = np.arange(1, num_epochs + 1)

    df = pd.DataFrame(data={"train loss": tlosses, "valid loss": vlosses})
    sns.set(style="whitegrid")
    g = sns.FacetGrid(df, height=6)
    g = g.map(sns.lineplot, x=epochs, y=tlosses, marker="o", label="train")
    g = g.map(sns.lineplot, x=epochs, y=vlosses, color="red", marker="o", label="valid")
    g.set(ylim=(0, None))
    g.add_legend()
    plt.xticks(epochs)
    plt.show()
