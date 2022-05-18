import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import TimeSeriesSplit
from graph_traffic.get_data import ubs_index


def timeseries_cv(estimator, x, y, with_previous_timesteps=True):
    seq_len = x.shape[1]
    ts_cv = TimeSeriesSplit(
        n_splits=10,
        gap=seq_len
    )
    all_splits = list(ts_cv.split(x, y))

    train_losses = []
    test_losses = []
    # predictions = {}
    for train, test in all_splits:
        x_train = x[train]
        y_train = y[train]#[..., [0]]
        x_test = x[test]
        y_test = y[test]#[..., [0]]

        estimator.fit(x_train, y_train)

        train_pred = estimator.predict(x_train)
        if with_previous_timesteps:
            loss = np.abs(train_pred - y_train[..., [0]]).mean()
        else:
            loss = np.abs(train_pred - y_train).mean()
        train_losses.append(loss)

        # if predict is not None:
        #     for sample in predict:
        #         if sample in test:
        #             predictions[sample] = estimator.predict(x[[sample]])[0]

        test_pred = estimator.predict(x_test)
        if with_previous_timesteps:
            loss = np.abs(test_pred - y_test[..., [0]]).mean()
        else:
            loss = np.abs(test_pred - y_test).mean()
        test_losses.append(loss)

    return estimator, train_losses, test_losses


def plot_predictions(estimator, x, y, random_samples, ids_list, seq_len):
    _, ubs_dict = ubs_index(ids_list)
    labels_dict = {v: k for (k, v) in ubs_dict.items()}

    #random_samples = [x for x in random_samples if x in predictions.keys()]

    fig, ax = plt.subplots(len(ids_list), len(random_samples), figsize=(25, 10), sharex="col",
                           sharey="row")
    for i, sample in enumerate(random_samples):

        predictions = estimator.predict(x[[sample]])[0]

        x_values = x[sample][..., 0]
        y_values = y[sample][..., 0]

        past_timestamps = np.copy(x[sample][..., 0, 1])
        for j, t in enumerate(past_timestamps[1:]):
            if past_timestamps[j] > t:
                past_timestamps[j+1] = past_timestamps[j+1] + 24
        future_timestamps = [past_timestamps[-1] + 0.25 * h for h in range(1, seq_len + 1)]
        timestamps = np.concatenate([past_timestamps, future_timestamps])
        for sensor in range(len(ids_list)):
            x_target = x_values[:, sensor]
            y_target = y_values[:, sensor]
            true_values = np.concatenate([x_target, y_target])
            preds_sensor = predictions[:, sensor, 0].ravel()
            ax[sensor, i].plot(timestamps, true_values, label='True values', marker='.', zorder=-10)
            ax[sensor, i].scatter(past_timestamps, x_target[:seq_len], marker='X', edgecolors='k', label='Inputs',
                                  c='#2ca02c', s=64)
            ax[sensor, i].scatter(future_timestamps, preds_sensor, marker='X', edgecolors='k', label='Predictions',
                                  c='#ff7f0e', s=64)

            if i == 0:
                ax[sensor, i].set_ylabel(f"Cars/hour, sensor {labels_dict[sensor]}")
            if sensor == len(ids_list) - 1:
                ax[sensor, i].set_xticks(timestamps, [f"{t % 24:.0f}" for t in timestamps])
                ax[sensor, i].set_xlabel(f"Sample number {i}, hour")
                ax[sensor, i].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()


def plot_predictions_from_features(estimators, x, y, random_samples, ids_list, seq_len):
    _, ubs_dict = ubs_index(ids_list)
    fig, ax = plt.subplots(len(ids_list), len(random_samples), figsize=(25, 10), sharex="col",
                           sharey="row")
    for i, sample in enumerate(random_samples):
        past_timestamps = np.copy(x[sample][..., 0, 1])
        for j, t in enumerate(past_timestamps[1:]):
            if past_timestamps[j] > t:
                past_timestamps[j+1] = past_timestamps[j+1] + 24
        future_timestamps = [past_timestamps[-1] + 0.25 * h for h in range(1, seq_len + 1)]
        timestamps = np.concatenate([past_timestamps, future_timestamps])

        for sensor in ids_list:
            sensor_id = ubs_dict[sensor]
            x_features = x[sample][:, sensor_id, 1:]
            y_features = y[sample][:, sensor_id, 1:]
            features = np.concatenate([x_features, y_features], axis=0)
            x_target = x[sample][:, sensor_id, 0]
            y_target = y[sample][:, sensor_id, 0]
            true_values = np.concatenate([x_target, y_target])
            # preds_sensor = estimator.predict(x[sample][:, sensor, 1:]).ravel()
            preds_sensor = estimators[sensor_id].predict(features).ravel()
            ax[sensor_id, i].plot(timestamps, true_values, label='True values', marker='.', zorder=-10)
            # ax[sensor, i].scatter(past_timestamps, x_target[:seq_len], marker='X', edgecolors='k', label='Inputs',
            #                       c='#2ca02c', s=64)
            ax[sensor_id, i].scatter(timestamps, preds_sensor, marker='X', edgecolors='k', label='Predictions',
                                  c='#ff7f0e', s=64)

            if i == 0:
                ax[sensor_id, i].set_ylabel(f"Cars/hour, sensor {sensor}")
            if sensor_id == len(ids_list) - 1:
                ax[sensor_id, i].set_xticks(timestamps, [f"{t % 24:.0f}" for t in timestamps])
                ax[sensor_id, i].set_xlabel(f"Sample number {i}, hour")
                ax[sensor_id, i].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()