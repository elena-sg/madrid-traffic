import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import TimeSeriesSplit
from graph_traffic.get_data import ubs_index
import matplotlib as mpl

from sklearn.metrics import mean_squared_error, mean_absolute_error

mpl.rcParams['axes.grid'] = False
mpl.rcParams.update({'font.size': 12})


def timeseries_cv(estimator, x, y, with_previous_timesteps=True, with_alpha=False):
    seq_len = x.shape[1]
    ts_cv = TimeSeriesSplit(
        n_splits=3,
        gap=seq_len
    )
    all_splits = list(ts_cv.split(x, y))

    train_losses = []
    test_losses = []
    estimators = []
    # predictions = {}
    if with_alpha:
        alphas = []
    for train, test in all_splits:
        x_train = x[train]
        y_train = y[train]#[..., [0]]
        x_test = x[test]
        y_test = y[test]#[..., [0]]

        estimator.fit(x_train, y_train)

        estimators.append(estimator)
        train_pred = estimator.predict(x_train)
        if with_previous_timesteps:
            loss = mean_absolute_error(train_pred, y_train[..., [0]]), mean_squared_error(train_pred, y_train[..., [0]])
        else:
            loss = mean_absolute_error(train_pred, y_train), mean_squared_error(train_pred, y_train)
        train_losses.append(loss)

        # if predict is not None:
        #     for sample in predict:
        #         if sample in test:
        #             predictions[sample] = estimator.predict(x[[sample]])[0]

        test_pred = estimator.predict(x_test)
        if with_previous_timesteps:
            loss = mean_absolute_error(test_pred, y_test[..., [0]]), mean_squared_error(test_pred, y_test[..., [0]])
        else:
            loss = mean_absolute_error(test_pred, y_test), mean_squared_error(test_pred, y_test)
        test_losses.append(loss)
        if with_alpha:
            alphas.append(estimator[-1].alpha_)

    if not with_alpha:
        return estimators, train_losses, test_losses
    else:
        return estimators, train_losses, test_losses, alphas


def plot_predictions(estimator, x, y, random_samples, ids_list, seq_len, name_save=None):
    _, ubs_dict = ubs_index(ids_list)
    labels_dict = {v: k for (k, v) in ubs_dict.items()}

    #random_samples = [x for x in random_samples if x in predictions.keys()]

    fig, ax = plt.subplots(len(ids_list), len(random_samples), figsize=(15, len(ids_list)*2), sharex="col",
                           sharey="row")
    if len(ids_list) == 1:
        ax = [ax]
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
            ax[sensor][i].plot(timestamps, true_values, label='True values', marker='.', zorder=-10)
            # ax[sensor][i].scatter(past_timestamps, x_target[:seq_len], marker='X', edgecolors='k', label='Previous observations',
            #                       c='#2ca02c', s=64)
            ax[sensor][i].scatter(future_timestamps, preds_sensor, marker='X', edgecolors='k', label='Predictions',
                                  c='#ff7f0e', s=64)

            if i == 0:
                #ax[sensor][i].set_ylabel(f"Cars/hour, sensor {labels_dict[sensor]}")
                ax[sensor][i].set_ylabel("Cars/hour")
            if sensor == len(ids_list) - 1:
                ax[sensor][i].set_xticks(timestamps, [f"{t % 24:.0f}" for t in timestamps])
                #ax[sensor][i].set_xlabel(f"Sample number {i}, hour")
                ax[sensor][i].set_xlabel("Hour of the day")
                ax[sensor][i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[sensor][i].legend()
            ax[sensor][i].set_xlabel(f"Sample number {i+1}")
    fig.tight_layout()
    if name_save is not None:
        plt.savefig(name_save)
    plt.show()


def plot_predictions_from_features(estimators, x, y, random_samples, ids_list, seq_len, features_from_index=3):
    _, ubs_dict = ubs_index(ids_list)
    fig, ax = plt.subplots(len(ids_list), len(random_samples), figsize=(25, len(ids_list)*2), sharex="col",
                           sharey="row")
    if len(ids_list) == 1:
        ax = [ax]
    for i, sample in enumerate(random_samples):
        past_timestamps = np.copy(x[sample][..., 0, 1])
        for j, t in enumerate(past_timestamps[1:]):
            if past_timestamps[j] > t:
                past_timestamps[j+1] = past_timestamps[j+1] + 24
        future_timestamps = [past_timestamps[-1] + 0.25 * h for h in range(1, seq_len + 1)]
        timestamps = np.concatenate([past_timestamps, future_timestamps])

        for sensor in ids_list:
            sensor_id = ubs_dict[sensor]
            x_features = x[sample][:, sensor_id, features_from_index:]
            y_features = y[sample][:, sensor_id, features_from_index:]
            features = np.concatenate([x_features, y_features], axis=0)
            x_target = x[sample][:, sensor_id, 0]
            y_target = y[sample][:, sensor_id, 0]
            true_values = np.concatenate([x_target, y_target])
            # preds_sensor = estimator.predict(x[sample][:, sensor, 1:]).ravel()
            preds_sensor = estimators[sensor_id].predict(features).ravel()
            ax[sensor_id][i].plot(timestamps, true_values, label='True values', marker='.', zorder=-10)
            # ax[sensor, i].scatter(past_timestamps, x_target[:seq_len], marker='X', edgecolors='k', label='Inputs',
            #                       c='#2ca02c', s=64)
            ax[sensor_id][i].scatter(timestamps, preds_sensor, marker='X', edgecolors='k', label='Predictions',
                                  c='#ff7f0e', s=64)

            if i == 0:
                ax[sensor_id][i].set_ylabel(f"Cars / hour, sensor {sensor}")
            if sensor_id == len(ids_list) - 1:
                ax[sensor_id][i].set_xticks(timestamps, [f"{t % 24:.0f}" for t in timestamps])
                ax[sensor_id][i].set_xlabel(f"Sample number {i}, hour")
                ax[sensor_id][i].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()


def train_estimators_by_sensor(ids_list, train_x, pipeline):
    estimators = {}
    _, ubs_dict = ubs_index(ids_list)
    train_losses = dict()
    test_losses = dict()
    for sensor in ids_list:
        i = ubs_dict[sensor]
        train_flat = train_x[:, :, i, :].reshape(-1, train_x.shape[-1])
        _, index = np.unique(train_flat, axis=0, return_index=True)
        index = np.sort(index)
        train_flat = train_flat[index]
        train_x_flat = train_flat[:, 3:]
        train_y_flat = train_flat[:, 0].ravel()
        estimators[i], train_losses[i], test_losses[i] = timeseries_cv(
            pipeline,
            train_x_flat, train_y_flat, with_previous_timesteps=False)
        #print(np.mean(train_losses), np.mean(test_losses))

    return estimators, train_losses, test_losses