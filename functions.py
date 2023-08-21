import sklearn

def r_squared(y, y_hat):
    """
    Calculating r2 score between actual y and predicted y. 
    """
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)


def test_regression_model(variable, df):
    """
    Test prediction performance of regression model by calculating total MAE on all test images, MAE on left / right eye images.
    """

    test_df_filtered = test_df_filtered[test_df_filtered[variable].notna()]
    df_left = test_df_filtered[test_df_filtered['left_right'] == '21015']
    df_right = test_df_filtered[test_df_filtered['left_right'] == '21016']

    test_df_filtered['predictions'] = test_df_filtered[variable].mean()
    mean_absolute_error = sklearn.metrics.mean_absolute_error(test_df_filtered[variable], test_df_filtered['predictions'])

    df_left['predictions'] = df_left[variable].mean()
    mean_absolute_error_left = sklearn.metrics.mean_absolute_error(df_left[variable], df_left['predictions'])

    df_right['predictions'] = df_right[variable].mean()
    mean_absolute_error_right = sklearn.metrics.mean_absolute_error(df_right[variable], df_right['predictions'])

    return mean_absolute_error, mean_absolute_error_left, mean_absolute_error_right