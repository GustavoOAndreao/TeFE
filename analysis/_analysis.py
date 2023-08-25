import pandas as pd
import numpy as np
import csv
import pickle
from pyspark.sql import SparkSession
import json
import plotly.express as px
import plotly.graph_objects as go
import winsound
import os
from names import names

from icecream import ic
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from operator import add
import kaleido
import plotly.io as pio

def load_into_df(_file):
    if 'pkl' in _file:
        with open(_file, 'rb') as f:
            df = pickle.load(f)
    elif 'csv' in _file:
        df = pd.read_csv (_file)
    elif 'feather' in _file:
        df = pd.read_feather(_file)
    elif 'parquet' in _file:
        df = pd.read_parquet(_file)
    else:
        print('not loaded, format unsupported')
        df = pd.DataFrame(['Not loaded', 'format unsupported'])

    return df


def iqr_plotly(var, dataframe, x_axis, groupby=None, remove_outliers=True, _sum=False, _old=False, base_100=False):

    if groupby is None:
        groupby = [x_axis]

    # var = 'capacity'
    # name_o_var = 'Number of adaptations'

    # .groupby(['period', 'seed'],as_index=False)['Lumps'].sum().groupby(['period'], as_index=False).quantile(0.5)

    x =     list(dataframe.groupby([x_axis], as_index=False)[x_axis].max()[x_axis])
    x_rev = list(x[ : : -1])

    # _sum = True

    if _sum is False:
        DF = dataframe.groupby(['period', 'seed'],as_index=False)[var].sum() if _sum is True else dataframe.groupby(['period', 'seed'],as_index=False)[var].mean()

        _y_max=  list(dataframe.groupby(groupby, as_index=False)[var].quantile(1)  [var])
        y_upper= list(dataframe.groupby(groupby, as_index=False)[var].quantile(.75)[var])
        y_median=list(dataframe.groupby(groupby, as_index=False)[var].median()     [var])
        y_mean=  list(dataframe.groupby(groupby, as_index=False)[var].mean()       [var])
        y_bottom=list(dataframe.groupby(groupby, as_index=False)[var].quantile(.25)[var])
        _y_min=  list(dataframe.groupby(groupby, as_index=False)[var].quantile(0)  [var])

        """ OLD
        _y_max=  list(DF.groupby(groupby, as_index=False)[var].quantile(1)  [var])
        y_upper= list(DF.groupby(groupby, as_index=False)[var].quantile(.75)[var])
        y_median=list(DF.groupby(groupby, as_index=False)[var].median()     [var])
        y_mean=  list(DF.groupby(groupby, as_index=False)[var].mean()       [var])
        y_bottom=list(DF.groupby(groupby, as_index=False)[var].quantile(.25)[var])
        _y_min=  list(DF.groupby(groupby, as_index=False)[var].quantile(0)  [var])
        """

    else:

        _y_max=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].sum().groupby(['period'], as_index=False).quantile(1)  [var])
        y_upper= list(dataframe.groupby(['period', 'seed'],as_index=False)[var].sum().groupby(['period'], as_index=False).quantile(.75)[var])
        y_median=list(dataframe.groupby(['period', 'seed'],as_index=False)[var].sum().groupby(['period'], as_index=False).median()     [var])
        y_mean=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].sum().groupby(['period'], as_index=False).mean()       [var])
        y_bottom=list(dataframe.groupby(['period', 'seed'],as_index=False)[var].sum().groupby(['period'], as_index=False).quantile(.25)[var])
        _y_min=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].sum().groupby(['period'], as_index=False).quantile(0)  [var])

        """
        _y_max=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(1).groupby(['period'], as_index=False)   .sum()[var])
        y_upper= list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(.75).groupby(['period'], as_index=False) .sum()[var])
        y_median=list(dataframe.groupby(['period', 'seed'],as_index=False)[var].median().groupby(['period'], as_index=False)      .sum()[var])
        y_mean=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].mean().groupby(['period'], as_index=False)        .sum()[var])
        y_bottom=list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(.25).groupby(['period'], as_index=False) .sum()[var])
        _y_min=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(0).groupby(['period'], as_index=False)   .sum()[var])
        """

    if remove_outliers is True:
        y_max = []
        y_min = []
        for i in range(0,len(_y_max)):
            y_max.append(min(max(_y_max[i], y_upper[i], y_median[i], y_bottom[i], _y_min[i]),
                             y_upper[i] + 1.5*(y_upper[i]-y_bottom[i])))
            y_min.append(max(min(_y_max[i], y_upper[i], y_median[i], y_bottom[i], _y_min[i]),
                             y_bottom[i] - 1.5*(y_upper[i]-y_bottom[i])))
    else:
        y_max = _y_max
        y_min = _y_min

    if base_100 is True:
        y_max = [ 100 * i/max(y_max[0], 0.1 / 10 ** 25) for i in y_max]
        y_upper = [ 100 * i/max(y_upper[0], 0.1 / 10 ** 25) for i in y_upper]
        y_median = [ 100 * i/max(y_median[0], 0.1 / 10 ** 25) for i in y_median]
        y_mean = [ 100 * i/max(y_mean[0], 0.1 / 10 ** 25) for i in y_mean]
        y_bottom = [ 100 * i/max(y_bottom[0], 0.1 / 10 ** 25) for i in y_bottom]
        y_min = [ 100 * i/max(y_min[0], 0.1 / 10 ** 25) for i in y_min]

    y_min = y_min[: : -1]
    y_bottom=y_bottom[: : -1]

    return x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min


def simple_graph(name_o_graph, var, dataframe, x_axis, groupby=None, remove_outliers=True, show=True, log_y=False, log_x=False, color='232,126,4', _sum=False):

    x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min = iqr_plotly(var, dataframe, x_axis, groupby, remove_outliers, _sum=_sum)

    fig = go.Figure(go.Scatter(name="IQR",
                             x=x + x_rev,
                             y=y_upper+y_bottom,
                             fill='toself',
                             fillcolor='rgba(%s,0.4)' % color,
                             line=dict(color='rgba(255,255,255,0)')))

    fig.add_trace(go.Scatter(name="Max and min",
                             x=x + x_rev,
                             y=y_max+y_min,
                             fill='toself',
                             fillcolor='rgba(%s,0.2)' % color,
                             line=dict(color='rgba(255,255,255,0)')))

    fig.add_trace(go.Scatter(name="Median",
                             x=x + x_rev,
                             y=y_median,
                            mode='lines',
                             line=dict(color='rgba(%s,1)' % color, dash='dot')))

    fig.update_yaxes(type="log") if log_y is True else None
    fig.update_xaxes(type="log") if log_x is True else None
    fig.update_layout(barmode='overlay', template="simple_white")

    title = str(name_o_graph)

    fig.update_layout(
        title=title,
        xaxis_title=str(x_axis),
        yaxis_title=str(name_o_graph.split(' - ')[2]),
        barmode='overlay',
        template="simple_white")

    # fig.show()

    file_name = title + ".html"
    pathfile='Figures/'

    fig.write_html(pathfile + file_name)

    return fig.show() if show is True else None


def simple_add_trace(name_o_var, var, dataframe, x_axis, groupby=None, remove_outliers=True, color=None, _sum=False,
                     base_100=False, _dash=None, _pattern=None):

    if color is None:
        color = '232,126,4'

    if _dash is None:
        _dash = 'dot'

    x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min = iqr_plotly(var,
                                                                             dataframe,
                                                                             x_axis, groupby, remove_outliers,
                                                                             base_100=base_100, _sum=_sum)

    IQR = dict(name="IQR (%s)" % name_o_var,
               x=x + x_rev,
               y=y_upper+y_bottom,
               fill='toself',
               fillcolor='rgba(%s,0.4)' % color,
               line=dict(color='rgba(255,255,255,0)'))

    max_min = dict(name="Max and min (%s)" % name_o_var,
                   x=x + x_rev,
                   y=y_max+y_min,
                   fill='toself',
                   fillcolor='rgba(%s,0.2)' % color,
                   line=dict(color='rgba(255,255,255,0)'))

    median = dict(name="Median (%s)" % name_o_var,
                  x=x + x_rev,
                  y=y_median,
                  mode='lines',
                  line=dict(color='rgba(%s,1)' % color, dash=_dash))

    return IQR, max_min, median


def scatter_graph(full_x, full_y, speed=False, groupby=None, show=True, time=0, normalization=True, # If normalization is false we standardize
                  _name=None, _line=False):

    global agent_dict

    update = {}

    if groupby is None:
        groupby = 'period'

    # dataframe = agents_df  # .loc[agents_df['genre'] == 'TP']

    x_var, DF_x, _normal_x, axis_x = full_x[0], full_x[1], full_x[2],full_x[3]
    y_var, DF_y, _normal_y, axis_y = full_y[0], full_y[1], full_y[2],full_y[3]

    # DF_x = DF_x.groupby(['period', 'seed'],as_index=False)[x_var].sum() if _sum_x is True else DF_x.groupby(['period', 'seed'],as_index=False)[x_var].mean()
    # DF_y = DF_y.groupby(['period', 'seed'],as_index=False)[y_var].sum() if _sum_y is True else DF_y.groupby(['period', 'seed'],as_index=False)[y_var].mean()

    x = list(DF_x.groupby(groupby, as_index=False)[x_var].mean()[x_var])
    y = list(DF_y.groupby(groupby, as_index=False)[y_var].mean()[y_var])
    # print(x,y)

    regression = True
    if speed is True:
        """priv_goal[period] - priv_goal[period - 1]
                                 ) / priv_goal[period - 1] if priv_goal[period - 1] > 0 else 1"""
        x = [(x[i] - x[i-time])/x[i-time] if x[i-time] != 0 else 'to_remove' for i in range(time, len(x))]
        # x[0] = 0
        y = [(y[i] - y[i-time])/y[i-time] if y[i-time] != 0 else 'to_remove' for i in range(time, len(y))]
        # y[0] = 0

        to_pop = []
        _max = len(x)-1
        for _ in range(0, len(x)-1):
            if x[_] == 'to_remove' or y[_] == 'to_remove':
                to_pop.append(_)

        to_pop.reverse()
        if len(to_pop) == len(x):
            x = y = [0]  # for i in range(len(x))]
            regression = False

        else:
            for i in to_pop:
                del x[i]
                del y[i]

    else:
        # We don't normalize if we want the speeds
        if _normal_x is True or _normal_y is True:
            if normalization is True:

                x = [x[i]- np.mean(x)/np.std(x) for i in range(len(x))] if _normal_x is True else x
                y = [y[i]- np.mean(y)/np.std(y) for i in range(len(y))] if _normal_y is True else y

            else:
                x = [(x[i] - min(x))/max(0.1 / 10 ** 25, (max(x) - min(x))) for i in range(len(x))] if _normal_x is True else x
                y = [(y[i] - min(y))/max(0.1 / 10 ** 25, (max(y) - min(y))) for i in range(len(y))] if _normal_y is True else y

    color = list(DF_x.groupby(['period'], as_index=False).max()['period'])[time:]
    # color = [i/time if i % time == 0 else 'to_remove' for i in range(len(color))]
    # color = list(filter('to_remove'.__ne__, color))
    # print(color)

    fig = go.Figure(go.Scatter(x = x,
                               y = y,
                               mode='markers',
                               marker=dict(
                                   color= color,
                                   colorscale='Viridis',
                                   line_width=1,
                                   showscale=True))) if _line is False else go.Figure(
        go.Scatter(x = x,
                   y = y,
                   mode='lines',
                   line=dict(color='rgba(0,0,0,1)'))
    )

    regression_type = 'linear'

    if regression_type == 'linear':

        reg = LinearRegression().fit(np.vstack(x), y)
        best_fit = reg.predict(np.vstack(x))

        # print(reg.intercept_)
        # print(reg.coef_)

        update.update({'intercept': reg.intercept_,
                       'coef': reg.coef_[0]})


    if regression_type == 'weighted average':

        window = 12
        # average_data = []
        # for ind in range(len(y) - window + 1):
        #     average_data.append(np.mean(y[ind:ind + window]))

        i = 0
        # Initialize an empty list to store moving averages
        best_fit=[]

        # Loop through the array to consider
        # every window of size 3
        while i < len(y) - window + 1:
            # Store elements from i to i+window_size
            # in list to get the current window
            _window = y[i: i + window]

            # Calculate the average of current window
            window_average = round(sum(_window) / window, 2)

            # Store the average of current
            # window in moving average list
            best_fit.append(window_average)

            # Shift window to right by one position
            i += 1

    fig.add_trace(go.Scatter(name='line of best fit', x=x, y=best_fit, mode='lines'))


    # print('blam')
    fig.add_trace(go.Scatter(
        x=[min(x), max(x)],
        y=[min(y), max(y)],
        mode='lines',
        line=dict(color='rgba(0,0,0,0.8)', dash='dot'),
        showlegend= False))

    #fig.update_yaxes(type="log")
    #fig.update_xaxes(type="log")

    title= str(x_var) + " in relation to " + str(y_var) if _name is None else _name

    """if speed is True:
        title = 'speed of ' + title"""

    fig.update_layout(
        title= title,
        xaxis_title=str(names[x_var]) + ' ' + str(axis_x),
        yaxis_title=str(names[y_var]) + ' ' + str(axis_y),
        barmode='overlay',
        template="simple_white",
        autosize=False,
        font_family="Times New Roman",
        font=dict(size=10)
    )

    # fig.show()

    fig.update_traces(showlegend=False)

    # fig.add_vline(x=x_mean, line_width=0.3, line_dash="dash")  # , name='Mean x value (%s)' % x_mean)
    fig.add_vline(x=np.median(x), line_width=0.2, line_dash="dot")  # , name='Median x value (%s)' % np.median(x))

    # fig.add_hline(y=y_mean, line_width=0.3, line_dash="dash")  # , name='Mean y value (%s)' % y_mean)
    fig.add_hline(y=np.median(y), line_width=0.2, line_dash="dot")  # , name='Median y value (%s)' % np.median(y))

    _file_name = title + ".html"  # _name is None else _name + title + ".html"
    if _line is True:
        _file_name = 'line ' + _file_name
    _pathfile='Figures/'

    above = 0
    below = 0
    score = 0
    for point in range(len(y)-1):
        line_y = (min(y) - max(y)) / (min(x) - max(x)) * (x[point] - min(x)) + min(y)

        score *= 0.99

        if y[point] > line_y:
            # we are above the 45º degree line
            above += 1
        elif y[point] < line_y:
            below += 1

        score += y[point] - line_y

    points_above = above/(len(y)-1)
    points_below = below/(len(y)-1)

    # print(points_above)
    # print(points_below)
    # print(above_score)
    # print(below_score)

    fig.write_html(_pathfile + _file_name)

    agent_dict[_file_name] = {}
    update.update({
        'points_below': points_below,
        'points_above': points_above,
        'score': score,
    })

    agent_dict[_file_name].update(update)
    # print(agent_dict[_file_name])

    return fig.show() if show is True else print(str(_file_name) + ' is done')


def iqr_table(var, dataframe, x_axis, groupby=None, remove_outliers=True, _sum=False, _loc=None, _old=False):

    if groupby is None:
        groupby = [x_axis]

    # var = 'capacity'
    # name_o_var = 'Number of adaptations'

    # .groupby(['period', 'seed'],as_index=False)['Lumps'].sum().groupby(['period'], as_index=False).quantile(0.5)

    x =     list(dataframe.groupby([x_axis], as_index=False)[x_axis].max()[x_axis])
    x_rev = list(x[ : : -1])

    # _sum = True

    if _sum is False:
        DF = dataframe.groupby(['period', 'seed'],as_index=False)[var].sum() if _sum is True else dataframe.groupby(['period', 'seed'],as_index=False)[var].mean()

        _y_max=  list(dataframe.groupby(groupby, as_index=False)[var].quantile(1)  [var])
        y_upper= list(dataframe.groupby(groupby, as_index=False)[var].quantile(.75)[var])
        y_median=list(dataframe.groupby(groupby, as_index=False)[var].median()     [var])
        y_mean=  list(dataframe.groupby(groupby, as_index=False)[var].mean()       [var])
        y_bottom=list(dataframe.groupby(groupby, as_index=False)[var].quantile(.25)[var])
        _y_min=  list(dataframe.groupby(groupby, as_index=False)[var].quantile(0)  [var])

        """ OLD
        _y_max=  list(DF.groupby(groupby, as_index=False)[var].quantile(1)  [var])
        y_upper= list(DF.groupby(groupby, as_index=False)[var].quantile(.75)[var])
        y_median=list(DF.groupby(groupby, as_index=False)[var].median()     [var])
        y_mean=  list(DF.groupby(groupby, as_index=False)[var].mean()       [var])
        y_bottom=list(DF.groupby(groupby, as_index=False)[var].quantile(.25)[var])
        _y_min=  list(DF.groupby(groupby, as_index=False)[var].quantile(0)  [var])
        """

    else:

        _y_max=  list(dataframe.loc[_loc].groupby(['period', 'seed'],as_index=False).sum().groupby(
            groupby, as_index=False)[var].quantile(1)  [var])
        y_upper= list(dataframe.loc[_loc].groupby(['period', 'seed'],as_index=False).sum().groupby(
            groupby, as_index=False)[var].quantile(.75)[var])
        y_median=list(dataframe.loc[_loc].groupby(['period', 'seed'],as_index=False).sum().groupby(
            groupby, as_index=False)[var].median()     [var])
        y_mean=  list(dataframe.loc[_loc].groupby(['period', 'seed'],as_index=False).sum().groupby(
            groupby, as_index=False)[var].mean()       [var])
        y_bottom=list(dataframe.loc[_loc].groupby(['period', 'seed'],as_index=False).sum().groupby(
            groupby, as_index=False)[var].quantile(.25)[var])
        _y_min=  list(dataframe.loc[_loc].groupby(['period', 'seed'],as_index=False).sum().groupby(
            groupby, as_index=False)[var].quantile(0)  [var])

        """_y_max=  list(dataframe.groupby(['period', 'seed'],as_index=False).sum().loc[_loc].groupby(
            ['period'], as_index=False)[var].quantile(1)  .iloc[:, 1:2].iloc[:, 0])
        y_upper= list(dataframe.groupby(['period', 'seed'],as_index=False).sum().loc[_loc].groupby(
            ['period'], as_index=False)[var].quantile(.75).iloc[:, 1:2].iloc[:, 0])
        y_median=list(dataframe.groupby(['period', 'seed'],as_index=False).sum().loc[_loc].groupby(
            ['period'], as_index=False)[var].median()     .iloc[:, 1:2].iloc[:, 0])
        y_mean=  list(dataframe.groupby(['period', 'seed'],as_index=False).sum().loc[_loc].groupby(
            ['period'], as_index=False)[var].mean()       .iloc[:, 1:2].iloc[:, 0])
        y_bottom=list(dataframe.groupby(['period', 'seed'],as_index=False).sum().loc[_loc].groupby(
            ['period'], as_index=False)[var].quantile(.25).iloc[:, 1:2].iloc[:, 0])
        _y_min=  list(dataframe.groupby(['period', 'seed'],as_index=False).sum().loc[_loc].groupby(
            ['period'], as_index=False)[var].quantile(0)  .iloc[:, 1:2].iloc[:, 0])"""


        """
        _y_max=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(1).groupby(['period'], as_index=False)   .sum()[var])
        y_upper= list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(.75).groupby(['period'], as_index=False) .sum()[var])
        y_median=list(dataframe.groupby(['period', 'seed'],as_index=False)[var].median().groupby(['period'], as_index=False)      .sum()[var])
        y_mean=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].mean().groupby(['period'], as_index=False)        .sum()[var])
        y_bottom=list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(.25).groupby(['period'], as_index=False) .sum()[var])
        _y_min=  list(dataframe.groupby(['period', 'seed'],as_index=False)[var].quantile(0).groupby(['period'], as_index=False)   .sum()[var])
        """

    if remove_outliers is True:
        y_max = []
        y_min = []
        for i in range(0,len(_y_max)):
            y_max.append(min(max(_y_max[i], y_upper[i], y_median[i], y_bottom[i], _y_min[i]), y_upper[i] + 1.5*(y_upper[i]-y_bottom[i])))
            y_min.append(max(min(_y_max[i], y_upper[i], y_median[i], y_bottom[i], _y_min[i]), y_bottom[i] - 1.5*(y_upper[i]-y_bottom[i])))
    else:
        y_max = _y_max
        y_min = _y_min

    y_min = y_min[: : -1]
    y_bottom=y_bottom[: : -1]

    return x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min

def _locs(_df, _str, mwh):

    if mwh is True:
        __locs = {'total': (_df['status'] == 'contracted'),
                  'green': (_df['status'] == 'contracted') & (_df['source'] != 0),
                  'dirty': (_df['status'] == 'contracted') & (_df['source'] == 0)}
    else:
        __locs = {'total': (_df['status'] == 'contracted') | (_df['status'] == 'built'),
                  'green': (_df['source'] != 0),
                  'dirty': (_df['source'] == 0)}


    return __locs[_str]


def line_generator_pd(_last_dict, _growth_dict, _column_name=''):

    row_1 = [_column_name + 'mean', _column_name + 'growth']
    row_2 = [_last_dict['y_mean'], _growth_dict['y_mean']]

    return [row_1, row_2]

def pos(lst):
    return list(filter(lambda num: num != 0, lst))

def line_generator(var, _df, _groupby, x_axis, _loc=None, _remove_outliers=True, _sum=False, _avoided=False):
    """

    :param _df:
    :param _groupby:
    :param _cond: condition can be the code itself! ex.: (agents_df['genre'] == 'EP') | (agents_df['genre'] == 'TP'), just be aware that then the df must also be in the condition
    :param _sum:
    :return:
    """

    x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min = iqr_table(var, _df, x_axis, _groupby, _remove_outliers, _loc=_loc, _sum=_sum)

    # last period
    # last = ['last_y_upper', 'last_y_median', 'last_y_mean', 'last_y_bottom', 'last_y_max', 'last_y_min']
    last = ['last_y_median', 'last_y_mean']
    last_var = [y_median[-1], y_mean[-1]]

    # growth

    ### the pos function is only to get the first

    growth = ['growth_y_upper', 'growth_y_median', 'growth_y_mean', 'growth_y_bottom', 'growth_y_max', 'growth_y_min']
    growth_var = [# (y_upper[-1]  - pos(y_upper)[0] )/pos(y_upper)[0]  ,
                  (y_median[-1] - pos(y_median)[0])/pos(y_median)[0] ,
                  (y_mean[-1]   - pos(y_mean)[0]  )/pos(y_mean)[0]   ,]
                  # (y_bottom[-1] - pos(y_bottom)[0])/pos(y_bottom)[0] ,
                  # (y_max[-1]    - pos(y_max)[0]   )/pos(y_max)[0]    ,
                  # (y_min[-1]    - pos(y_min)[0]   )/pos(y_min)[0]    ]

    last_dict = {}
    growth_dict = {}

    # names = ['y_upper', 'y_median', 'y_mean', 'y_bottom', 'y_max', 'y_min']
    names = ['y_median', 'y_mean']
    if _avoided is True:
        last_var[0] = sum(y_median)
        last_var[1] = sum(y_mean)

    n=0
    for i in names:
        last_dict[i] = last_var[n]
        growth_dict[i] = growth_var[n]
        n += 1

    return last_dict, growth_dict


if __name__ == '__main__':

    """
    Order:
    
    1 - analysis csv
    2 - figures
    
    """

    list_of_files = os.listdir('.')
    list_of_files.remove('Figures')

    dfs_names = {'__mix': {},
                 '__agents': {},
                 '__contracts': {},
                 '__technologic': {}}

    for _file in list_of_files:
        _entry1 = None
        if os.stat(_file).st_size > 100:
            if '.csv' in _file:
                if 'mix' in _file:
                    _entry1 = 'mix'

                elif 'agents' in _file:
                    _entry1 = 'agents'

                elif 'contracts' in _file:
                    _entry1 = 'contracts'

                elif 'technologic' in _file:
                    _entry1 = 'technologic'

        if _entry1 is not None:
            dfs_names['__' + _entry1][_file.split('__')[0]] = _file

    # print(dfs_names)

    dfs = {}
    for _name1 in dfs_names:
        for _name2 in dfs_names[_name1]:
            print('LOADING ' + _name2 + _name1)
            try:
                dfs[_name2 + _name1] = load_into_df(dfs_names[_name1][_name2])
                print('LOADED ' + _name2 + _name1)
            except:
                None
    print('DONE WITH EM ALL')

    # print(dfs)

    """dfs_names ={
        '__mix': {'NO_NO': 'ALL_NO_NO_normal_mix.csv',
                  'NO_YES': 'ALL_NO_YES_normal_mix.csv',
                  'YES_NO': 'ALL_YES_NO_normal_mix.csv',
                  'ALL_YES_YES': 'ALL_YES_YES_normal_mix.csv',
                  'NONE_YES_YES': 'NONE_YES_YES_normal_agents.csv'},
        '__agents': {'NO_NO': 'ALL_NO_NO_normal_agents.csv',
                     'NO_YES': 'ALL_NO_YES_normal_agents.csv',
                     'YES_NO': 'ALL_YES_NO_normal_agents.csv',
                     'ALL_YES_YES': 'ALL_YES_YES_normal_agents.csv',
                     'NONE_YES_YES': 'NONE_YES_YES_normal_agents.csv'}
    }"""

    """dfs = {}
    for _ in dfs_names:

        dfs[_] = load_into_df(dfs_names[_])"""
    report = True

    if report is True:
        _sum_vars = ['MWh', 'capacity', 'avoided_emissions']
        _normal_vars = []  #  ['price']
        rows_for_df = {}
        _rows_for_df = []

        _dfs = []
        for i in list(dfs.keys()):
            if 'mix' in i:
                _dfs.append(i)

        for var in _sum_vars + _normal_vars:
            for _ in _dfs:
                for _type in ['total', 'green', 'dirty']:
                    df = dfs[_]
                    _loc = _locs(df, _type, mwh= True if var == 'MWh' else False)
                    _sum = True if var in _sum_vars else False
                    if var in ['avoided_emissions']:
                        dicts = line_generator(var, df, 'period', 'period', _remove_outliers=True, _loc=_loc, _sum=_sum, _avoided=True)
                    else:
                        dicts = line_generator(var, df, 'period', 'period', _remove_outliers=True, _loc=_loc, _sum=_sum)
                    line = line_generator_pd(dicts[0], dicts[1], var)
                    rows_for_df[_ + _type + var] = line
                    _rows_for_df.append([_.split('__')[0] + ' ' + _type + ' ' + var, line[1][0], line[1][1]])

                    if var in ['price', 'avoided_emissions']:
                        break

        print(rows_for_df)
        main_df = pd.DataFrame(_rows_for_df, columns=['Type', 'MWhmean', 'MWhgrowth'])
        main_df.to_csv('report' + '.csv', index=False)

    """
    mix graphs now
    """

    mix_df = {}

    pre_titles = []
    for i in list(dfs.keys()):
        if 'mix' in i:
            pre_titles.append(i)

    # pre_titles = []  # uncomment to select just certain dataframes
    for pre_title in pre_titles:
        number = 1895

        mix_df = dfs[pre_title]
        pathfile = 'Figures/'

        ### Doing a graph for the generation of electricity

        fig = go.Figure()

        thermal_iqr, thermal_max_min, thermal_median = simple_add_trace('Electricity generated by thermal', 'MWh',
                                                                        mix_df.loc[
                                                                            (mix_df['status'] == 'contracted') & (
                                                                                        mix_df['source'] == 0)],
                                                                        'period', groupby=None, remove_outliers=True,
                                                                        color='232,126,4', _sum=True, _dash='solid')

        fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))

        fig.add_trace(go.Scatter(name=thermal_median['name'],
                                 x=thermal_median['x'],
                                 y=thermal_median['y'],
                                 mode=thermal_median['mode'],
                                 line=thermal_median['line']))

        solar_iqr, solar_max_min, solar_median = simple_add_trace('Electricity generated by solar', 'MWh',
                                                                  mix_df.loc[(mix_df['status'] == 'contracted') & (
                                                                              mix_df['source'] == 2)],
                                                                  'period', groupby=None, remove_outliers=True,
                                                                  color='126,232,4',
                                                                  _sum=True, _dash='dashdot')

        fig.add_trace(go.Scatter(name=solar_iqr['name'],
                                 x=solar_iqr['x'],
                                 y=solar_iqr['y'],
                                 fill=solar_iqr['fill'],
                                 fillcolor=solar_iqr['fillcolor'],
                                 line=solar_iqr['line']))

        fig.add_trace(go.Scatter(name=solar_median['name'],
                                 x=solar_median['x'],
                                 y=solar_median['y'],
                                 mode=solar_median['mode'],
                                 line=solar_median['line']))

        wind_iqr, wind_max_min, wind_median = simple_add_trace('Electricity generated by wind', 'MWh',
                                                               mix_df.loc[(mix_df['status'] == 'contracted') & (
                                                                           mix_df['source'] == 1)],
                                                               'period', groupby=None, remove_outliers=True,
                                                               color='4,126,232', _sum=True, base_100=False, _pattern='x')

        fig.add_trace(go.Scatter(name=wind_iqr['name'],
                                 x=wind_iqr['x'],
                                 y=wind_iqr['y'],
                                 fill=wind_iqr['fill'],
                                 fillcolor=wind_iqr['fillcolor'],
                                 line=wind_iqr['line']))

        fig.add_trace(go.Scatter(name=wind_median['name'],
                                 x=wind_median['x'],
                                 y=wind_median['y'],
                                 mode=wind_median['mode'],
                                 line=wind_median['line']))

        fig.update_yaxes(type="log")

        fig.update_layout(
            title=names[pre_title.split('__')[0]] + 'Electricity generation ',
            xaxis_title='period (months)',
            yaxis_title='MWh',
            barmode='overlay',
            template="simple_white")

        file_name = str(number) + ' - ' + names[pre_title.split('__')[0]] + 'Electricity generation' + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        print(file_name.split('.html')[0], 'is done')

        number += 1

        fig = go.Figure()

        thermal_iqr, thermal_max_min, thermal_median = simple_add_trace('Electricity generated by thermal', 'MWh',
                                                                        mix_df.loc[
                                                                            (mix_df['status'] == 'contracted') & (
                                                                                    mix_df['source'] == 0)],
                                                                        'period', groupby=None, remove_outliers=True,
                                                                        color='232,126,4', _sum=True, _dash='solid')

        fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))

        fig.add_trace(go.Scatter(name=thermal_median['name'],
                                 x=thermal_median['x'],
                                 y=thermal_median['y'],
                                 mode=thermal_median['mode'],
                                 line=thermal_median['line']))

        renewable_iqr, renewable_max_min, renewable_median = simple_add_trace('Electricity generated by renewable', 'MWh',
                                                                  mix_df.loc[(mix_df['status'] == 'contracted') & (
                                                                          mix_df['source'] == 2)],
                                                                  'period', groupby=None, remove_outliers=True,
                                                                  color='126,232,4',
                                                                  _sum=True, _dash='dashdot')

        fig.add_trace(go.Scatter(name=renewable_iqr['name'],
                                 x=renewable_iqr['x'],
                                 y=renewable_iqr['y'],
                                 fill=renewable_iqr['fill'],
                                 fillcolor=renewable_iqr['fillcolor'],
                                 line=renewable_iqr['line']))

        fig.add_trace(go.Scatter(name=renewable_median['name'],
                                 x=renewable_median['x'],
                                 y=renewable_median['y'],
                                 mode=renewable_median['mode'],
                                 line=renewable_median['line']))

        fig.update_yaxes(type="log")

        fig.update_layout(
            title=names[pre_title.split('__')[0]] + 'Electricity generation ',
            xaxis_title='period (months)',
            yaxis_title='MWh',
            barmode='overlay',
            template="simple_white")

        file_name = str(number) + ' - ' + names[pre_title.split('__')[0]] + 'Electricity generation' + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        print(file_name.split('.html')[0], 'is done')

        """number += 1

        fig = go.Figure()

        fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))

        fig.add_trace(go.Scatter(name=solar_iqr['name'],
                                 x=solar_iqr['x'],
                                 y=solar_iqr['y'],
                                 fill=solar_iqr['fill'],
                                 fillcolor=solar_iqr['fillcolor'],
                                 line=solar_iqr['line']))

        fig.add_trace(go.Scatter(name=wind_iqr['name'],
                                 x=wind_iqr['x'],
                                 y=wind_iqr['y'],
                                 fill=wind_iqr['fill'],
                                 fillcolor=wind_iqr['fillcolor'],
                                 line=wind_iqr['line']))

        fig.update_yaxes(type="log")

        fig.update_layout(
            title=names[pre_title.split('__')[0]] + 'Electricity generation (IQR)',
            xaxis_title='period (months)',
            yaxis_title='MWh',
            barmode='overlay',
            template="simple_white")

        file_name = str(number) + ' - ' + names[pre_title.split('__')[0]] + 'Electricity generation (IQR)' + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        print(file_name.split('.html')[0], 'is done')"""

        number += 1

        # fig.show()

        ### figure of capacity

        fig = go.Figure()

        thermal_iqr, thermal_max_min, thermal_median = simple_add_trace('Thermal capacity', 'capacity',
                                                                        mix_df.loc[
                                                                            (
                                                                                    mix_df['source'] == 0)],
                                                                        'period', groupby=None, remove_outliers=True,
                                                                        color='232,126,4', _sum=True, _dash='solid')

        fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))

        fig.add_trace(go.Scatter(name=thermal_median['name'],
                                 x=thermal_median['x'],
                                 y=thermal_median['y'],
                                 mode=thermal_median['mode'],
                                 line=thermal_median['line']))

        solar_iqr, solar_max_min, solar_median = simple_add_trace('Solar capacity', 'capacity',
                                                                  mix_df.loc[(
                                                                          mix_df['source'] == 2)],
                                                                  'period', groupby=None, remove_outliers=True,
                                                                  color='126,232,4',
                                                                  _sum=True, _dash='dashdot', _pattern='+')

        fig.add_trace(go.Scatter(name=solar_iqr['name'],
                                 x=solar_iqr['x'],
                                 y=solar_iqr['y'],
                                 fill=solar_iqr['fill'],
                                 fillcolor=solar_iqr['fillcolor'],
                                 line=solar_iqr['line']))

        fig.add_trace(go.Scatter(name=solar_median['name'],
                                 x=solar_median['x'],
                                 y=solar_median['y'],
                                 mode=solar_median['mode'],
                                 line=solar_median['line']))

        wind_iqr, wind_max_min, wind_median = simple_add_trace('Wind capacity', 'capacity',
                                                               mix_df.loc[(
                                                                       mix_df['source'] == 1)],
                                                               'period', groupby=None, remove_outliers=True,
                                                               color='4,126,232', _sum=True)

        fig.add_trace(go.Scatter(name=wind_iqr['name'],
                                 x=wind_iqr['x'],
                                 y=wind_iqr['y'],
                                 fill=wind_iqr['fill'],
                                 fillcolor=wind_iqr['fillcolor'],
                                 line=wind_iqr['line']))

        fig.add_trace(go.Scatter(name=wind_median['name'],
                                 x=wind_median['x'],
                                 y=wind_median['y'],
                                 mode=wind_median['mode'],
                                 line=wind_median['line']))

        fig.update_yaxes(type="log")

        fig.update_layout(
            title=names[pre_title.split('__')[0]] + 'Electric capacity',
            xaxis_title='period (months)',
            yaxis_title='MW',
            barmode='overlay',
            template="simple_white")

        file_name = str(number) + ' - ' + names[pre_title.split('__')[0]] + 'Electric capacity' + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        print(file_name.split('.html')[0], 'is done')

        number += 1

        fig = go.Figure()

        thermal_iqr, thermal_max_min, thermal_median = simple_add_trace('Thermal capacity', 'capacity',
                                                                        mix_df.loc[
                                                                            (
                                                                                    mix_df['source'] == 0)],
                                                                        'period', groupby=None, remove_outliers=True,
                                                                        color='232,126,4', _sum=True, _dash='solid')

        fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))

        fig.add_trace(go.Scatter(name=thermal_median['name'],
                                 x=thermal_median['x'],
                                 y=thermal_median['y'],
                                 mode=thermal_median['mode'],
                                 line=thermal_median['line']))

        Renewable_iqr, Renewable_max_min, Renewable_median = simple_add_trace('Renewable capacity', 'capacity',
                                                                  mix_df.loc[(
                                                                          mix_df['source'] != 0)],
                                                                  'period', groupby=None, remove_outliers=True,
                                                                  color='126,232,4',
                                                                  _sum=True, _dash='dashdot', _pattern='+')

        fig.add_trace(go.Scatter(name=Renewable_iqr['name'],
                                 x=Renewable_iqr['x'],
                                 y=Renewable_iqr['y'],
                                 fill=Renewable_iqr['fill'],
                                 fillcolor=Renewable_iqr['fillcolor'],
                                 line=Renewable_iqr['line']))

        fig.add_trace(go.Scatter(name=Renewable_median['name'],
                                 x=Renewable_median['x'],
                                 y=Renewable_median['y'],
                                 mode=Renewable_median['mode'],
                                 line=Renewable_median['line']))

        fig.update_yaxes(type="log")

        fig.update_layout(
            title=names[pre_title.split('__')[0]] + 'Electric capacity',
            xaxis_title='period (months)',
            yaxis_title='MW',
            barmode='overlay',
            template="simple_white")

        file_name = str(number) + ' - ' + names[pre_title.split('__')[0]] + 'Electric capacity' + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        print(file_name.split('.html')[0], 'is done')

        # fig.show()

        ### figure of capacity

        """fig = go.Figure()

        fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))

        fig.add_trace(go.Scatter(name=solar_iqr['name'],
                                 x=solar_iqr['x'],
                                 y=solar_iqr['y'],
                                 fill=solar_iqr['fill'],
                                 fillcolor=solar_iqr['fillcolor'],
                                 line=solar_iqr['line']))

        fig.add_trace(go.Scatter(name=wind_iqr['name'],
                                 x=wind_iqr['x'],
                                 y=wind_iqr['y'],
                                 fill=wind_iqr['fill'],
                                 fillcolor=wind_iqr['fillcolor'],
                                 line=wind_iqr['line']))

        fig.update_yaxes(type="log")

        fig.update_layout(
            title=names[pre_title.split('__')[0]] + 'Electric capacity (IQR)',
            xaxis_title='period (months)',
            yaxis_title='MW',
            barmode='overlay',
            template="simple_white")

        file_name = str(number) + ' - ' + names[pre_title.split('__')[0]] + 'Electric capacity (IQR)' + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        print(file_name.split('.html')[0], 'is done')"""

        number += 1

        graphs_list = [
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Avoided emissions',
                "var": 'avoided_emissions',
                "dataframe": mix_df.loc[mix_df['status'] == 'contracted'],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Emissions',
                "var": 'emissions',
                "dataframe": mix_df.loc[mix_df['status'] == 'contracted'],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Total capacity',
                "var": 'capacity',
                "dataframe": mix_df,
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Wind capacity',
                "var": 'capacity',
                "dataframe": mix_df.loc[mix_df['source'] == 1],
                "x_axis": 'period',
                'sum': True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Solar capacity',
                "var": 'capacity',
                "dataframe": mix_df.loc[mix_df['source'] == 2],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Thermal capacity',
                "var": 'capacity',
                "dataframe": mix_df.loc[mix_df['source'] == 0],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Price',
                "var": 'price',
                "dataframe": mix_df.loc[mix_df['status'] == 'contracted'],
                "x_axis": 'period',
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Price of solar',
                "var": 'price',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 2)],
                "x_axis": 'period',
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Price of wind',
                "var": 'price',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 1)],
                "x_axis": 'period',
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Price of thermal',
                "var": 'price',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 0)],
                "x_axis": 'period',
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Electricity produced',
                "var": 'MWh',
                "dataframe": mix_df.loc[mix_df['status'] == 'contracted'],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Electricity produced by solar',
                "var": 'MWh',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 1)],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Electricity produced by wind',
                "var": 'MWh',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 2)],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + 'Electricity produced by thermal',
                "var": 'MWh',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 0)],
                "x_axis": 'period',
                "sum": True
            },
        ]

        for graph in graphs_list:
            if 'log_y' in graph:
                log_y = graph['log_y']
            else:
                log_y = False
            if 'log_x' in graph:
                log_x = graph['log_x']
            else:
                log_x = False

            if 'sum' in graph:
                _sum = True
            else:
                _sum = False
            _name = str(number) + ' - ' + graph['name_o_var']
            simple_graph(_name, graph['var'], graph['dataframe'], graph['x_axis'], groupby=None,
                         remove_outliers=True, show=False, log_y=log_y, log_x=log_x, _sum=_sum)
            number += 1
            print(_name, 'is done')

    pre_titles = []
    """
    Agents graphs first
    """

    agent_dict = {}

    for i in list(dfs.keys()):
        if 'agents' in i:
            pre_titles.append(i)
    print(pre_titles)

    # pre_titles = ['YES_YES__agents']  # uncomment and put a certain dataframe name to avoid checking all

    public_var = 'current_state_norm'
    max_bndes = False
    max_epm = False
    max_tpm = False
    min_bndes = False
    min_epm = False
    min_tpm = False
    # max_profit = False
    # min_profit = False
    for pre_title in pre_titles:

        # _df = dfs[pre_title].loc[(dfs[pre_title]['genre'] == 'EP') | (dfs[pre_title]['genre'] == 'TP')]

        # max_profit = _df['profits'].max() if max_profit is False else max(
        #             _df['profits'].max(), max_profit)
        # min_profit = _df['profits'].max() if min_profit is False else max(
        #             _df['profits'].max(), min_profit)

        if 'YES' in pre_title:
            try:
                _df = dfs[pre_title].loc[dfs[pre_title]['genre'] == 'DBB']
                max_bndes = _df['current_state'].max() if max_bndes is False else max(
                    _df['current_state'].max(), max_bndes)
                min_bndes = _df['current_state'].min() if min_bndes is False else min(
                    _df['current_state'].min(), min_bndes)
            except:
                None

            try:
                _df = dfs[pre_title].loc[dfs[pre_title]['genre'] == 'EPM']
                max_epm = _df['current_state'].max() if max_epm is False else max(
                    _df['current_state'].max(), max_epm)
                min_epm = _df['current_state'].min() if min_epm is False else min(
                    _df['current_state'].min(), min_epm)
            except:
                None

            try:
                _df = dfs[pre_title].loc[dfs[pre_title]['genre'] == 'TPM']
                max_tpm = _df['current_state'].max() if max_epm is False else max(
                    _df['current_state'].max(), max_epm)
                min_tpm = _df['current_state'].min() if min_epm is False else min(
                    _df['current_state'].min(), min_epm)
            except:
                None

    print(max_bndes, max_epm, min_bndes, min_epm)

    for pre_title in pre_titles:
        agents_df = dfs[pre_title]
        list_o_var = ['LSS_tot', 'LSS_weak']
        private_var = 'profits'
        public_var = 'current_state'
        if 'YES' in pre_title:
            public_var = 'current_state_norm'
            print('normalizing', pre_title.split('__')[0])

            for row in range(0, len(agents_df)):
                if agents_df.loc[row, :]['genre'] == 'DBB':
                    agents_df.loc[row, 'current_state_norm'] = (agents_df.loc[row, 'current_state'] - min_bndes
                                                                ) / (max_bndes - min_bndes)
                elif agents_df.loc[row, :]['genre'] == 'EPM':
                    agents_df.loc[row, 'current_state_norm'] = (agents_df.loc[row, 'current_state'] - min_epm
                                                                ) / (max_epm - min_epm)

                elif agents_df.loc[row, :]['genre'] == 'TPM':
                    agents_df.loc[row, 'current_state_norm'] = (agents_df.loc[row, 'current_state'] - min_tpm
                                                                ) / (max_tpm - min_tpm)

        private_cond = (agents_df['genre'] == 'EP') | (agents_df['genre'] == 'TP')
        public_cond = (agents_df['genre'] != 'EP') & (agents_df['genre'] != 'TP') & (agents_df['genre'] != 'DD')

        private = {'df': agents_df.loc[private_cond], 'var': private_var}
        public = {'df': agents_df.loc[public_cond], 'var': public_var}

        if 'NO_NO' not in pre_title:
            number = 0
            list_o_graphs = [['LSS_tot', public, 'LSS_tot', private], ['LSS_weak', public, 'LSS_weak', private],
                             [public_var, public, 'LSS_tot', private], [public_var, public, 'LSS_weak', private],
                             ['LSS_tot', public, private_var, private], ['LSS_weak', public, private_var, private],
                             [public_var, public, private_var, private],
                             [public_var, public, 'LSS_tot', public],
                             [public_var, public, 'LSS_weak', public],
                             [private_var, private, 'LSS_tot', private],
                             [private_var, private, 'LSS_weak', private]]  # x is public, y is private
        else:
            number = 45  # (number of speeds + actual values + standardized) *
            list_o_graphs = [[private_var, private, 'LSS_tot', private],
                             [private_var, private, 'LSS_weak', private]]

        for _graph in list_o_graphs:
            x_axis_name = names['public agents'] if _graph[1]['var'] == public['var'] else names['private agents']
            y_axis_name = names['public agents'] if _graph[3]['var'] == public['var'] else names['private agents']
            # name = _graph[0] + ' in relation to ' + _graph[1]
            for speed in [True, False]:
                # name = name + ' (every ' + str(time) + ' months) '
                if speed is True:
                    for time in [12, 24, 48]:
                        # name = 'speed of ' + name
                        try:
                            scatter_graph(
                                [_graph[0], _graph[1]['df'], False, x_axis_name],
                                [_graph[2], _graph[3]['df'], False, y_axis_name],
                                speed=speed,
                                show=False,
                                time=time,
                                _name=str(number) + ' - ' + names[pre_title.split('__')[0]] + 'speeds of ' + names[_graph[
                                    0]] + ' of ' + x_axis_name + ' in relation to ' + names[_graph[
                                          2]] + ' of ' + y_axis_name + ' (every ' + str(time) + ' months) ',
                            )
                        except:
                            print('We failed with the graph numbah ' + str(number) + ' of the ' + names[
                                pre_title.split('__')[0]])
                        number += 1
                else:
                    for norm in [False, True]:
                        if norm is True:
                            for normalization in [False]:  # , True]:  # They are basically the same
                                norm_name = ' (normalized)' if normalization is True else ' (standardized)'
                                scatter_graph(
                                    [_graph[0], _graph[1]['df'], norm, x_axis_name],
                                    [_graph[2], _graph[3]['df'], norm, y_axis_name],
                                    speed=speed,
                                    show=False,
                                    _name=str(number) + ' - ' + names[pre_title.split('__')[0]] + names[_graph[
                                        0]] + ' of ' + x_axis_name + ' in relation to ' + names[_graph[
                                              2]] + ' of ' + y_axis_name + norm_name,
                                    normalization=normalization)
                                number += 1
                        else:
                            # name = name + ' (actual values)'
                            scatter_graph(
                                [_graph[0], _graph[1]['df'], norm, x_axis_name],
                                [_graph[2], _graph[3]['df'], norm, y_axis_name],
                                speed=speed,
                                show=False,
                                _name=str(number) + ' - ' + names[pre_title.split('__')[0]] + names[_graph[
                                    0]] + ' of ' + x_axis_name + ' in relation to ' + names[_graph[
                                          2]] + ' of ' + y_axis_name + ' (actual values)')
                            number += 1

        graphs_list = [
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Number of adaptations',
                "var": 'LSS_tot',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Number of adaptations of Technology producers',
                "var": 'LSS_tot',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Number of adaptations of Energy providers',
                "var": 'LSS_tot',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Full number of adaptations',
                "var": 'LSS_weak',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[
                    0]] + ' ' + 'Full number of adaptations of Technology producers',
                "var": 'LSS_weak',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Full number of adaptations of Energy providers',
                "var": 'LSS_weak',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Money',
                "var": 'wallet',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Money of Technology producers',
                "var": 'wallet',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Money of Energy providers',
                "var": 'wallet',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Ammount to shareholders',
                "var": 'shareholder_money',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Ammount to shareholders of Technology producers',
                "var": 'shareholder_money',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Ammount to shareholders of Energy providers',
                "var": 'shareholder_money',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Investment in capacity of Technology producers',
                "var": 'capacity',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period',
            },
            {
                "name_o_var": names[pre_title.split('__')[
                    0]] + ' ' + 'Investment in R&D to shareholders of Technology producers',
                "var": 'RandD',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period',
            },
            {
                "name_o_var": names[pre_title.split('__')[0]] + ' ' + 'Remaining demand',
                "var": 'Remaining_demand',
                "dataframe": agents_df.loc[agents_df['genre'] == 'DD'],
                "x_axis": 'period'
            },
        ]

        for graph in graphs_list:
            if 'log_y' in graph:
                log_y = graph['log_y']
            else:
                log_y = False
            if 'log_x' in graph:
                log_x = graph['log_x']
            else:
                log_x = False
            _name = str(number) + ' - ' + graph['name_o_var']
            simple_graph(_name, graph['var'], graph['dataframe'], graph['x_axis'], groupby=None,
                         remove_outliers=True, show=False, log_y=log_y, log_x=log_x)
            number += 1
            print(_name, 'is done')

    print(agent_dict)
    agent_dict_df = pd.DataFrame(agent_dict)
    total = agent_dict_df
    total = total.T
    total = total.reset_index()
    total.columns = ['Type',
                     'intercept',
                     'coeficient',
                     'points below',
                     'points above',
                     'score']
    total = total.sort_values(by=['Type'])
    total['Number'] = [int(str(i).split(' - ')[0]) for i in total['Type']]
    total['Type'] = [str(i).split(' - ')[1] + ' -- ' + str(i).split(' - ')[2] for i in total['Type']]
    total['Type'] = [str(i).split('.html')[0] for i in total['Type']]
    total = total.sort_values(by=['Number'])
    total = total.astype({'intercept': 'float', 'coeficient': 'float', 'points above': 'float', 'points below': 'float',
                          'score': 'float'})
    total.to_csv('agent_dict' + '.csv', index=False)

    duration = 1500  # milliseconds
    freq = 880  # Hz
    winsound.Beep(freq, duration)
