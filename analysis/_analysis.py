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


def iqr_plotly(var, dataframe, x_axis, groupby=None, remove_outliers=True, _sum=False, _old=False):

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
            y_max.append(min(max(_y_max[i], y_upper[i], y_median[i], y_bottom[i], _y_min[i]), y_upper[i] + 1.5*(y_upper[i]-y_bottom[i])))
            y_min.append(max(min(_y_max[i], y_upper[i], y_median[i], y_bottom[i], _y_min[i]), y_bottom[i] - 1.5*(y_upper[i]-y_bottom[i])))
    else:
        y_max = _y_max
        y_min = _y_min

    y_min = y_min[: : -1]
    y_bottom=y_bottom[: : -1]

    return x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min

def simple_graph(name_o_var, var, dataframe, x_axis, groupby=None, remove_outliers=True, show=True, log_y=False, log_x=False, color='232,126,4', _sum=False):

    x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min = iqr_plotly(var, dataframe, x_axis, groupby, remove_outliers, _sum=_sum)

    fig = go.Figure(go.Scatter(name="IQR (%s)" % name_o_var,
                             x=x + x_rev,
                             y=y_upper+y_bottom,
                             fill='toself',
                             fillcolor='rgba(%s,0.4)' % color,
                             line=dict(color='rgba(255,255,255,0)')))

    fig.add_trace(go.Scatter(name="Max and min (%s)" % name_o_var,
                             x=x + x_rev,
                             y=y_max+y_min,
                             fill='toself',
                             fillcolor='rgba(%s,0.2)' % color,
                             line=dict(color='rgba(255,255,255,0)')))

    fig.add_trace(go.Scatter(name="Median (%s)" % name_o_var,
                             x=x + x_rev,
                             y=y_median,
                            mode='lines',
                             line=dict(color='rgba(%s,1)' % color, dash='dot')))

    fig.update_yaxes(type="log") if log_y is True else None
    fig.update_xaxes(type="log") if log_x is True else None
    fig.update_layout(barmode='overlay', template="simple_white")

    title=str(name_o_var) + ' over ' + str(x_axis)

    fig.update_layout(
        title=title,
        xaxis_title=str(x_axis),
        yaxis_title=str(name_o_var),
        barmode='overlay',
        template="simple_white")

    # fig.show()

    file_name = title + ".html"
    pathfile='Figures/'

    fig.write_html(pathfile + file_name)

    return fig.show() if show is True else None

def simple_add_trace(name_o_var, var, dataframe, x_axis, groupby=None, remove_outliers=True, color=None, _sum=False):

    if color == None:
        color='232,126,4'

    x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min = iqr_plotly(var, dataframe, x_axis, groupby, remove_outliers, _sum=_sum)

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
                  line=dict(color='rgba(%s,1)' % color, dash='dot'))

    return IQR, max_min, median

def scatter_graph(full_x, full_y, speed=False, groupby=None, show=True, time=1, normalization=True, # If normalization is false we standardize
                  _name=None, _line=False):

    if groupby is None:
        groupby = 'period'

    # dataframe = agents_df  # .loc[agents_df['genre'] == 'TP']

    x_var, DF_x, _normal_x, axis_x = full_x[0], full_x[1], full_x[2],full_x[3]
    y_var, DF_y, _normal_y, axis_y = full_y[0], full_y[1], full_y[2],full_y[3]


    # DF_x = DF_x.groupby(['period', 'seed'],as_index=False)[x_var].sum() if _sum_x is True else DF_x.groupby(['period', 'seed'],as_index=False)[x_var].mean()
    # DF_y = DF_y.groupby(['period', 'seed'],as_index=False)[y_var].sum() if _sum_y is True else DF_y.groupby(['period', 'seed'],as_index=False)[y_var].mean()

    x =list(DF_x.groupby(groupby, as_index=False)[x_var].mean()[x_var])
    y =list(DF_y.groupby(groupby, as_index=False)[y_var].mean()[y_var])
    # print(x,y)

    if speed is True:
        """priv_goal[period] - priv_goal[period - 1]
                                 ) / priv_goal[period - 1] if priv_goal[period - 1] > 0 else 1"""
        x= [(x[i] - x[i-time])/x[i-time] if x[i-time] != 0 else 'to_remove' for i in range(time,len(x))]
        #x[0] = 0
        y= [(y[i] - y[i-time])/y[i-time] if y[i-time] != 0 else 'to_remove' for i in range(time,len(y))]
        #y[0] = 0

        to_pop=[]
        _max = len(x)-1
        for _ in range(0, len(x)-1):
            if x[_] == 'to_remove' or y[_] == 'to_remove':
                to_pop.append(_)

        to_pop.reverse()
        if len(to_pop) == len(x):
            x = y = [0]  # for i in range(len(x))]

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
                x = [(x[i]- min(x))/(max(x) - min(x)) for i in range(len(x))] if _normal_x is True else x
                y = [(y[i]- min(y))/(max(y) - min(y)) for i in range(len(y))] if _normal_y is True else y



    color = list(DF_x.groupby(['period'], as_index=False).max()['period'])
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

    reg = LinearRegression().fit(np.vstack(x), y)
    best_fit = reg.predict(np.vstack(x))

    fig.add_trace(go.Scatter(name='line of best fit', x=x, y=best_fit, mode='lines'))

    if normalization is False and (_normal_x is True or _normal_y is True):
        # print('blam')
        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[0,1],
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
        xaxis_title=str(x_var) + ' ' + str(axis_x),
        yaxis_title=str(y_var) + ' ' + str(axis_y),
        barmode='overlay',
        template="simple_white",
        autosize=False,
        font_family="Times New Roman",
        font=dict(size=10)
    )

    # fig.show()

    x_mean = (min(x) + max(x))/2
    y_mean = (min(y) + max(y))/2

    fig.update_traces(showlegend=False)

    fig.add_vline(x=x_mean, line_width=0.3, line_dash="dash")  # , name='Mean x value (%s)' % x_mean)
    fig.add_vline(x=np.median(x), line_width=0.3, line_dash="dot")  # , name='Median x value (%s)' % np.median(x))

    fig.add_hline(y=y_mean, line_width=0.3, line_dash="dash")  # , name='Mean y value (%s)' % y_mean)
    fig.add_hline(y=np.median(y), line_width=0.3, line_dash="dot")  # , name='Median y value (%s)' % np.median(y))

    _file_name = title + ".html"  # _name is None else _name + title + ".html"
    if _line is True:
        _file_name = 'line ' + _file_name
    _pathfile='Figures/'

    fig.write_html(_pathfile + _file_name)

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

def _locs(_df, _str):

    __locs = {'total': (_df['status'] == 'contracted'),
              'green': (_df['status'] == 'contracted') & (_df['source'] != 0),
              'dirty': (_df['status'] == 'contracted') & (_df['source'] == 0)}

    return __locs[_str]


def line_generator_pd(_last_dict, _growth_dict, _column_name=''):

    row_1 = [_column_name + 'mean', _column_name + 'growth']
    row_2 = [_last_dict['y_mean'], _growth_dict['y_mean']]

    return [row_1, row_2]

def pos(lst):
    return list(filter(lambda num: num != 0, lst))

def line_generator(var, _df, _groupby, x_axis, _loc=None, _remove_outliers=True, _sum=False):
    """

    :param _df:
    :param _groupby:
    :param _cond: condition can be the code itself! ex.: (agents_df['genre'] == 'EP') | (agents_df['genre'] == 'TP'), just be aware that then the df must also be in the condition
    :param _sum:
    :return:
    """

    x, x_rev, y_upper, y_median, y_mean, y_bottom, y_max, y_min = iqr_table(var, _df, x_axis, _groupby, _remove_outliers, _loc=_loc, _sum=_sum)

    # last period
    last = ['last_y_upper', 'last_y_median', 'last_y_mean', 'last_y_bottom', 'last_y_max', 'last_y_min']
    last_var = [y_upper[-1], y_median[-1], y_mean[-1], y_bottom[-1], y_max[-1], y_min[-1]]

    # growth

    ### the pos function is only to get the first

    growth = ['growth_y_upper', 'growth_y_median', 'growth_y_mean', 'growth_y_bottom', 'growth_y_max', 'growth_y_min']
    growth_var = [(y_upper[-1]  - pos(y_upper)[0] )/pos(y_upper)[0]  ,
                  (y_median[-1] - pos(y_median)[0])/pos(y_median)[0] ,
                  (y_mean[-1]   - pos(y_mean)[0]  )/pos(y_mean)[0]   ,
                  (y_bottom[-1] - pos(y_bottom)[0])/pos(y_bottom)[0] ,
                  (y_max[-1]    - pos(y_max)[0]   )/pos(y_max)[0]    ,
                  (y_min[-1]    - pos(y_min)[0]   )/pos(y_min)[0]    ]

    last_dict = {}
    growth_dict = {}

    names = ['y_upper', 'y_median', 'y_mean', 'y_bottom', 'y_max', 'y_min']

    n=0
    for i in names:
        last_dict[i] = last_var[n]
        growth_dict[i] = growth_var[n]
        n+=1

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
            try:
                dfs[_name2 + _name1] = load_into_df(dfs_names[_name1][_name2])
                print('LOADED ' + _name2 + _name1)
            except:
                None

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

    """_sum_vars = ['MWh', 'capacity', 'avoided_emissions']
    _normal_vars = ['price']
    rows_for_df = {}
    _rows_for_df = []

    _dfs = ['NO_NO__mix', 'NO_YES__mix', 'YES_NO__mix', 'YES_YES__mix', "NONE_YES_YES__mix"]

    for var in _sum_vars + _normal_vars:
        for _ in _dfs:
            for _type in ['total', 'green', 'dirty']:
                df = dfs[_]
                _loc = _locs(df, _type)
                _sum = True if var in _sum_vars else False
                dicts = line_generator(var, df, 'period', 'period', _remove_outliers=True, _loc=_loc, _sum=_sum)
                line = line_generator_pd(dicts[0], dicts[1], var)
                rows_for_df[_ + _type + var] = line
                _rows_for_df.append([_ + ' ' + _type + ' ' + var, line[1][0], line[1][1]])

                if var in ['price', 'avoided_emissions']:
                    break

    print(rows_for_df)
    main_df = pd.DataFrame(_rows_for_df, columns=['Type', 'MWhmean', 'MWhgrowth'])
    main_df.to_csv('report' + '.csv', index=False)"""

    """
    Agents graphs first
    """


    pre_titles = ["05_YES_YES__agents", "0_YES_YES__agents", "1_NO_NO__agents", "1_NO_YES__agents", "1_YES_NO__agents"]

    # pre_titles = ['YES_YES__agents']  # uncomment and put a certain dataframe name to avoid checking all

    for pre_title in pre_titles:
        agents_df = dfs[pre_title]
        list_o_var = ['LSS_tot', 'LSS_weak']
        private_var = 'profits'
        public_var = 'current_state'
        if 'YES_YES' in pre_title:
            public_var = 'current_state_norm'
            max_bndes = agents_df.loc[agents_df['genre'] == 'DBB']['current_state'].max()
            max_epm = agents_df.loc[agents_df['genre'] == 'EPM']['current_state'].max()

            min_bndes = agents_df.loc[agents_df['genre'] == 'DBB']['current_state'].min()
            min_epm = agents_df.loc[agents_df['genre'] == 'EPM']['current_state'].min()

            for row in range(0, len(agents_df)):
                if agents_df.loc[row, :]['genre'] == 'DBB':
                    agents_df.loc[row, 'current_state_norm'] = (agents_df.loc[row, 'current_state'] - min_bndes) / (max_bndes - min_bndes)
                elif agents_df.loc[row, :]['genre'] == 'EPM':
                    agents_df.loc[row, 'current_state_norm'] = (agents_df.loc[row, 'current_state'] - min_epm) / (max_epm - min_epm)

        private = {'df': agents_df.loc[(agents_df['genre'] == 'EP') | (agents_df['genre'] == 'TP')], 'var': private_var}
        public = {'df': agents_df.loc[(agents_df['genre'] != 'EP') & (agents_df['genre'] != 'TP') & (agents_df['genre'] != 'DD')], 'var': public_var}

        graphs_list = [
            {
                "name_o_var": 'Number of adaptations' + ' ' + pre_title,
                "var": 'LSS_tot',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Number of adaptations of Technology producers' + ' ' + pre_title,
                "var": 'LSS_tot',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Number of adaptations of Energy providers' + ' ' + pre_title,
                "var": 'LSS_tot',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Full number of adaptations' + ' ' + pre_title,
                "var": 'LSS_weak',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Full number of adaptations of Technology producers' + ' ' + pre_title,
                "var": 'LSS_weak',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Full number of adaptations of Energy providers' + ' ' + pre_title,
                "var": 'LSS_weak',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Money' + ' ' + pre_title,
                "var": 'wallet',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Money of Technology producers' + ' ' + pre_title,
                "var": 'wallet',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Money of Energy providers' + ' ' + pre_title,
                "var": 'wallet',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Ammount to shareholders' + ' ' + pre_title,
                "var": 'shareholder_money',
                "dataframe": agents_df,
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Ammount to shareholders of Technology producers' + ' ' + pre_title,
                "var": 'shareholder_money',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Ammount to shareholders of Energy providers' + ' ' + pre_title,
                "var": 'shareholder_money',
                "dataframe": agents_df.loc[agents_df['genre'] == 'EP'],
                "x_axis": 'period'
            },
            {
                "name_o_var": 'Investment in capacity of Technology producers' + ' ' + pre_title,
                "var": 'capacity',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period',
            },
            {
                "name_o_var": 'Investment in R&D to shareholders of Technology producers' + ' ' + pre_title,
                "var": 'RandD',
                "dataframe": agents_df.loc[agents_df['genre'] == 'TP'],
                "x_axis": 'period',
            },
            {
                "name_o_var": 'Remaining demand' + ' ' + pre_title,
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
            simple_graph(graph['name_o_var'], graph['var'], graph['dataframe'], graph['x_axis'], groupby=None,
                         remove_outliers=True, show=False, log_y=log_y, log_x=log_x)
        simple_graph('Remaining demand' + ' ' + pre_title, 'Remaining_demand', agents_df.loc[agents_df['genre'] == 'DD'], 'period',
                     groupby=None, remove_outliers=True, show=False)

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
            number = 36
            list_o_graphs = [[private_var, private, 'LSS_tot', private],
                             [private_var, private, 'LSS_weak', private]]

        for _graph in list_o_graphs:
            x_axis_name = 'public agents' if _graph[1]['var'] == public['var'] else 'private agents'
            y_axis_name = 'public agents' if _graph[3]['var'] == public['var'] else 'private agents'
            # name = _graph[0] + ' in relation to ' + _graph[1]
            for speed in [True, False]:
                # name = name + ' (every ' + str(time) + ' months) '
                if speed is True:
                    for time in [1, 12]:
                        # name = 'speed of ' + name
                        try:
                            scatter_graph(
                                [_graph[0], _graph[1]['df'], False, x_axis_name],
                                [_graph[2], _graph[3]['df'], False, y_axis_name],
                                speed=speed,
                                show=False,
                                time=time,
                                _name=str(number) + ' - ' + pre_title + ' ' + 'speeds of ' + _graph[0] + ' of ' + x_axis_name + ' in relation to ' + _graph[
                                    2] + ' of ' + y_axis_name + ' (every ' + str(time) + ' months) ',
                            )
                        except:
                            print('We failed with the graph numbah ' + str(number) + ' of the ' + pre_title)
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
                                    _name=str(number) + ' - ' + pre_title + ' ' + _graph[0] + ' of ' + x_axis_name + ' in relation to ' + _graph[
                                        2] + ' of ' + y_axis_name + norm_name,
                                    normalization=normalization)
                                number += 1
                        else:
                            # name = name + ' (actual values)'
                            scatter_graph(
                                [_graph[0], _graph[1]['df'], norm, x_axis_name],
                                [_graph[2], _graph[3]['df'], norm, y_axis_name],
                                speed=speed,
                                show=False,
                                _name=str(number) + ' - ' + pre_title + ' ' + _graph[0] + ' of ' + x_axis_name + ' in relation to ' + _graph[
                                    2] + ' of ' + y_axis_name + ' (actual values)')
                            number += 1

    """
    mix graphs now
    """

    pre_titles = ["05_YES_YES__mix", "0_YES_YES__mix", "1_NO_NO__mix", "1_NO_YES__mix", "1_YES_NO__mix"]
    for pre_title in pre_titles:

        mix_df = dfs[pre_title]
        pathfile = 'Figures/'

        ### Doing a graph for the generation of electricity

        fig = go.Figure()

        thermal_iqr, thermal_max_min, thermal_median = simple_add_trace('Electricity generated by thermal', 'MWh',
                                                                        mix_df.loc[
                                                                            (mix_df['status'] == 'contracted') & (
                                                                                        mix_df['source'] == 0)],
                                                                        'period', groupby=None, remove_outliers=True,
                                                                        color='232,126,4', _sum=True)

        """fig.add_trace(go.Scatter(name=thermal_iqr['name'],
                                 x=thermal_iqr['x'],
                                 y=thermal_iqr['y'],
                                 fill=thermal_iqr['fill'],
                                 fillcolor=thermal_iqr['fillcolor'],
                                 line=thermal_iqr['line']))"""

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
                                                                  _sum=True)

        """fig.add_trace(go.Scatter(name=solar_iqr['name'],
                                 x=solar_iqr['x'],
                                 y=solar_iqr['y'],
                                 fill=solar_iqr['fill'],
                                 fillcolor=solar_iqr['fillcolor'],
                                 line=solar_iqr['line']))"""

        fig.add_trace(go.Scatter(name=solar_median['name'],
                                 x=solar_median['x'],
                                 y=solar_median['y'],
                                 mode=solar_median['mode'],
                                 line=solar_median['line']))

        wind_iqr, wind_max_min, wind_median = simple_add_trace('Electricity generated by wind', 'MWh',
                                                               mix_df.loc[(mix_df['status'] == 'contracted') & (
                                                                           mix_df['source'] == 1)],
                                                               'period', groupby=None, remove_outliers=True,
                                                               color='4,126,232', _sum=True)

        """fig.add_trace(go.Scatter(name=wind_iqr['name'],
                                 x=wind_iqr['x'],
                                 y=wind_iqr['y'],
                                 fill=wind_iqr['fill'],
                                 fillcolor=wind_iqr['fillcolor'],
                                 line=wind_iqr['line']))"""

        fig.add_trace(go.Scatter(name=wind_median['name'],
                                 x=wind_median['x'],
                                 y=wind_median['y'],
                                 mode=wind_median['mode'],
                                 line=wind_median['line']))

        fig.update_layout(
            title='Electricity generation ' + pre_title,
            xaxis_title='period',
            yaxis_title='MWh',
            barmode='overlay',
            template="simple_white")

        file_name = 'Electricity generation' + '_' + pre_title + ".html"
        pathfile = 'Figures/'

        fig.write_html(pathfile + file_name)

        # fig.show()

        graphs_list = [
            {
                "name_o_var": 'Avoided emissions' + ' ' + pre_title,
                "var": 'avoided_emissions',
                "dataframe": mix_df,
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Total capacity' + ' ' + pre_title,
                "var": 'capacity',
                "dataframe": mix_df,
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Wind capacity' + ' ' + pre_title,
                "var": 'capacity',
                "dataframe": mix_df.loc[mix_df['source'] == 1],
                "x_axis": 'period',
                'sum': True
            },
            {
                "name_o_var": 'Solar capacity' + ' ' + pre_title,
                "var": 'capacity',
                "dataframe": mix_df.loc[mix_df['source'] == 2],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Thermal capacity' + ' ' + pre_title,
                "var": 'capacity',
                "dataframe": mix_df.loc[mix_df['source'] == 0],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Price' + ' ' + pre_title,
                "var": 'price',
                "dataframe": mix_df,
                "x_axis": 'period',
            },
            {
                "name_o_var": 'Electricity produced' + ' ' + pre_title,
                "var": 'MWh',
                "dataframe": mix_df.loc[mix_df['status'] == 'contracted'],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Electricity produced by solar' + ' ' + pre_title,
                "var": 'MWh',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 1)],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Electricity produced by wind' + ' ' + pre_title,
                "var": 'MWh',
                "dataframe": mix_df.loc[(mix_df['status'] == 'contracted') & (mix_df['source'] == 2)],
                "x_axis": 'period',
                "sum": True
            },
            {
                "name_o_var": 'Electricity produced by thermal' + ' ' + pre_title,
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
            simple_graph(graph['name_o_var'], graph['var'], graph['dataframe'], graph['x_axis'], groupby=None,
                         remove_outliers=True, show=False, log_y=log_y, log_x=log_x, _sum=_sum)

    duration = 1500  # milliseconds
    freq = 880  # Hz
    winsound.Beep(freq, duration)
