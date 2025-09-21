import os
import pandas as pd
import numpy as np
import openpyxl

from scipy import stats
from scipy.stats import linregress
from scipy.interpolate import interp1d

from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error

from yellowbrick.cluster import KElbowVisualizer

from datetime import date

import holoviews as hv
import colorcet
# from holoviews.operation import histogram
from bokeh.models import HoverTool
hv.extension('bokeh')

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from IPython.display import SVG


import numpy as np
from scipy.spatial import distance

def rescale_data(x):
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)

def Ripley_K(x, scale):
    x = rescale_data(x)
    x_pairs = distance.cdist(x, x, 'euclidean')  # All pairwise distances
    return np.sum(x_pairs <= scale) / len(x)

def Ripley_L(x, scale):
    x = rescale_data(x)
    return np.sqrt(Ripley_K(x, scale) / np.pi)

# improved definiton of r2_score (https://stats.stackexchange.com/questions/590199/how-to-motivate-the-definition-of-r2-in-sklearn-metrics-r2-score) (scikitlearn uses out-of-sample y_mean)
def r2_score(y_train, y, y_pred):  #y_train and y can be the same if determining r2 for training data (use y_train and y_validation for validation data)
    y_bar = np.mean(y_train)
    RSS = np.sum((y - y_pred)**2)   # Residual Sum of Squares
    TSS = np.sum((y - y_bar)**2)    # Total Sum of Squares
    r2_score = 1 - (RSS / TSS)
    return r2_score

## Allow for user to input both name (string) or index number for column headers
def get_column_loc(column, dataframe):
    if isinstance(column, str):
        return dataframe.columns.get_loc(column), column
    else:
        return column, dataframe.columns[column]

# adapted from https://birdlet.github.io/2018/06/06/rdkit_svg_web/
def DrawMol(dataframe, smiles_column_loc, image_column, molSize=(200, 100), kekulize=True):
    images = []
    for smiles_string in dataframe.iloc[:, smiles_column_loc]:
        try:
            mc = Chem.MolFromSmiles(smiles_string)
            if kekulize:
                try:
                    Chem.Kekulize(mc)
                except:
                    mc = Chem.Mol(smiles_string.ToBinary())

            if not mc.GetNumConformers():
                Chem.rdDepictor.Compute2DCoords(mc)

            drawer = rdMolDraw2D.MolDraw2DSVG(*molSize)
            drawer.DrawMolecule(mc)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace('svg:', '')
            images.append(SVG(svg).data)
        except:
            images.append(None)
    
    try:
        dataframe.insert(smiles_column_loc+1, image_column, images)
    except: #  reason for error 
        dataframe[image_column] = images
        
    return dataframe


def scatter_plot(
    dataframe: pd.DataFrame, 
    x: str,  # x-axis data
    y: str,  # y-axis data
    title: str = '',  # title of plot
    x_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    x_range: tuple = None,  # range of x-axis
    y_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    y_range: tuple = None,  # range of y-axis
    legend: str = '',  # string with data label if using classifiers/building plots by category
    legend_position: str = 'bottom_right',  # location of legend (default: 'top_left')
    svgs: str = None,  # string with column name of svgs 
    hover_list: list = None,  # list of column names with data to be shown on hover 
    marker: str = 'o',  # marker type - most of the matplotlib markers are supported (https://matplotlib.org/stable/api/markers_api.html)
    bubbleplot: bool = False,  # if True, will create a bubble plot
    size: int = 10,  # size of markers (recommended: 10-20)
    bubblesize: str = None,  # string with column name for size of points in bubbleplot
    heatmap: bool = False,  # if True, will create a heatmap
    heatmap_col: str = '',  # color of heatmap
    heatmap_label: str = 'default', # label for heatmap colorbar
    heatmap_color: str = 'Plasma',  # color of heatmap
    color: str = '#931319',  # color of markers
    outline: str = '#29323d',  # color of marker outline
    line_width: int = 1,  # width of marker outline
    alpha: int = 1,  # transparency of markers
    groupby: str = None,  # string with column name to group data by
    height: int = 500,  #plot height (recommended: 500)
    width: int = 500,  #plot width (recommended: 500)
    fontscale: int = 1.2,  # scale of font size
):
    
    """
    scatter_plot function based off of HoloViews 'Scatter' element. See documentation for more information:
    hv.help(hv.Scatter)
    https://holoviews.org/reference/elements/bokeh/Scatter.html
    """

    if x_label == 'default':  # if no x_label provided, use x column name
        x_label = x
    if y_label == 'default':  # if no y_label provided, use y column name
        y_label = y
    if heatmap_label == 'default':  # if no heatmap_label provided, use heatmap_col column name
        heatmap_label = heatmap_col

    if not x_range:
        x_min = min(dataframe[x]); x_max = max(dataframe[x])
        x_buffer = abs(x_max-x_min)/10
        x_range = (x_min-x_buffer, x_max+x_buffer)
    if not y_range:
        y_min = min(dataframe[y]); y_max = max(dataframe[y])
        y_buffer = abs(y_max-y_min)/10
        y_range = (y_min-y_buffer, y_max+y_buffer)

    if groupby is not None and hover_list is not None:
        # color = hv.Cycle(color).values
        hover_list.insert(0, groupby)

    if svgs == None and hover_list == None: # no hover information provided
        if title == 'default':  # if no title provided, define from x, y labels
            title = f'{y_label} vs. {x_label}'
        plt = hv.Scatter(dataframe, kdims=[x], vdims=[y], label=legend).opts(title=title, xlabel=x_label, ylabel=y_label, align='center', marker=marker, legend_position=legend_position, height=height, width=width, color=color, alpha=alpha, size=size, line_color=outline, line_width=line_width, fontscale=fontscale)
    else:  # hover information provided, build list of hover tools
        hover_list.insert(0, y)
        tooltips = f'<div>end' # beginning of tooltips if no svgs provided
        if svgs != None:
            tooltips = f'<div><div>@{svgs}{{safe}}</div>end'  # beginning of tooltips if svgs are provided
            hover_list.insert(1, svgs)
        if len(hover_list) < 4:
            for label in hover_list:
                if label != svgs and label != y:
                    tooltips = tooltips.replace('end', f'<div><span style="font-size: 17px; font-weight: bold;">@{label}</span></div>end')
        else:
            for label in hover_list:
                if label != svgs and label != y:
                    tooltips = tooltips.replace('end', f'<div><span style="font-size: 12px;">{label}: @{label}</span></div>end')
        
        tooltips = tooltips.replace('end', '</div>')
        hover = HoverTool(tooltips=tooltips)
        if heatmap == False and bubbleplot == False:  # if no heatmap or bubbleplot, build scatter plot  
            if title == 'default':  # if no title provided, define from x, y labels
                title = f'{y_label} vs. {x_label}'          
            plt = hv.Scatter(dataframe, kdims=[x], vdims=hover_list, label=legend).opts(title=title, xlabel=x_label, ylabel=y_label, align='center', marker=marker, legend_position=legend_position, height=height, width=width, tools=[hover], color=color, alpha=alpha, size=size, line_color=outline, line_width=line_width, fontscale=fontscale)

        elif heatmap == True and bubbleplot == False:
            if heatmap_col not in hover_list:
                hover_list.append(heatmap_col)
            if title == 'default':  # if no title provided, define from x, y labels
                title = f'{y_label} vs. {x_label}, colored by {heatmap_col}'
            plt = hv.Scatter(dataframe, kdims=[x], vdims=hover_list, label=legend).opts(title=title, xlabel=x_label, ylabel=y_label, align='center', marker=marker, height=height, width=width, tools=[hover], color=heatmap_col, cmap=heatmap_color, colorbar=True, clabel=heatmap_label, alpha=alpha, size=size, line_color=outline, line_width=line_width, fontscale=fontscale)

        elif heatmap == False and bubbleplot == True:
            if bubblesize not in hover_list:
                hover_list.append(bubblesize)
            if title == 'default':  # if no title provided, define from x, y labels
                title = f'{y_label} vs. {x_label}, sized by {bubblesize}'
            min_size = min(dataframe[bubblesize]); max_size = max(dataframe[bubblesize])
            plt = hv.Scatter(dataframe, kdims=[x], vdims=hover_list, label=legend).opts(title=title, xlabel=x_label, ylabel=y_label, align='center', marker=marker, height=height, width=width, tools=[hover], color=color, alpha=alpha, size=((hv.dim(bubblesize)-min_size)/(max_size-min_size)*(max_size-min_size)+min_size)*6*size, line_color=outline, line_width=line_width, fontscale=fontscale)

        elif heatmap == True and bubbleplot == True:
            if heatmap_col not in hover_list:
                hover_list.append(heatmap_col)
            if bubblesize not in hover_list:
                hover_list.append(bubblesize)

            if title == 'default':
                title = f'{y_label} vs. {x_label}, colored by {heatmap_col}, sized by {bubblesize}'
            min_size = min(dataframe[bubblesize]); max_size = max(dataframe[bubblesize])
            plt = hv.Scatter(dataframe, kdims=[x], vdims=hover_list, label=legend).opts(title=title, xlabel=x_label, ylabel=y_label, align='center', marker=marker, legend_position=legend_position,height=height, width=width, tools=[hover], color=heatmap_col, cmap=heatmap_color, colorbar=True, clabel=heatmap_label, alpha=alpha, size=((hv.dim(bubblesize)-min_size)/(max_size-min_size)*(max_size-min_size)+min_size)*6*size, line_color=outline, xlim=x_range, ylim=y_range, line_width=line_width, fontscale=fontscale)
        
        if groupby != None:
            # color = hv.Cycle(color).values
            plt = plt.opts(color=groupby, cmap=color)

        return plt
        
def plot_slope(
    dataframe: pd.DataFrame, 
    x: str,  # string with column name, used to determine slope
    y: str,  # string with column name, used to determine slope
    x_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    y_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    color: str = '#000000',  # color of slope line
    line_width: int = 2,  # width of slope line
    alpha: int = 1,  # transparency of slope line
    height: int = 500,  #plot height (recommended: 500)
    width: int = 500  #plot width (recommended: 500)
):
    
    
    if x_label == 'default':  # if no x_label provided, use x column name
        x_label = x
    if y_label == 'default':  # if no y_label provided, use y column name
        y_label = y

    slope, intercept, r_value, p_value, std_err = stats.linregress(dataframe[x], dataframe[y])
    slope_plt = hv.Slope(slope, intercept).opts(xlabel=x_label, ylabel=y_label, line_color=color, line_width=line_width, alpha=alpha, height=height, width=width)
    return slope_plt, r_value


def plot_confidenceinterval(
        dataframe: pd.DataFrame,  # dataframe
        x: str,  # string with column name, used to determine confidence interval
        y: str,  # string with column name, used to determine confidence interval
        x_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
        x_range: tuple = None,  # range of x-axis
        y_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
        y_range: tuple = None,  # range of y-axis
        ci: int = 0.999,  # confidence interval (0.9-0.99 recommended)
        color: str = '#5289a1',  # color of confidence interval
        outline: str = '#FFFFFF',  # color of confidence interval line
        alpha: int = 0.2,  # transparency of confidence interval
        height: int = 500,  #plot height (recommended: 500)
        width: int = 500  #plot width (recommended: 500)
):
        
    """ 
    Confidence interval calculations use inferences made on the mean and variance of the distributed data (assumes normal distribution)
    and is calculated by applying a student-t test. Plotting function based off of HoloViews 'Area' element as 'area between curves'. 
    See documentation for more information:
    hv.help(hv.Area)
    https://holoviews.org/reference/elements/bokeh/Area.html
    
    """  

    if x_label == 'default':  # if no x_label provided, use x column name
        x_label = x
    if y_label == 'default':  # if no y_label provided, use y column name
        y_label = y

    if not x_range:
        x_min = min(dataframe[x]); x_max = max(dataframe[x])
        x_buffer = abs(x_max-x_min)/10
        x_range = (x_min-x_buffer, x_max+x_buffer)
    if not y_range:
        y_min = min(dataframe[y]); y_max = max(dataframe[y])
        y_buffer = abs(y_max-y_min)/10
        y_range = (y_min-y_buffer, y_max+y_buffer)

    n = len(dataframe[x])
    t_value = stats.t.ppf(1 - (1 - ci) / 2, n - 2)  # t-value for confidence interval (student-t test for n-2 degrees of freedom)
    x_mean = np.mean(dataframe[x])  # mean of x values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(dataframe[x], dataframe[y])

    S_xx = (n * np.sum(dataframe[x] ** 2) - np.sum(dataframe[x]) ** 2) / n  # sample-corrected sum of squares (sum of the square of the difference between x and its mean)
    S_xy = (n * np.sum(dataframe[x] * dataframe[y]) - np.sum(dataframe[x]) * np.sum(dataframe[y])) / n  # sample-corrected covariance for x and y 
    S_yy = (n * np.sum(dataframe[y] ** 2) - np.sum(dataframe[y]) ** 2) / n  # sample-corrected sum of squares (sum of the square of the difference between y and its mean)
    
    SSE = S_yy - slope * S_xy # sum of squared estimate of errors (deviation of the observed value from the estimated value)
    s2 = SSE / (n - 2)  #variance of the x, y data
    s = np.sqrt(s2)  # standard deviation of the x, y data

    unique_x = np.unique(dataframe[x])  # unique x values (prevents overplotting of confidence interval)
    mean_upperconfidence_list = slope * unique_x + intercept + t_value * s * np.sqrt((1 / n + (np.square(unique_x - x_mean)) / S_xx))  # line for upper confidence interval
    mean_lowerconfidence_list = slope * unique_x + intercept - t_value * s * np.sqrt((1 / n + (np.square(unique_x - x_mean)) / S_xx))  # line for lower confidence interval

    upper_spread = interp1d(x=unique_x, y=mean_upperconfidence_list, kind='quadratic', fill_value='extrapolate')  # interpolation function for upper confidence interval (smooths line)
    lower_spread = interp1d(x=unique_x, y=mean_lowerconfidence_list, kind='quadratic', fill_value='extrapolate')  # interpolation function for lower confidence interval (smooths line)

    ci_x = np.linspace(min(unique_x) - abs(max(unique_x) - min(unique_x)) / 2, max(unique_x) + abs(max(unique_x) - min(unique_x)) / 2, num=1000)  # x values for confidence interval plot (extends beyond data range)
    ci_upper_y = upper_spread(ci_x)  # y values for upper confidence interval plot corresponding to 'extended' x values
    ci_lower_y = lower_spread(ci_x)  # y values for lower confidence interval plot corresponding to 'extended' x values

    # plot confidence interval
    ci_plt = hv.Area((ci_x, ci_upper_y, ci_lower_y), vdims=['ci_y1', 'ci_y2']).opts(xlabel=x_label, ylabel=y_label, color=color, alpha=alpha, line_color=outline, height=height, width=width, xlim=x_range, ylim=y_range)
    return ci_plt

def bar_graph(
    dataframe: pd.DataFrame,
    x: str,  # string with column name, used to determine x-axis
    y: str,  # string with column name, used to determine y-axis
    x_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    y_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)

    title: str = 'default',  # title of plot
    discrete_x: bool = False,  # if True, will create a bar graph with discrete x-axis
    svgs: str = None,  # string with column name of svgs 
    hover_list: list = None,  # list of column names with data to be shown on hover 
    color: str = '#5289a1',  # color of bars
    alpha: int = 1,  # transparency of bars
    height: int = 500,  #plot height (recommended: 500)
    width: int = 500  #plot width (recommended: 500)
):
    
    """ 
    bar_graph function (if continuous x-axis) based off of HoloViews 'Histogram' element. See documentation for more information:
    hv.help(hv.Histogram)
    http://dev.holoviews.org/reference/elements/bokeh/Histogram.html

    for non-continuous x-axis, bar_graph function is based on 'hv.Bars' element. See documentation for more information:
    hv.help(hv.Bars)
    http://dev.holoviews.org/reference/elements/bokeh/Bars.html
    
    """

    if x_label == 'default':  # if no x_label provided, use x column name
        x_label = x
    if y_label == 'default':  # if no y_label provided, use y column name
        y_label = y
    if title == 'default':  # if no title provided, define from x, y labels
        title = f'{y_label} vs. {x_label}'

    
    if discrete_x == False:  # continuous x-axis, use Histogram element
        if svgs == None and labels == None:
            plt = hv.Histogram(dataframe, kdims=[x], vdims=[y]).opts(xlabel=x_label, ylabel=y_label, title=title, color=color, alpha=alpha, height=height, width=width)
        else: 
            hover_list.insert(0, y)
            tooltips = f'<div>end' # beginning of tooltips if no svgs provided
            if svgs != None:
                tooltips = f'<div><div>@{svgs}{{safe}}</div>end'  # beginning of tooltips if svgs are provided
                hover_list.insert(1, svgs)
            if len(hover_list) < 4:
                for label in hover_list:
                    if label != svgs and label != y:
                        tooltips = tooltips.replace('end', f'<div><span style="font-size: 17px; font-weight: bold;">@{label}</span></div>end')
            else:
                for label in hover_list:
                    if label != svgs and label != y:
                        tooltips = tooltips.replace('end', f'<div><span style="font-size: 12px;">{label}: @{label}</span></div>end')
            
            tooltips = tooltips.replace('end', '</div>')
            hover = HoverTool(tooltips=tooltips)
            plt = hv.Histogram(dataframe, kdims=[x], vdims=hover_list).opts(xlabel=x_label, ylabel=y_label, title=title, tools=[hover], color=color, alpha=alpha, height=height, width=width)
    else:  # discrete x-axis, use Bars element
        if svgs == None and labels == None:
            plt = hv.Bars(dataframe, kdims=[x], vdims=[y]).opts(xlabel=x_label, ylabel=y_label, title=title, color=color, alpha=alpha, height=height, width=width)
        else: 
            hover_list.insert(0, y)
            tooltips = f'<div>end' # beginning of tooltips if no svgs provided
            if svgs != None:
                tooltips = f'<div><div>@{svgs}{{safe}}</div>end'  # beginning of tooltips if svgs are provided
                hover_list.insert(1, svgs)
            if len(hover_list) < 4:
                for label in hover_list:
                    if label != svgs and label != y:
                        tooltips = tooltips.replace('end', f'<div><span style="font-size: 17px; font-weight: bold;">@{label}</span></div>end')
            else:
                for label in hover_list:
                    if label != svgs and label != y:
                        tooltips = tooltips.replace('end', f'<div><span style="font-size: 12px;">{label}: @{label}</span></div>end')
            
            tooltips = tooltips.replace('end', '</div>')
            hover = HoverTool(tooltips=tooltips)
            plt = hv.Bars(dataframe, kdims=[x], vdims=hover_list).opts(xlabel=x_label, ylabel=y_label, title=title, tools=[hover], color=color, alpha=alpha, height=height, width=width, xlim=(min(dataframe[x]), max(dataframe[x])), ylim=(min(dataframe[y]), max(dataframe[y])))
        return plt
    

def kmeans_score(dataframe, k):
    # matplotlib inline
    
    elbow_plot = KElbowVisualizer(KMeans(n_clusters=k, n_init='auto'), random_state=42)
    elbow_plot.fit(dataframe)
    return elbow_plot


def sanitize_column_names(df):
    df.columns = df.columns.str.replace('[^a-zA-Z0-9]', '_')  # replace non-alphanumeric characters with '_'
    df.columns = df.columns.str.replace('[ ,-]', '_', regex=True)  # replace spaces, commas, and hyphens with '_'
    return df

def find_kmeans_centroids(dataframe, centroid_coordinates, chemical_space_coordinate_columns, ligand_id_column):

    kmeans_centroids = []
    for i in range(0, len(centroid_coordinates)):
        euclidian_distances = np.linalg.norm(dataframe[chemical_space_coordinate_columns].values - centroid_coordinates[i], axis=1)
        closest_index = np.argmin(euclidian_distances)
        kmeans_centroids.append(dataframe.iloc[closest_index][ligand_id_column])
    # Check:
    if len(kmeans_centroids) != len(centroid_coordinates):
        raise ValueError("Number of clusters and number of ligand IDs identified as cluster centroids do not match.")
    return kmeans_centroids