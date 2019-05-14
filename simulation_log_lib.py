from numba import jit
import datetime
import os
import warnings
# imports file with all custum functions
from importlib import reload
from pathlib import Path
from random import randint

import matplotlib as mpl
import matplotlib.gridspec as gridspec
# plot libraries
import matplotlib.pyplot as plt
import numpy as np
# scientific libraries
import pandas as pd
import pandas_datareader.data as web
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
# help to construct the rolling correlation
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
# bootstrap methods
from arch.bootstrap import (CircularBlockBootstrap, IIDBootstrap,
                            MovingBlockBootstrap, StationaryBootstrap)
from matplotlib.ticker import FormatStrFormatter
from pandas_datareader.famafrench import get_available_datasets
from scipy.stats.mstats import gmean
#import brewer2mpl
from statsmodels.graphics.tsaplots import plot_pacf
from tqdm import tqdm, tqdm_notebook  # progress bars

warnings.filterwarnings("ignore")
# os.chdir("C:\\Users\Pieter-Jan\Documents\Factor_Crashes\Poster")

#import numba as nb
#from numba import jitclass

# plot parameters
sns_params = {
    'font.family': 'serif',
    'font.size': 12,
    'font.weight': 'medium',
    'figure.figsize': (10, 7),
}
plt.style.use('seaborn-talk')
plt.style.use('bmh')
sns.set_context(sns_params)
savefig_kwds = dict(dpi=300, bbox_inches='tight', frameon=True, format='png')
nanex_colors = ("#f92b20", "#fe701b", "#facd1f", "#d6fd1c", "#65fe1b",
                "#1bfe42", "#1cfdb4", "#1fb9fa", "#1e71fb", "#261cfd")
nanex_cmap = mpl.colors.ListedColormap(nanex_colors, name='nanex_cmap')
plt.register_cmap('nanex_cmap', cmap=nanex_cmap)
set2 = [
    '#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9'
]
#random state
np.random.seed(31415) #random state

########################################################################################################################################

def cumsum_plot(size=(15, 8),
                ylab='Growth of $1 invested (log scale)',
                xlab="Date",
                markers=["o", "v", "*", "X"],
                time_freq=252,
                fontsize_title=18,
                simple=True,
                df=None,
                dic=None,
                title=None,
                ax=None):
    """
    plots the cumulative return

    params:
    -------

        - df: pandas dataframe with the raw returns
        - size: tuple determining the figure size
        - ylab: string indicating the ylabel
        - xlab: string indicating the xlabel
        - dic:  dictionary for changing the legend names
        - title: string indicating the title
        - markers:  list with marker symbols to indicate the different factors
        - time_freq: integer indicating the time frequency (eg. daily: 252)
        - ax: matplotlib object to specify your own ax
        - fontsize_title: integer indicating the fontsize of the title


    returns:
    --------

        - plot via ax
    """
    if not ax:
        fig, (ax) = plt.subplots(1, 1, figsize=size)
        
    for i, c in enumerate(df.columns):
        ax.plot(df.index, (1 + df[c] / 100).cumprod(),
                label=dic.get(c) if dic is not None else str(c),
                alpha=1,
                marker=None if df.shape[1] > 4 else markers[i],
                markevery=time_freq*5,
                markersize=8,
                linestyle='--',
                linewidth=2)
    ax.set_title(title,
                 y=1.02,
                 fontsize=fontsize_title,
                 fontweight="bold",
                 style="italic",
                 ha="center")
    ax.set_ylabel(ylab,fontsize=14)
    ax.set_xlabel(xlab,fontsize=14)
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlim(df.index[0],
                df.index[-1])
    ax.grid(True, which='major', linestyle='-.', color='0.8')



########################################################################################################################################

def kappa_ratio(threshold_return=0, threshold_risk=0, order=3, time_freq=None, df=None):
    """
    calculates the kappa ratio(s) as explained above

    params:
    -------

        - df: pandas dataframe with returns
        - treshold_risk: integer that determines when deviations are taken into account
        - treshold_return: calculate excess return
        - order: float indicating the magnitude of the deviations
        - time_frequency: integer indicating the number of trading days

    returns:
    --------

        - a numpy array with the kappa ratio(s)
    """

    diff = threshold_risk-df
    # Set the minimum of each to 0
    diff = pd.DataFrame((diff).clip(0))
    # Return the sum of the different to the power of order
    risk = ((diff**order).mean())**(1 / order) * (time_freq)**(1 / order)
    r = np.mean(df.values,axis=0) * time_freq
    return r/risk


########################################################################################################################################


def drawdown(df=None):
    """
    calculate drawdown

    params:
    -------

        - df: pandas dataframe with the raw returns

    returns
    -------

        - numpy array with drawdown
    """

    p = np.cumsum((df/100).values, axis=0)
    rm = np.maximum.accumulate(p)
    return 1-np.exp(p-rm)

########################################################################################################################################


@jit
def max_time_dd(df=None):
    """
    calculate max number of succesive days your in a drawdown state

    params:
    -------

        - numpy array

    returns
    -------

        - numpy array with various local maximums
    """

    store = np.zeros(len(df))
    prev = 0.0
    for i in range(len(df)):

        if df[i] == 0.0:
            prev = 0.0
        else:
            prev += 1
            store[i] = prev

    return store

########################################################################################################################################


def perf_stat(
        stats=[
            "Arithmetric Return (%)", "Arithmetric Volatility (%)", "Max Drawdown (%)"],
        all_stats=False,
        df=None,
        dic_perf=None):
    """
    appends the performance statistics in a pandas data frame

    params:
    -------

        - df: pandas dataframe with the raw returns
        - dic_perf: dictionary with the functions to 
        - all_stats: boolean indicating whether 
                     to calculate alles stats in the dictionary
          calculate the (performance) statistics of interest
        - stats: list indicating the statsitics to calculate

    returns:
    --------

        - pandas dataframe with the calculated statistics
    """

    values = []
    index = []
    stats_s = dic_perf.keys() if all_stats else stats
    for s in stats_s:
            values.append(dic_perf.get(s)(df=df))
            index.append(s)

    return pd.DataFrame(values, index=index, columns=df.columns)


########################################################################################################################################


def Return_DD(markers=["o", "v", "*", "X"],
              factor=['Mkt-RF', 'HML', 'WML', 'HML-WML'],
              loc="best",
              figsize=(12, 12),
              height_ratio=[2.25, .7, .6, .7, .6],
              add_rf=False,
              final_value=False,
              df=None,
              dic_labels=None,
              start=None,
              end=None,
              time_frequency=None):
    """
    plots the cumulative return and drawdown

    params:
    -------

        - df: pandas dataframe with the raw returns
        - dic_labels: dictionary with for changing the labels in the legend
        - markers:  list with marker symbols to indicate the different factors
        - factor: a list with the factors to plate
        - start: date object, indicating the start date
        - end: date object, indicating the ending date
        - loc: string indicating the locating of the legend for the dd figures
        - figsize: tuple indicating the figure size
        - height_ratio: list with the heights of the figures
        - time_frequency: integer indicating the time frequency (eg. daily: 252)
        - add_rf: boolean indicating to display the risk free rate
        - final_value: boolean indicating to plot the final value

    returns:
    --------

        - None (plots a figure)
    """

    start = start if start is not None else df.index[0]
    end = end if end is not None else df.index[-1]

    if end is None:
        end = len(df)

    f, (ax) = plt.subplots(len(factor)+1,
                           1,
                           figsize=figsize,
                           gridspec_kw={'height_ratios': height_ratio})

    # convert once for all colummns
    df_exp = np.exp(np.cumsum(df.loc[start:end, :] / 100))
    # index same for all
    index = df.loc[start:end].index

    for i, c in enumerate(df.loc[:, factor].columns):

        # plot the return paths and drawodown
        ax[0].plot(index,
                   df_exp.loc[start:end, c],
                   label=dic_labels.get(c),
                   marker=markers[i],
                   markevery=time_frequency * 5,
                   markersize=8,
                   linestyle='--',
                   linewidth=1.5)
        if c == 'Mkt-RF' and final_value:
            ax[0].annotate('$%0.2f' % df_exp.loc[end, c], xy=(1, df_exp.loc[end, c]-50), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
        elif c == 'WML' and final_value:
            ax[0].annotate('$%0.2f' % df_exp.loc[end, c], xy=(1, df_exp.loc[end, c]+10), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
        elif final_value:
            ax[0].annotate('$%0.2f' % df_exp.loc[end, c], xy=(1, df_exp.loc[end, c]), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
                       
        ax[1 + i].plot(index,
                       drawdown(df.loc[start:end, c]),
                       color="C{}".format(i), lw=2)

        ax[1 + i].fill_between(index,
                               drawdown(df.loc[start:end, c]),
                               0,
                               alpha=.5,
                               label=dic_labels.get(c),
                               color="C{}".format(i))

        # plot style params for the return paths
        ax[0].set_yscale('log')
        ax[0].set_ylabel("Growth of $1 invested (log scale)", fontsize=14)
        ax[0].grid(True, which='major', linestyle='-.', color='0.8')
        ax[0].set_xlim([start, end])

        # plot style params for the drawdown figure
        if i != 0:
            ax[i].xaxis.set_major_formatter(plt.NullFormatter())
            ax[i].xaxis.set_ticks_position('none')
        ax[i + 1].set_xlim(start, end)
        ax[i + 1].grid(True, which='major', linestyle='-.', color='0.8')
        ax[i + 1].legend(loc=loc, frameon=True,framealpha=1)
        ax[i + 1].set_ylabel("Drawdown", fontsize=14)
        ax[len(factor)].set_xlabel("Date", fontsize=16)

        # common style params
        ax[i + 1].yaxis.set_tick_params(labelsize=12)
        ax[i + 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        f.tight_layout()

    # add risk free rate
    if add_rf:

        ax[0].plot(index,
                   df_exp.loc[start:end, "RF"],
                   label="Risk Free Rate (RF)",
                   markersize=8,
                   linestyle='--',
                   linewidth=1.5)
    # add legend
    ax[0].legend(loc="best", frameon=True, framealpha=1)

    # title
    plt.suptitle(f"Portfolio value and drawdown: \
{start:%Y} to {end:%Y}",
                 y=1.02,
                 fontsize=18,
                 fontweight="bold",
                 style="italic",
                 ha="center")

    return None

########################################################################################################################################


def rolling_spearman_quick(seqa=None, seqb=None, window=None):
    
    """
    calculates the rolling correlation (spearman)
    
        seqa: pandas dataframe with return series a
        seqb: pandas dataframe with return series b
        window: innteger indicating the window
    
    retruns numpy array with the rolling correlation
    """
    
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)



########################################################################################################################################


def boot_ci(bootmethod=CircularBlockBootstrap, store={},
            func=None, nr_boot=100, ci_method='basic',
            size=0.95, blocks=None, df=None, portfolio=None):
    
    """
     construct the bootstrap CI estimates for some performance metric
     
      - bootmethod: metod from the arch package, 
                    indicating how to bootstrap (mbb, cbb, sb)
      - store: dictionary to store the estimates
      - func: function for the performance metric
      - ci_method: string indicating how to compute the CI interval
      - size: integer indicating the size of the CI
      - blocks: integer indicating the (average) blockszie
      - df: pandas dataframe with the time series
      - portfolio: string indicating the portfolio to bootstrap
     
      
    """
    
    df = df.loc[:,portfolio]
    bs = bootmethod(blocks, df)
    ci = bs.conf_int(func, nr_boot, method=ci_method,size=size)
    store.append(pd.DataFrame(ci))
    return store


########################################################################################################################################

def SR_A_CI(SR_0=0, alpha=0.05, df=None, higher_moments=False, 
          SR_e=None, Y3_e=None, Y4_e=None, n=None):
    
    
    if n is None:
        n =  len(df)
    if not higher_moments:  
        Y3_e, Y4_e = 0,0
    if SR_e is None:
        SR_e = np.mean(df)/np.std(df)
    if higher_moments and Y3_e is None:
        Y3_e, Y4_e = stats.skew(df), stats.kurtosis(df)
             
    se = np.sqrt((1-Y3_e*SR_e+((Y4_e+3-1)/4)*SR_e**2)/(n-1))
    alpha_out =  stats.norm.ppf(1-alpha/2)
    lower, upper = SR_e-se*alpha_out, SR_e+se*alpha_out
    return(lower,upper)

########################################################################################################################################

def PSR(SR_0=0, df=None, higher_moments=False, 
        SR_e=None, Y3_e=None, Y4_e=None, n=None):
    
    if n is None:
        n = len(df)
    if not higher_moments:    
        Y3_e, Y4_e = 0,0
    if SR_e is None:    
        SR_e = np.mean(df)/np.std(df)
    if higher_moments and Y3_e is None:
        Y3_e, Y4_e = stats.skew(df),stats.kurtosis(df)
            
    numerator = (SR_e-SR_0)*np.sqrt(n-1)
    denominator =  np.sqrt(1-Y3_e*SR_e+((Y4_e+3-1)/4)*SR_e**2)
    T = numerator/denominator
    prob = stats.norm.cdf(T,0,1)
    #print(f"Test statistic PSR: {T}, Probability: {prob}")
    return (T,prob)

########################################################################################################################################

def max_SR(e_var_SR=1/2,N=2):
    return np.sqrt(e_var_SR)*((1-np.euler_gamma)*stats.norm.ppf(1-1/N)\
                       +np.euler_gamma*stats.norm.ppf(1-1/N*np.e**-1))

########################################################################################################################################


def boot(N_paths=None, method=None, obs_path=None, add_noise=False):
    """
    performs the bootstrap methods:

        - Indepent Identical Distributed (IID)
        - Moving Block Bootstrap (MBB)
        - Circular Block Bootstrap (CBB)
        - Stationary Bootstrap (SB)

    params:
    ------- 
        - rf: boolean indicating whether to cross bootstrap the risk free rate
        - N_paths: number of bootstraps (price paths)
        - method: data structure of the arch package
        - add_noise: boolean indicating to add noise to the bootstraps
        - obs_path: pandas dataframe containing the data of the original returns

    returns: 
    --------

        - the bootstrapped simulations, each column is a new price path

    """
    boot_r = []

    for data in method.bootstrap(N_paths):
        boot_r.append(data[0][0].reset_index(drop=True))

    out_b = pd.concat(boot_r, axis=1, ignore_index=True)

    if add_noise:

        scale = np.std(obs_path.values)
        se = (scale/np.sqrt(len(obs_path)))
        n = len(obs_path)
        print(f"adding noise: mean 0 and sd: {se}")
        out_b += np.transpose(np.split(np.random.normal(loc=0,
                                                        scale=se, size=N_paths*n), N_paths))

    return out_b


########################################################################################################################################


def sample_normal(obs_path=None, N=None, add_noise=False):
    """
    samples data from a normal distribution (same lengts as data)

    params:
    ------

        - obs_path: pandas dataframe containing the data of the original returns
        - N: integer indicating the number of samples
        - add_noise: boolean indicating to add noise to the samples

    returns
    -------

        - pandas dataframe with in the columns the generated paths

    """

    mean = np.mean(obs_path.values)
    scale = np.std(obs_path.values)
    n = len(obs_path)

    out_norm = np.transpose(np.split(np.random.normal(loc=mean,
                                                      scale=scale, size=N*n), N))

    if add_noise:
        
        se = (scale/np.sqrt(n))
        print(f"adding noise: mean 0 and sd: {se}")
        out_norm += np.transpose(np.split(np.random.normal(loc=0,
                                                           scale=se, size=N*n), N))

    return pd.DataFrame(out_norm)


########################################################################################################################################


def plot_summary_stats(figsize=(15, 10),
                       bins=25, sims=None,
                       ih=None):
    """

    plots the histogram(s) of the calculated statistic(s)

    params:
    -------

        - sims: pandas dataframe with the calculated statisitc(s) of interest
        - year: integer indicating the investment horizon
        - figsize: tuple indicating the figure size
        - bins: integer indicating the number of bins used to make the histogram

    returns:
    --------

        - None (plots a figure)

    """

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    sims.hist(ax=ax, bins=bins)
    fig.suptitle(f'Investment horizon:\
{f"{ih} year(s)" if ih is not None else "total sample length"}',
                 fontsize=20,
                 y=1.08,
                 fontweight="bold")
    plt.tight_layout()


########################################################################################################################################


def investment_horizons(investment_horizons=[1, 3, 5, 10, 20, 30],
                        simulation_tech='Normal',
                        plotting=True,
                        store_output_dic={},
                        observed_path=None,
                        sims=None,
                        sum_stats=None,
                        perf_functions=None,
                        freq=None):
    """
     wraps around the functions above to perform 
     the bootstrap and storing the necessary information

     params:
     -------

         - observed_path: pandas dataframe with the original return series
         - sims: pandas dataframe
         - investment_horizon: list indicating the investment horizons(s)
         - sum_stats: a list indicating the statistic(s) to calculate
         - store_output_dic: dictionary to store the output information
         - simulation_tech: string indicating the simulation method        
         - plotting: boolean; visualizing the histogram(s) of the statistic(s) 

    returns:
    --------

        - a dictionary with the stored information (statistics)
        - also vizualize some random selected return paths including 
          the original return series
        - if plotting true the histograms for each simulation method
          are visualized
    """

    #  statistic the observed period
    out_Ostats = perf_stat(df=observed_path,
                           stats=sum_stats,
                           dic_perf=perf_functions).T

    #  statistic for the total simulated period
    # you have a matrix with in rows the dates and the columns
    # are the different simulated paths
    out_Sstats = perf_stat(df=sims,
                           stats=sum_stats,
                           dic_perf=perf_functions).T

    if plotting:
        plot_summary_stats(sims=out_Sstats)

    # save performance statistics about total sample length and
    # the observed(obs) vs simulated (sim) statistics
    out_dic = {'Total sample length': {'Simulations': out_Sstats}}

    dic_investment_h = {}  # dictionary for the investment horizons
    for ih in investment_horizons:
        ih_units = int(freq * ih)  # trading days/month in a year
        start = 0
        end = start + ih_units

        #  statistic simulated period conditioned on ih
        # you have a matrix with in rows the dates and the columns
        # are the different simulated paths
        outS_perf = perf_stat(df=sims.iloc[start:end, :],
                              stats=sum_stats,
                              dic_perf=perf_functions).T
        dic_investment_h[ih] = outS_perf

        if plotting:

            # histogram of the statistic
            plot_summary_stats(sims=outS_perf, ih=ih)

    # add dic investment horizon to the outer dictionary
    out_dic['Investment horizon'] = dic_investment_h

    # save performance stats for the investment horizons
    if observed_path.columns[0] in store_output_dic.keys():
        store_output_dic[observed_path.columns[0]][simulation_tech] = out_dic

    else:
        store_output_dic[observed_path.columns[0]] = {simulation_tech: out_dic}

    return store_output_dic


########################################################################################################################################


class Simulation:
    """
    Contains 5 methods to perform the simulation:

        - sample from a normal distribution
        - iid boostrap
        - mbb
        - cbb 
        - sb

    It only uses the investment_horzions function

    """

    def __init__(self,
                 blocksize=126,
                 n_paths=100,
                 investment_horizon=[15],
                 plotting=True,
                 add_noise=False,
                 store_sim = False,
                 data=None,
                 frequency=None,
                 stats=None,
                 perf_functions=None,
                 store_output=None):
        """
        contains the parameters used for all methods:

            - data: pandas dataframe; containg the data
            - blocksize: integer; indcicating the blocksize
              this blocksize is also used for the mean blocksize
              in the stationary bootstrap
            - perf_functions: dictionary; with the performance functions
            - n_paths: integer; indicating the nr of simulations
            - investment horizon: list indicating the investment horizons(s)
            - frequency: integer; indicating the number of periods (e.g. daily 252)
            - stats: list indicating the statistics to calculate          
            - plotting: boolean; visualizing the histogram(s) of the statistic(s)                        
            - store_output: dictionary to store the information
            - add_noise: boolean indicating to add noise to the samples (bootstraps)
        """

        self.data = data
        self.blocksize = blocksize
        self.perf_functions = perf_functions
        self.n_paths = n_paths
        self.ih = investment_horizon
        self.frequency = frequency
        self.stats = stats
        self.plotting = plotting
        self.store_output = store_output
        self.add_noise = add_noise
        self.store_sim = store_sim
        self.simulated_paths = {}

    def normal(self):
        """
        return paths simulated from a noraml distribution 
        using the sample mean and sample standard deviation 
        from the observed data

        params:

            - self: see above

        return:

            - none
        """

        print("NORMAL \n")

        out_normal = sample_normal(obs_path=self.data, N=self.n_paths,
                                   add_noise=self.add_noise)
        if self.store_sim:
            self.simulated_paths['Normal'] = out_normal.iloc[:,:out_normal.shape[1] 
                                                             if out_normal.shape[1]<100 else 100]

        self.store_output = investment_horizons(
            observed_path=self.data,
            sims=out_normal,
            investment_horizons=self.ih,
            freq=self.frequency,
            sum_stats=self.stats,
            perf_functions=self.perf_functions,
            store_output_dic=self.store_output,
            simulation_tech='Normal',
            plotting=self.plotting)

        return None

    def iid_bootstrap(self):
        """  
        return paths simulated using the i.i.d. bootstrap 

        params:

            - self: see above

        return:

            - none
        """

        print("\nIID BOOTSTRAP \n")
        bs = IIDBootstrap(self.data)
        out_iid = boot(N_paths=self.n_paths, method=bs,
                       obs_path=self.data, add_noise=self.add_noise)
        if self.store_sim:
            self.simulated_paths['IB'] = out_iid.iloc[:,:out_iid.shape[1] 
                                                             if out_iid.shape[1]<100 else 100]

        self.store_output = investment_horizons(
            observed_path=self.data,
            sims=out_iid,
            investment_horizons=self.ih,
            freq=self.frequency,
            sum_stats=self.stats,
            perf_functions=self.perf_functions,
            store_output_dic=self.store_output,
            simulation_tech='IB',
            plotting=self.plotting)

        return None

    def mbb_bootstrap(self):
        """
        return paths simulated using the moving block bootstrap 

        params:
        -------

            - self: see above

        return:
        -------

            - none
        """

        print("\nMB BOOTSTRAP \n")
        bs = MovingBlockBootstrap(self.blocksize, self.data)
        out_mbb = boot(N_paths=self.n_paths, method=bs,
                       obs_path=self.data, add_noise=self.add_noise)        
        if self.store_sim:
            self.simulated_paths['MBB'] = out_mbb.iloc[:,:out_mbb.shape[1] 
                                                             if out_mbb.shape[1]<100 else 100]

        self.store_output = investment_horizons(
            observed_path=self.data,
            sims=out_mbb,
            investment_horizons=self.ih,
            freq=self.frequency,
            sum_stats=self.stats,
            perf_functions=self.perf_functions,
            store_output_dic=self.store_output,
            simulation_tech='MBB',
            plotting=self.plotting)

        return None

    def cbb_bootstrap(self):
        """
        return paths simulated using the circular block bootstrap 

        params:
        -------

            - self: see above

        return:
        -------

            - none
        """

        print("\nCIRCULAR BOOTSTRAP \n")
        bs = CircularBlockBootstrap(self.blocksize, self.data)
        out_cbb = boot(N_paths=self.n_paths, method=bs,
                       obs_path=self.data, add_noise=self.add_noise)
        if self.store_sim:
            self.simulated_paths['CBB'] = out_cbb.iloc[:,:out_cbb.shape[1] 
                                                             if out_cbb.shape[1]<100 else 100]

        self.store_output = investment_horizons(
            observed_path=self.data,
            sims=out_cbb,
            investment_horizons=self.ih,
            sum_stats=self.stats,
            freq=self.frequency,
            perf_functions=self.perf_functions,
            store_output_dic=self.store_output,
            simulation_tech='CBB',
            plotting=self.plotting)

        return None

    def sb_bootstrap(self):
        """
        return paths simulated using the stationary  bootstrap 

        params:
        -------

            - self: see above

        return:
        -------

            - none
        """

        print("\nSTATIONARY BOOTSTRAP \n")
        bs = StationaryBootstrap(self.blocksize, self.data)
        out_sbb = boot(N_paths=self.n_paths, method=bs,
                       obs_path=self.data, add_noise=self.add_noise)
        if self.store_sim:
            self.simulated_paths['SB'] = out_sbb.iloc[:,:out_sbb.shape[1] 
                                                             if out_sbb.shape[1]<100 else 100]

        self.store_output = investment_horizons(
            observed_path=self.data,
            sims=out_sbb,
            investment_horizons=self.ih,
            freq=self.frequency,
            sum_stats=self.stats,
            perf_functions=self.perf_functions,
            store_output_dic=self.store_output,
            simulation_tech='SB',
            plotting=self.plotting)

        return None

    def visualize(self, size=(18, 18), nr_paths=50, last_15=None):
        """
          visualizes the simulated price paths for the different simulation methods

          params:
          -------


              - self: see above
              - size: tuple with the figure size
              - nr_pahts: integer indicating the number of paths
              - last_15: pandas dataframe with the last 15 years 
                         of the observed returns
              
          return:
          -------

              - none
         """
        
        fig, ax = plt.subplots(2,
                               round(len(self.simulated_paths.keys()) / 2),
                               figsize=size)
        ax = ax.ravel()
        i = 0
     
        for key, value in self.simulated_paths.items():
            
                           
            nr_paths = value.shape[0] if nr_paths > value.shape[0] else nr_paths
            
            # last 15 years
                 
            if last_15 is not None:
                ih_units = int(self.ih[0]*self.frequency)
                value = value.iloc[:ih_units,:nr_paths]
                # simulated path
                sim_prices = np.exp(np.cumsum(value/100))
                # observed path
                observed_path = last_15.iloc[:ih_units,:]
                
            else:               
                # simulated path
                value = value.iloc[:,:nr_paths]
                sim_prices = np.exp(np.cumsum(value/100))
                # observed path
                observed_path = self.data
                

            # plot simulated path
            #sim_prices = np.exp(np.cumsum(value/100))
            ax[i].plot(sim_prices, alpha=.5, lw=1)

            # plot observed path
            observed_price = np.exp((observed_path / 100).reset_index(
                drop=True).cumsum())
            observed_price.plot(legend=True,
                                ax=ax[i],
                                lw=2,
                                ls='--',
                                color='#AF1705')

            ax[i].set_yscale('log')
            ax[i].set_xlim(0, len(observed_price))
            ax[i].set_title(f'Simulation method: {key}',
                            fontsize=14,
                            fontweight='bold',
                            y=1.02)
             
            i += 1
    
        ax[0].set_ylabel('Growth of $1 invested (log scale)',fontsize=12)
        ax[2].set_ylabel('Growth of $1 invested (log scale)',fontsize=12)
        ax[2].set_xlabel('Time',fontsize=12)
        ax[3].set_xlabel('Time',fontsize=12)
        plt.suptitle(f'{nr_paths} random selected simulated paths',
                     fontsize=16,
                     fontweight="bold",
                     style="italic",
                     y=1.05)

        plt.tight_layout()


########################################################################################################################################


def append_stats(stat="Geometric Return (%)", dic=None):
    """
    appends the calculated statistic from each:
        - strategy
        - simulation method
        - investment horizon
    together in a pandas dataframe

    params:
    -------

        - stats: string indicating the statis
        - dic: dictionary where the statistic is stored

    return:
    -------

        -a pandas dataframe with 4 columns
            - statistic
            - investmenth horizon
            - strategy
            - simulatoin method
    """

    createdf = True
    for key0, value0 in dic.items():
        for key1, value1 in value0.items():
            for key2, value2 in value1['Investment horizon'].items():
                if createdf:
                    a = pd.DataFrame(value2[stat])
                    a["Strategy"] = key0
                    a["Simulation technique"] = key1
                    a["Investment Horizon (year)"] = key2
                    createdf = False
                else:
                    b = pd.DataFrame(value2[stat])
                    b["Strategy"] = key0
                    b["Simulation technique"] = key1
                    b["Investment Horizon (year)"] = key2
                    a = a.append(b)
    return a


########################################################################################################################################


def permutation_plot(statistic='Arithmetic Return (%)',
                     period="1927-2019",
                     hist=False,
                     dic=None,
                     investment_h=None,
                     obs_stats=None):
    """
    plots the distribution of the statistic condtioned on strategy, 
    simulation method and investment horizon
    and simulation method

    params:
    -------

        - dic: dictionary containing the statistic(s)
        - investment_h: integer indicating the investment horizon
        - statistic: string indicating the statistic
        - obs_stats: integer with the observed statistic based on the original returns
        - period: string with time period over which the permutations are calculated 
        - hist: boolean indicating whether to plot the histogram or not

    return:
    -------

        - None (plots a figure)
    """

    f, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()
    i = 0
    df_stat = append_stats(stat=statistic, dic=dic)
    for f, df_f in df_stat.groupby(["Strategy"], sort=False):
        for sim, df_sim in df_f.groupby(["Simulation technique"], sort=False):
            out = df_sim[df_sim["Investment Horizon (year)"] ==
                         investment_h][statistic].values
            sns.distplot(out,
                         ax=axes[i],
                         label=sim,
                         kde_kws={"lw": 4},
                         hist=hist)
        axes[i].set_title(f'Portfolio: {f}',
                          fontsize=14,
                          fontweight="bold",
                          y=1.02)

        # add observed statistic
        if obs_stats is not None:
            axes[i].axvline(x=obs_stats[i],
                            linestyle='--',
                            color="red",
                            label=f"observed statistic",
                            linewidth=3)

        axes[i].get_legend().set_visible(False)  # remove legend
        axes[2].set_xlabel(xlabel=statistic, fontsize=14)
        axes[3].set_xlabel(xlabel=statistic, fontsize=14)
        axes[0].set_ylabel(ylabel="Density", fontsize=14)
        axes[2].set_ylabel(ylabel="Density", fontsize=14)

        # set between 0 and 100 in case of max drawdown
        #if statistic == 'Max Drawdown (%)':
            #axes[i].set_xlim(0, 100)
        i += 1

    #plt.tight_layout()

    # add legend on top
    nr_s_methods = len(dic[list(dic.keys())[0]].keys())
    axes[1].legend(
        bbox_to_anchor=(nr_s_methods /
                        12 if obs_stats is None else nr_s_methods / 12 + .2,
                        1.25),
        ncol=nr_s_methods if obs_stats is None else nr_s_methods + 1,
        prop={'size': 14},
        fontsize="medium")
    plt.suptitle(f'Permuted distribution: {period}\n\
Statistic: {statistic} under a {int(investment_h)} year investment horizon',
                 fontsize=18,
                 fontweight="bold",
                 style="italic",
                 y=1.025)


########################################################################################################################################


def describe_dist(percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99],
                  colname="Mkt-RF",
                  nr_neg=True,
                  standerdize=False,
                  df=None):
    """
    describes the distribtution of the statiscs numerically

     params:
     -------

         - df: pandas dataframe with the statistic
         - percentiles: list indicating the percentiles to calculate
         - colname: string indicating the name of factor (strategy)
         - nr_neg: boolean indication if you want to calculate the nr
          of negative returns for the various investment horizons,
          this is only usefull for statistics like geometric return, 
          not for drawdown
         - standerdize: boolean indicating to standerdize the percentiles

    return:
    -------

          - pandas data series
    """

    count = len(df)
    mean = np.mean(df)
    std = np.std(df)
    minimum, maximum = min(df), max(df)
    kurtosis, skew = stats.kurtosis(df.values), stats.skew(df.values)
    neg_r = sum(df < 0)
    perc = [np.percentile(df, i) for i in percentiles]
    if standerdize:
        perc = ((np.array(perc)-mean)/std).tolist()
    index = ["count", "mean", "std", "min", "max", "kurtosis", "skew"]\
        + ["# neg. R"] if nr_neg else ["count", "mean",
                                       "std", "min", "max", "kurtosis", "skew"]

    calculations = [count, mean, std, minimum, maximum, kurtosis, skew
                    ] + [neg_r] if nr_neg else [
                        count, mean, std, minimum, maximum, kurtosis, skew
    ]
    calculations += perc
    calculations = np.array(calculations).squeeze()
    [index.append(f'{i} %') for i in percentiles]

    return pd.Series(calculations, index)


########################################################################################################################################


def performance_matrix(nr_neg=True,
                       sim_meth=["Normal", "IB", "CBB", "SB"],
                       ih=[1, 3, 5, 10, 20, 30],
                       standerdize=False,
                       dic=None,
                       strategy=None,
                       statistic=None):
    """
    loops over the simulation technique and investment horizon(s)
    to put all information in a nice pandas data matrix

    params:
    -------

        - dic: dictionary with the stored statistic
        - strategy: string indicating the strategy
        - statistic: string indicating the statistic
        - nr_neg: boolean indication if you want to calculate the nr
          of negative returns for the various investment horizons,
          this is only usefull for statistics like geometric return, 
          not for drawdown
        - sim_meth: list indicating the simulation methods
        - ih: list indicating the investment horizon
        - standerdize: boolean indicating to standerdize the percentiles

    return:
    -------

        - a pandas multindex dataframe with 
          the distributed summarized numerically      

    """

    df = append_stats(stat=statistic, dic=dic)
    out = {
        i: describe_dist(df=di[di["Strategy"] == strategy][statistic],
                         nr_neg=nr_neg,
                         standerdize=standerdize)
        for i, di in df.groupby(
            ["Simulation technique", "Investment Horizon (year)"], sort=False)
    }
    df_1 = pd.DataFrame(out).T

    index = pd.MultiIndex.from_product(
        [sim_meth, ih], names=["Simulation method", "Investment Horizon"])

    columns = pd.MultiIndex.from_product(
        [[f'{strategy} : {statistic}'], df_1.columns],
        names=["Strategy and Statistic*", "Descriptives"])
    return pd.DataFrame(df_1.values, index=index, columns=columns)


########################################################################################################################################


def p_values(right=False,
             df=None,
             stat=None,
             observed=None,
             perf_stats_dic=None):
    """
    calculates the p-values

    params:
    -------

        - df: pandas dataframe with the observed statistic
        - stat: string indicating the statistic
        - observed: pandas dataframe with the orginial return series
        - right: boolean indicating the side of the test, 
          to determine what is more extreme

    return:
    -------

        - a pandas dataframe with the p-values
    """

    if right:
        c = {
            i: sum(sim[stat].values < perf_stat(df=observed, 
                                                stats=[stat], 
                                                dic_perf=perf_stats_dic)[
                    i[0]].values[0])
            for i, sim in df.groupby(["Strategy", "Simulation technique"],
                                     sort=False)
        }
    else:
        c = {
            i: sum(sim[stat].values > perf_stat(df=observed, 
                                                stats=[stat], 
                                                dic_perf=perf_stats_dic)[
                    i[0]].values[0])
            for i, sim in df.groupby(["Strategy", "Simulation technique"],
                                     sort=False)
        }
    return pd.DataFrame(c, index=[stat]).T


########################################################################################################################################


def p_table(right=["Max Drawdown (%)","Max TUW (Year)"],
            stat=None,
            dic=None,
            observed=None,
            perf_stats_dic=None):
    """
    appends the p-values for different statistics , investment horizons, 
    strategies and simulation method 

    params:
    -------

        - stat: a list indicating the statistic(s)
        - dic: dictionary containing the statistic(s)
        - observed: pandas dataframe containing the original returns
        - right: list indicating for which statistic more extreme 
          is right-sided

    returns:
    --------

        - a pandas multi index dataframe with the p-values

    """

    out = append_stats(stat=stat, dic=dic)
    df = p_values(df=out,
                  stat=stat[0],
                  observed=observed,
                  right=True,
                  perf_stats_dic=perf_stats_dic)
    for s in stat[1:]:
        out = append_stats(stat=s, dic=dic)
        df[s] = p_values(df=out,
                         stat=s,
                         observed=observed,
                         right=False if s in right else True,
                         perf_stats_dic=perf_stats_dic)
    df = pd.DataFrame(df)
    index = pd.MultiIndex.from_product(
        [['Mkt-RF', 'HML', 'WML', 'HML-WML'], 
        ['Normal', 'IB', 'CBB', 'SB']],
        names=['Portfolio', 'Simulatin Method'])
    columns = pd.MultiIndex.from_product([df.columns],
                                         names=['Performance statistic*'])
    return pd.DataFrame(df.values, index=index, columns=columns)