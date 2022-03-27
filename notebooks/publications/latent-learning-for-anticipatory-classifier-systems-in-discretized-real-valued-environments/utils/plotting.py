import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from pylab import cm

ALGS_NO = 7

# Common color palette
palette = cm.get_cmap('tab10', ALGS_NO)

COLORS = {
    'acs': palette(0),
    'acs2': palette(1),
    'acs2_oiq': palette(2),
    'acs2_ga': palette(3),
    'acs2_ga_oiq': palette(4),
    'yacs': palette(5),
    'dynaq': palette(6),
}


def plot_comparison(metrics):
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))

    # Line styles
    # marker = metrics.index.get_level_values(1).max() / 10
    marker = 10
    mark_every = (np.linspace(0, marker, ALGS_NO) + marker).astype(int)
    line_props = {
        'linewidth': 3,
        'markersize': 9
    }

    acs_line_props = { 'label': 'ACS', 'color': COLORS['acs'], 'marker': 'x', 'markevery': mark_every[0], **line_props }
    acs2_line_props = {'label': 'ACS2', 'color': COLORS['acs2'], 'marker': 'v', 'markevery': mark_every[1], **line_props }
    acs2_oiq_line_props = { 'label': 'ACS2_OIQ', 'color': COLORS['acs2_oiq'], 'marker': 'v', 'markevery': mark_every[2], **line_props }
    acs2_ga_line_props = { 'label': 'ACS2_GA', 'color': COLORS['acs2_ga'], 'marker': 's', 'markevery': mark_every[3], **line_props }
    acs2_ga_oiq_line_props = { 'label': 'ACS2_GA_OIQ', 'color': COLORS['acs2_ga_oiq'], 'marker': 's', 'markevery': mark_every[4], **line_props }
    yacs_line_props = { 'label': 'YACS', 'color': COLORS['yacs'], 'marker': 'o', 'markevery': mark_every[5], **line_props }
    dynaq_line_props = { 'label': 'DynaQ', 'color': COLORS['dynaq'], 'marker': 'D', 'markevery': mark_every[6], **line_props }

    metrics['knowledge_100'] = metrics['knowledge'] * 100
    metrics['generalization_100'] = metrics['generalization'] * 100

    # Population
    metrics.loc['acs']['population'].plot(ax=axs[0, 0], **acs_line_props)
    metrics.loc['acs2']['population'].plot(ax=axs[0, 0], **acs2_line_props)
    metrics.loc['acs2_oiq']['population'].plot(ax=axs[0, 0], **acs2_oiq_line_props)
    metrics.loc['acs2_ga']['population'].plot(ax=axs[0, 0], **acs2_ga_line_props)
    metrics.loc['acs2_ga_oiq']['population'].plot(ax=axs[0, 0], **acs2_ga_oiq_line_props)
    metrics.loc['yacs']['population'].plot(ax=axs[0, 0], **yacs_line_props)
    metrics.loc['dynaq']['population'].plot(ax=axs[0, 0], **dynaq_line_props)
    axs[0, 0].set_title('Population size')
    axs[0, 0].set_ylabel('Number of rules/classifiers')
#     axs[0, 0].legend(loc='best', frameon=False)

    # Knowledge
    axs[0, 1].set_title('Knowledge')
    metrics.loc['acs']['knowledge_100'].plot(ax=axs[0, 1], **acs_line_props)
    metrics.loc['acs2']['knowledge_100'].plot(ax=axs[0, 1], **acs2_line_props)
    metrics.loc['acs2_oiq']['knowledge_100'].plot(ax=axs[0, 1], **acs2_oiq_line_props)
    metrics.loc['acs2_ga']['knowledge_100'].plot(ax=axs[0, 1], **acs2_ga_line_props)
    metrics.loc['acs2_ga_oiq']['knowledge_100'].plot(ax=axs[0, 1], **acs2_ga_oiq_line_props)
    metrics.loc['yacs']['knowledge_100'].plot(ax=axs[0, 1], **yacs_line_props)
    metrics.loc['dynaq']['knowledge_100'].plot(ax=axs[0, 1], **dynaq_line_props)
#     axs[0, 1].legend(loc='lower right', frameon=False)
    axs[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Generalization
    axs[1, 0].set_title('Generalization')
    metrics.loc['acs']['generalization_100'].plot(ax=axs[1, 0], **acs_line_props)
    metrics.loc['acs2']['generalization_100'].plot(ax=axs[1, 0], **acs2_line_props)
    metrics.loc['acs2_oiq']['generalization_100'].plot(ax=axs[1, 0], **acs2_oiq_line_props)
    metrics.loc['acs2_ga']['generalization_100'].plot(ax=axs[1, 0], **acs2_ga_line_props)
    metrics.loc['acs2_ga_oiq']['generalization_100'].plot(ax=axs[1, 0], **acs2_ga_oiq_line_props)
    metrics.loc['yacs']['generalization_100'].plot(ax=axs[1, 0], **yacs_line_props)
    metrics.loc['dynaq']['generalization_100'].plot(ax=axs[1, 0], **dynaq_line_props)
#     axs[1, 0].legend(loc='best', frameon=False)
    axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Trial time
    times = metrics.groupby('agent')['time'].mean().to_dict()

    labels = ['ACS', 'ACS2', 'ACS2_OIQ', 'ACS2_GA', 'ACS2_GA_OIQ', 'YACS', 'DynaQ']
    values = [times['acs'], times['acs2'], times['acs2_oiq'], times['acs2_ga'], times['acs2_ga_oiq'], times['yacs'], times['dynaq']]
    colors = [COLORS['acs'], COLORS['acs2'], COLORS['acs2_oiq'], COLORS['acs2_ga'], COLORS['acs2_ga_oiq'], COLORS['yacs'], COLORS['dynaq']]

    axs[1, 1].bar(labels, values, color=colors)
    axs[1, 1].set_xticklabels(labels, rotation=60)
    axs[1, 1].set_title('Average trial time')
    axs[1, 1].set_ylabel('Seconds [s]')

    # create some space below the plots by increasing the bottom-value
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.16)

    # Global legend
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(values), loc='lower center', prop={'size': 23})


def plot_bayes_comparison(*dfs):
    algs = COLORS.keys()

    fig = plt.figure(figsize=(22, 16))

    for i, df in enumerate(dfs):
        attributes = df.columns.to_list()
        theta = _radar_factory(len(attributes), frame='circle')

        ax = fig.add_subplot(2, 2, i+1, projection='radar')
        ax.get_yaxis().set_ticklabels([])
        ax.set_varlabels(attributes)
        ax.set_title(df.attrs['name'], pad=50)

        for i, alg in enumerate(algs):
            v = df.loc[alg].to_list()
            ax.plot(theta, v, color=COLORS[alg])
            ax.fill(theta, v, facecolor=COLORS[alg], alpha=0.25)

        # realign theta labels
        for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
            theta = np.pi/2 - theta
            y, x = np.cos(theta), np.sin(theta)
            if x >= 0.1:
                label.set_horizontalalignment('left')
            if x <= -0.1:
                label.set_horizontalalignment('right')
            if y >= 0.5:
                label.set_verticalalignment('bottom')
            if y <= -0.5:
                label.set_verticalalignment('top')

    fig.suptitle('Bayesian Estimation of metrics', fontsize=28)
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.85,
        # left=0.0, right=0.75,
        bottom=0.1,
        wspace=-0.25,
        hspace=0.35
    )

    # build legend
    fig.legend([alg.upper() for alg in algs], loc='lower center', ncol=len(algs), labelspacing=0.5, prop={'size': 23})


# def plot_bayes_radar(df: pd.DataFrame, env_name):
#     attributes = df.columns.to_list()
#     theta = _radar_factory(len(attributes), frame='circle')
#
#     algs = COLORS.keys()
#
#     fig = plt.figure(figsize=(12, 12), dpi=80)
#     ax = fig.add_subplot(111, projection='radar')
#     ax.get_yaxis().set_ticklabels([])
#     ax.set_varlabels(attributes)
#     ax.set_title(f'Bayesian Estimation for {env_name}', pad=50)
#
#     for i, alg in enumerate(algs):
#         v = df.loc[alg].to_list()
#         ax.plot(theta, v, color=COLORS[alg])
#         ax.fill(theta, v, facecolor=COLORS[alg], alpha=0.25)
#
#     # realign theta labels
#     for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
#         theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
#         theta = np.pi/2 - theta
#         y, x = np.cos(theta), np.sin(theta)
#         if x >= 0.1:
#             label.set_horizontalalignment('left')
#         if x <= -0.1:
#             label.set_horizontalalignment('right')
#         if y >= 0.5:
#             label.set_verticalalignment('bottom')
#         if y <= -0.5:
#             label.set_verticalalignment('top')
#
#     # build legend
#     ax.legend([alg.upper() for alg in algs],
#               loc='lower center',
#               ncol=len(algs),
#               labelspacing=0.5,
#               fontsize='small',
#               bbox_to_anchor=(0.5, -0.13),)
#
#     fig.tight_layout()


def _radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, size='small')

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)

    return theta
