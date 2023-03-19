from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output

from aeon_analysis.presocial.dash import helpers


df = pd.read_pickle(
    Path(
        "/nfs/nhome/live/jbhagat/ProjectAeon/aeon_analysis/aeon_analysis/presocial/data"
        "/presocial_data.pkl"
    )
)
df_pretty = df.applymap(helpers.prettify_ele)

app = Dash(__name__)

# app.layout = dash_table.DataTable(
#     df.to_dict("records"), [{"name": i, "id": i} for i in df.columns]
# )
app.layout = html.Div(
    [
        html.H1("Presocial Data Dashboard"),
        html.Div(
            [
                html.H3("Data Table"),
                dash_table.DataTable(
                    data=df_pretty.to_dict("records"), 
                    columns=[{"name": i, "id": i} for i in df_pretty.columns],
                    id="table",
                ),
            ]
        ),
        html.Div(
            [
                html.H3("Weight Viz"),
                dcc.Tab(
                    label="Weight Viz",
                    children=[
                        dcc.Graph(id="weight-enter"),
                        dcc.Graph(id="weight-diff"),
                    ],
                ),
            ]
        ),
        html.Div(
            [
                html.H3("Time Viz"),
                dcc.Tab(
                    label="Time Viz",
                    children=[
                        dcc.Graph(id="duration-all"),
                        dcc.Graph(id="time-threshold-change"),
                        dcc.Graph(id="time-both-patches-sampled"),
                    ],
                ),
            ]
        ),
        html.Div(
            [
                html.H3("Hard Patch Info"),
                dcc.Tab(
                    label="Hard Patch Analysis",
                    children=[
                        dcc.Graph(id="hard-patch-id"),
                        html.Div(id="hard-patch-count"),
                    ],
                ),
            ]
        ),
        html.Div(
            [
                html.H3("Wheel Data Viz"),
                dcc.Tab(
                    label="Wheel Data Viz",
                    children=[
                        dcc.Graph(id="distance-summary"),
                    ],
                ),
            ]
        ),
        html.Div(
            [
                html.H3("Pellet Data Viz"),
                dcc.Tabs(
                    id="tabs",
                    children=[
                        dcc.Tab(
                            label="Pellet Data Viz",
                            children=[dcc.Graph(id="pellets-summary")],
                        ),
                        dcc.Tab(
                            label="Threshold Distributions",
                            children=[
                                dcc.Graph(id="threshold-dist"),
                                dcc.Graph(id="threshold-time-series"),
                                dcc.Graph(id="avg-threshold-easy-hard"),
                            ],
                        ),
                    ],
                ),
            ]
        ),
    ]
)


# Data Table
# @app.callback(
#     Output('data-table', 'figure'),
#     Input('tabs', 'value')
# )
# def update_data_table(tab):
#     if tab:
#         return go.Figure(data=[go.Table(
#             header=dict(values=list(df.columns),
#                         fill_color='paleturquoise',
#                         align='left'),
#             cells=dict(values=[df[col] for col in df.columns],
#                        fill_color='lavender',
#                        align='left'))
#         ])


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="7777", debug=True)
