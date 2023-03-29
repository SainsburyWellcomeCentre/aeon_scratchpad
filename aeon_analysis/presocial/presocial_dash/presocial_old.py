# 1) Create all figures on initialization of `app.layout` within `dcc.Graph` objects
# 2) In each tab callback, update the colors only
# 3) In a separate callback for the "Update Colors" button, update all graph objects 
# colors directly, and reload the page via javascript

from pathlib import Path

import dash
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.development.base_component import ComponentRegistry
import dash_daq as daq
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from aeon_analysis.presocial.dash import helpers


df = pd.read_pickle(
    Path(
        "/nfs/nhome/live/jbhagat/ProjectAeon/aeon_analysis/aeon_analysis/presocial/data"
        "/presocial_data.pkl"
    )
)
df_pretty = df.applymap(helpers.prettify_ele)
bg_col = "#050505"
txt_col = "#f2f2f2"
plt_bg_col = "#0d0d0d"
tab_bg_col = "#003399"
tab_txt_col = "#f2f2f2"
mrkr_sz = 14
fig_ids = ["weight_enter"]  # , "weight_diff"]

app = Dash(__name__)

app.layout = html.Div(
    id="app",
    style={"backgroundColor": bg_col, "color": txt_col},
    children=[
        html.Div(id="dummy_output", style={"display": "none"}),
        html.Div(
            id="fig_ids_storage",
            children=json.dumps(fig_ids),
            style={"display": "none"},
        ),
        html.H1("Presocial Data Dashboard"),
        html.Div(
            [
                html.Div(
                    [
                        daq.ColorPicker(
                            id="dash_bg_col_picker",
                            label=("Dashboard Background Color"),
                            value={"hex": bg_col},
                        ),
                    ],
                    style={"display": "inline-block", "margin-right": "20px"},
                ),
                html.Div(
                    [
                        daq.ColorPicker(
                            id="dash_txt_col_picker",
                            label=("Dashboard Text Color"),
                            value={"hex": txt_col},
                        ),
                    ],
                    style={"display": "inline-block", "margin-right": "20px"},
                ),
                html.Div(
                    [
                        daq.ColorPicker(
                            id="plt_bg_col_picker",
                            label=("Plot Background Color"),
                            value={"hex": plt_bg_col},
                        ),
                    ],
                    style={"display": "inline-block", "margin-right": "20px"},
                ),
                html.Div(
                    [
                        html.Button("Update Colors", id="col_button"),
                    ],
                    style={"display": "flex", "justify-content": "left"},
                ),
            ]
        ),
        html.Div(
            children=[
                html.H3("Data Table"),
                dash_table.DataTable(
                    data=df_pretty.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_pretty.columns],
                    id="table",
                    style_header={"backgroundColor": bg_col, "color": txt_col},
                    style_cell={"backgroundColor": plt_bg_col, "color": txt_col},
                ),
            ],
        ),
        html.Div(
            [
                html.H3("Weight Viz"),
                dcc.Tabs(
                    id="weight_tabs",
                    children=[
                        dcc.Tab(
                            label="Weight Viz",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(id="weight_enter"),
                                dcc.Graph(id="weight_diff"),
                            ],
                        ),
                        dcc.Tab(
                            label="Weight Viz Summary",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(id="weight_enter_summary"),
                                dcc.Graph(id="weight_diff_summary"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            [
                html.H3("Time Viz"),
                dcc.Tabs(
                    id="time_tabs",
                    children=[
                        dcc.Tab(
                            label="Time Viz",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(id="session_dur"),
                                dcc.Graph(id="dur_post_thresh_change"),
                                dcc.Graph(id="dur_pre_both_patches_sampled"),
                            ],
                        ),
                        dcc.Tab(
                            label="Time Viz Summary",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(id="session_dur_summary"),
                                dcc.Graph(id="dur_post_thresh_change_summary"),
                                dcc.Graph(id="dur_pre_both_patches_sampled_summary"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            [
                html.H3("Hard Patch Info"),
                dcc.Tab(
                    label="Hard Patch Analysis",
                    children=[
                        dcc.Graph(id="hard_patch_id"),
                        html.Div(id="hard_patch_count"),
                    ],
                ),
            ],
        ),
        html.Div(
            [
                html.H3("Wheel Viz"),
                dcc.Tabs(
                    id="wheel_tabs",
                    children=[
                        dcc.Tab(
                            label="Wheel Viz",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(id="wheel_distance_spun"),
                            ],
                        ),
                        dcc.Tab(
                            label="Wheel Viz Summary",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(id="wheel_distance_spun_summary"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            [
                html.H3("Pellet Data Viz"),
                dcc.Tabs(
                    id="pellet_data_tabs",
                    children=[
                        dcc.Tab(
                            label="Pellets Obtained Viz",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[dcc.Graph(id="pellets_obtained")],
                        ),
                        dcc.Tab(
                            label="Pellets Obtained Viz Summary",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[dcc.Graph(id="pellets_obtained_summary")],
                        ),
                        dcc.Tab(
                            label="Probabilistic Pellet Thresholds Viz",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Tabs(
                                    id="prob_pel_thresh_tabs",
                                    children=[
                                        dcc.Tab(
                                            label="Probabilistic Pellet Thresholds"
                                            "Time_Series",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(id="prob_pel_thresh_time")
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Probabilistic Pellet Thresholds"
                                            "Distributions",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(id="prob_pel_thresh_dist")
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Probabilistic Pellet Thresholds"
                                            "Summary",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(id="prob_pel_thresh_summary")
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# app.clientside_callback(
#     ClientsideFunction(namespace="dash_clientside", function_name="updateColors"),
#     Output("dummy_output", "children"),
#     Input("col_button", "n_clicks"),
#     State("dash_bg_col_picker", "value"),
#     State("dash_txt_col_picker", "value"),
#     State("plt_bg_col_picker", "value"),
#     State("plot_ids_storage", "children"),
# )

def find_graphs(layout):
    graphs = []

    if isinstance(layout, list):
        for element in layout:
            graphs.extend(find_graphs(element))
    elif isinstance(layout, dict):
        for key, value in layout.items():
            graphs.extend(find_graphs(value))
    elif hasattr(layout, "children"):
        graphs.extend(find_graphs(layout.children))
    elif isinstance(layout, dcc.Graph):
        graphs.append(layout)

    return graphs

# outputs = [Output(fig, "figure") for fig in fig_ids]
# outputs.insert(0, Output("app", "style"))

@app.callback(
    # Output("app", "style"),
    Output("dummy_output", "style"),
    Input("col_button", "n_clicks"),
    State("dash_bg_col_picker", "value"),
    State("dash_txt_col_picker", "value"),
    State("plt_bg_col_picker", "value"),
    State("fig_ids_storage", "children"),
)
def update_colors(n_clicks, bg_col, txt_col, plt_bg_col, fig_ids):
    if n_clicks:
        import ipdb
        ipdb.set_trace()
        global app
        graphs = find_graphs(app.layout)
        for graph in graphs:
            graph.figure["layout"]["paper_bgcolor"] = bg_col
            graph.figure["layout"]["font"] = {"color": txt_col}
            graph.figure["layout"]['plot_bgcolor'] = plt_bg_col
        html.Script(src="assets/reload_page.js")
        # fig_ids = json.loads(fig_ids)
        # for plot_id in fig_ids:
        #     plot_class = None
        #     for cls in ComponentRegistry.registry:
        #         if cls.__name__ == plot_id:
        #             plot_class = cls
        #             break
        #     if plot_class is not None:
        #         ComponentRegistry.register(
        #             plot_class(
        #                 id=plot_id,
        #                 style={
        #                     "backgroundColor": plt_bg_col["hex"],
        #                     "color": txt_col["hex"],
        #                 },
        #                 config={"displayModeBar": False},
        #                 figure={
        #                     "layout": {
        #                         "plot_bgcolor": plt_bg_col["hex"],
        #                         "paper_bgcolor": bg_col["hex"],
        #                     }
        #                 },
        #             ),
        #             plot_id,
        #         )
    #     return {
    #         "backgroundColor": bg_col,
    #         "color": txt_col,
    #     }
    # import ipdb
    # ipdb.set_trace()
    # return (
    #     {
    #         "backgroundColor": bg_col["hex"],
    #         "color": txt_col["hex"],
    #     },
    #     {
    #         "layout": {
    #             "paper_bgcolor": bg_col["hex"],
    #             "plot_bgcolor": plt_bg_col["hex"],
    #             "font": txt_col["hex"],
    #         }
    #     },
    # )
    # app.layout.style["backgroundColor"] = bg_col["hex"]
    # app.layout.style["color"] = txt_col["hex"]
    # return


# 1a) Weight at enter time
@app.callback(Output("weight_enter", "figure"), Input("weight_tabs", "value"))
def update_weight_enter(tab):
    if tab:
        fig = px.line(df, x="enter", y="weight_enter", color="id", markers=True)
        fig.update_traces(marker={"size": mrkr_sz})
        fig.update_layout(
            title="Weight Enter",
            plot_bgcolor=plt_bg_col,
            paper_bgcolor=bg_col,
            font={"color": txt_col},
        )
        return fig


# 1b) Exit weight - Entry weight
@app.callback(Output("weight_diff", "figure"), Input("weight_tabs", "value"))
def update_weight_diff(tab):
    if tab:
        weight_diff = df["weight_exit"] - df["weight_enter"]
        fig = px.line(df, x="enter", y=weight_diff, color="id", markers=True)
        fig.update_traces(marker={"size": mrkr_sz})
        fig.update_layout(
            title="Weight Diff (Exit - Enter)",
            plot_bgcolor=plt_bg_col,
            paper_bgcolor=bg_col,
            font={"color": txt_col},
        )
        return fig

    
@app.callback(Output("weight_enter_summary", "figure"), Input("weight_tabs", "value"))
def update_weight_enter(tab):
    if tab:
        pass


@app.callback(Output("weight_diff_summary", "figure"), Input("weight_tabs", "value"))
def update_weight_diff(tab):
    if tab:
        pass


# # 2a) Session duration for all subjects and each subject
# @app.callback(
#     Output('session-duration', 'figure'),
#     Input('tabs', 'value')
# )
# def update_session_duration(tab):
#     if tab:
#         fig = px.line(df, x="enter", y="duration", color="id", markers=True)
#         fig.update_layout(title='Session Duration by Session and Subject')
#         return fig

# # 2b) Time until threshold change for all subjects and each subject
# @app.callback(
#     Output('time-threshold-change', 'figure'),
#     Input('tabs', 'value')
# )
# def update_time_threshold_change(tab):
#     if tab:
#         fig = px.line(df, x=df.index, y='time until the distance thresholds changed', color='subject id', markers=True)
#         fig.update_layout(title='Time until Threshold Change by Session and Subject')
#         return fig

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="7777", debug=True)
