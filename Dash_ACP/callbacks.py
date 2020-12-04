from dash.dependencies import Input, Output

from app import app

@app.callback(
    Output('id1', 'children'),
    Input('plot1', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('id3', 'children'),
    Input('plot', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)