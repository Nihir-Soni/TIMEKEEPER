import PySimpleGUI as sg
layout = [[sg.Column([[sg.Text(f'Line {i}')] for i in range(100)], scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)]]
window = sg.Window('Test', layout, resizable=True, size=(400, 300))
event, values = window.read()
window.close()
