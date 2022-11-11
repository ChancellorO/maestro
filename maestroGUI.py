import PySimpleGUI as sg
from astropy import units
from astropy import constants

sg.theme('Default1')

operator_lst = [
    ' * ', ' / '
]

layout = [
    [sg.Text('Value'), sg.InputText(key='Value', size=10), sg.Text('Unit'), sg.InputText(key='Unit', size=10), sg.Button('add')],
    [sg.Column([[]], key='-Column-')],
    [ sg.Button(x) for x in operator_lst ],
    [sg.Text('', key='Text'), sg.Button('Eval')],
    [sg.Text('Error, already an existing value - unit pair', key='Error', visible=False)]
]

window = sg.Window('Maestro', layout, finalize = True, size =(480, 480), resizable=True)


arr = []

#set of tuples
lst = []

counter = 0

def new_layout(value, unit, count):
    lst.append((value,unit))
    return [[sg.Text(value + ' ' + unit, key='Text' + str(count)), sg.Button('Ok', key='Ok' + str(count))]]

def parser(expression):
    print(expression)
    


while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    window['Error'].update(visible=False)
    if event == 'add':
        if (values['Value'].strip(), values['Unit'].strip()) in lst:
            window['Error'].update(visible=True)
            continue
        window.extend_layout(window['-Column-'], new_layout(values['Value'].strip(), values['Unit'].strip(), counter))
        counter += 1
    if event[:2] == 'Ok':
        text = window['Text']
        window['Text'].update(text.get() + window['Text' + event[2]].get())
    elif event in operator_lst:
        text = window['Text']
        window['Text'].update(text.get() + event)
    elif event == 'Eval':
        parser(window['Text'].get())

    print(lst)
window.close()