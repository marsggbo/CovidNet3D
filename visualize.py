import json
import os

import numpy as np
from graphviz import Digraph
from argparse import ArgumentParser

# windows settings
gv_path = r';C:\Program Files (x86)\Graphviz2.38\bin'
print(gv_path)
os.environ['PATH'] += gv_path

OPS = [
    'irs23',
    'irs25',
    'irs43',
    'irs45',
    'identity'
]

class Visualizer(object):
    def __init__(self, file, save_type='pdf', save_path='.', OPS=OPS):
        '''
            file: the json file to load cell structure
            OPS: predefined operations
        '''
        self.file = file
        self.save_type = save_type
        self.save_path = save_path
        self.OPS = OPS
        with open(file, 'r') as f:
            self.cell = json.load(f)

    def plot_cells(self):
        normal_cell, reduce_cell = self.split_cell_json(self.cell)
        self.plot_cell(normal_cell, 'normal')
        self.plot_cell(reduce_cell, 'reduce')

    def split_cell_json(self, cell):
        f = lambda x: int(x)
        normal_cell = {key: list(map(f, cell[key])) for key in cell if key.startswith('normal')}
        reduce_cell = {key: list(map(f, cell[key])) for key in cell if key.startswith('reduce')}
        return normal_cell, reduce_cell

    def plot_cell(self, cell, cell_type='normal'):
        dot = Digraph(
                format=self.save_type,
                edge_attr=dict(fontsize='20'),
                node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2'),
                engine='dot')
        dot.body.extend(['rankdir=LR'])

        dot.node("c_{k-2}", fillcolor='darkseagreen2')
        dot.node("c_{k-1}", fillcolor='darkseagreen2')
        steps = len(cell) // 4
        for i in range(steps):
            dot.node(str(i), fillcolor='lightblue')

        for i, key in enumerate(cell):
            if i%2 == 1:
                # op
                op_index = np.argmax(cell[key])
                op = self.OPS[op_index]
                dot.edge(edge_in, edge_out, label=op, fillcolor="gray")
            else:
                input_index = np.argmax(cell[key])
                if input_index == 0:
                    edge_in = "c_{k-2}"
                elif input_index == 1:
                    edge_in = "c_{k-1}"
                else:
                    edge_in = str(input_index - 2)
                edge_out = str(i//4)
        dot.node("c_{k}", fillcolor='palegoldenrod')
        for i in range(steps):
            dot.edge(str(i), "c_{k}", fillcolor="gray")

        filename = f'{cell_type}_' + os.path.basename(self.file).replace('.json', '')
        filename = os.path.join(self.save_path, filename)
        print(filename)
        dot.render(filename, view=False, cleanup=True)
        # return dot


if __name__ == "__main__":
    parser = ArgumentParser("Visualization")
    parser.add_argument('--file_path', default='outputs', type=str)
    parser.add_argument("--save_path", default='outputs/vis', type=str)
    parser.add_argument("--save_type", default='png', type=str)

    args = parser.parse_args()
    file_path = args.file_path
    save_type = args.save_type
    save_path = args.save_path
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            file = file_path
            vis = Visualizer(file, save_type, save_path)
            vis.plot_cells()
        elif os.path.isdir(file_path):
            for file in os.listdir(file_path):
                if file.endswith(".json"):
                    file = os.path.join(file_path, file)
                    vis = Visualizer(file, save_type, save_path)
                    vis.plot_cells()
    else:
        print('Wrong file path')