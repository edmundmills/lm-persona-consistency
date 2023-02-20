from pathlib import Path

from persona_consistency.plots import MetricsPlotter


def main():
    plotter = MetricsPlotter(
        task_names=[
            'subscribes-to-deontology',
            'conscientiousness',
        ],
        model_names=[
            'davinci',
            'davinci-instruct-beta',
            'text-davinci-001',
            'text-davinci-002',
            'text-davinci-003'
        ],
        data_path=Path('results') / '2_tasks'
    )
    plotter.plot_all()

if __name__ == '__main__':
    main()
